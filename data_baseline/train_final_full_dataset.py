"""
train_final_full_dataset.py - Entraînement final sur train + validation combinés

Ce script est utilisé une fois que tu as trouvé les meilleurs hyperparamètres.
Il fusionne train + validation pour entraîner le modèle final avant soumission Kaggle.

Usage:
    python train_final_full_dataset.py \
        --data_dir data_baseline/data \
        --loss mse \
        --epochs 50 \
        --out_ckpt model_final_full.pt
"""

import os
import argparse
import pickle
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from torch_geometric.data import Batch
from torch_geometric.nn import GPSConv, GINEConv, global_add_pool

from data_utils import load_id2emb, x_map, e_map


# =========================================================
# CONFIG
# =========================================================
DEFAULT_CONFIG = {
    'batch_size': 32,
    'epochs': 50,
    'lr': 5e-4,
    'hidden_dim': 256,
    'num_layers': 4,
    'num_heads': 4,
    'dropout': 0.1,
    'temperature': 0.07,
    'margin': 0.3,
}

ATOM_FEATURE_DIMS = [
    len(x_map['atomic_num']), len(x_map['chirality']), len(x_map['degree']),
    len(x_map['formal_charge']), len(x_map['num_hs']), len(x_map['num_radical_electrons']),
    len(x_map['hybridization']), len(x_map['is_aromatic']), len(x_map['is_in_ring']),
]
BOND_FEATURE_DIMS = [
    len(e_map['bond_type']), len(e_map['stereo']), len(e_map['is_conjugated']),
]


# =========================================================
# DATASET pour données fusionnées
# =========================================================
class CombinedGraphDataset(Dataset):
    """Dataset qui combine train + validation."""
    
    def __init__(self, graph_paths: list, emb_dicts: list):
        """
        Args:
            graph_paths: Liste de chemins vers les fichiers .pkl
            emb_dicts: Liste de dictionnaires {id: embedding}
        """
        self.graphs = []
        self.embeddings = []
        self.ids = []
        
        # Fusionner les embeddings
        combined_emb = {}
        for emb_dict in emb_dicts:
            combined_emb.update(emb_dict)
        
        # Charger et fusionner les graphes
        for path in graph_paths:
            print(f"Loading: {path}")
            with open(path, 'rb') as f:
                graphs = pickle.load(f)
            
            for g in graphs:
                if g.id in combined_emb:
                    self.graphs.append(g)
                    self.embeddings.append(combined_emb[g.id])
                    self.ids.append(g.id)
        
        print(f"Combined dataset: {len(self.graphs)} samples")
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx], self.embeddings[idx]


def collate_fn(batch):
    graphs, embs = zip(*batch)
    batch_graphs = Batch.from_data_list(list(graphs))
    batch_embs = torch.stack([torch.tensor(e) if not isinstance(e, torch.Tensor) else e for e in embs])
    return batch_graphs, batch_embs


# =========================================================
# ENCODERS
# =========================================================
class AtomEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, hidden_dim) for dim in ATOM_FEATURE_DIMS
        ])
    def forward(self, x):
        return sum(emb(x[:, i]) for i, emb in enumerate(self.embeddings))


class BondEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, hidden_dim) for dim in BOND_FEATURE_DIMS
        ])
    def forward(self, edge_attr):
        return sum(emb(edge_attr[:, i]) for i, emb in enumerate(self.embeddings))


# =========================================================
# MODEL: GPS
# =========================================================
class MolGNN(nn.Module):
    def __init__(self, hidden_dim=256, out_dim=768, num_layers=4, num_heads=4, dropout=0.1):
        super().__init__()
        
        self.atom_encoder = AtomEncoder(hidden_dim)
        self.bond_encoder = BondEncoder(hidden_dim)
        
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            local_nn = nn.Sequential(
                nn.Linear(hidden_dim, 2 * hidden_dim),
                nn.BatchNorm1d(2 * hidden_dim),
                nn.ReLU(),
                nn.Linear(2 * hidden_dim, hidden_dim),
            )
            local_conv = GINEConv(local_nn, train_eps=True, edge_dim=hidden_dim)
            gps_conv = GPSConv(
                channels=hidden_dim,
                conv=local_conv,
                heads=num_heads,
                dropout=dropout,
                attn_type='multihead',
            )
            self.convs.append(gps_conv)
        
        self.pool = global_add_pool
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, batch: Batch):
        h = self.atom_encoder(batch.x)
        edge_attr = self.bond_encoder(batch.edge_attr)
        
        for conv in self.convs:
            h = conv(h, batch.edge_index, batch.batch, edge_attr=edge_attr)
        
        g = self.pool(h, batch.batch)
        out = self.proj(g)
        return F.normalize(out, dim=-1)


# =========================================================
# LOSS FUNCTIONS
# =========================================================
def mse_loss(graph_emb, text_emb, **kwargs):
    return F.mse_loss(graph_emb, text_emb)


def info_nce_loss(graph_emb, text_emb, temperature=0.07, **kwargs):
    text_emb = F.normalize(text_emb, dim=-1)
    logits = graph_emb @ text_emb.T / temperature
    labels = torch.arange(len(logits), device=logits.device)
    loss_g2t = F.cross_entropy(logits, labels)
    loss_t2g = F.cross_entropy(logits.T, labels)
    return (loss_g2t + loss_t2g) / 2


def triplet_loss(graph_emb, text_emb, margin=0.3, **kwargs):
    text_emb = F.normalize(text_emb, dim=-1)
    sims = graph_emb @ text_emb.T
    B = graph_emb.size(0)
    device = graph_emb.device
    
    positive_sims = sims.diag()
    mask = torch.eye(B, device=device).bool()
    sims_masked = sims.masked_fill(mask, float('-inf'))
    hard_negative_sims, _ = sims_masked.max(dim=1)
    
    losses = F.relu(hard_negative_sims - positive_sims + margin)
    return losses.mean()


LOSS_FUNCTIONS = {
    'mse': mse_loss,
    'infonce': info_nce_loss,
    'triplet': triplet_loss,
}


# =========================================================
# TRAINING
# =========================================================
def train_epoch(model, loader, optimizer, device, loss_fn, loss_kwargs):
    model.train()
    total_loss, total = 0.0, 0

    pbar = tqdm(loader, desc="Training", leave=False)
    
    for graphs, text_emb in pbar:
        graphs = graphs.to(device)
        text_emb = text_emb.to(device)
        
        graph_emb = model(graphs)
        loss = loss_fn(graph_emb, text_emb, **loss_kwargs)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * graphs.num_graphs
        total += graphs.num_graphs
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data_baseline/data')
    parser.add_argument('--train_emb', type=str, default=None)
    parser.add_argument('--val_emb', type=str, default=None)
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'infonce', 'triplet'])
    parser.add_argument('--epochs', type=int, default=DEFAULT_CONFIG['epochs'])
    parser.add_argument('--batch_size', type=int, default=DEFAULT_CONFIG['batch_size'])
    parser.add_argument('--lr', type=float, default=DEFAULT_CONFIG['lr'])
    parser.add_argument('--temperature', type=float, default=DEFAULT_CONFIG['temperature'])
    parser.add_argument('--margin', type=float, default=DEFAULT_CONFIG['margin'])
    parser.add_argument('--hidden_dim', type=int, default=DEFAULT_CONFIG['hidden_dim'])
    parser.add_argument('--num_layers', type=int, default=DEFAULT_CONFIG['num_layers'])
    parser.add_argument('--out_ckpt', type=str, default='model_final_full.pt')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 70)
    print("FINAL TRAINING ON TRAIN + VALIDATION")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Loss: {args.loss}")
    print(f"Config: epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}")
    
    # Paths
    data_dir = Path(args.data_dir)
    train_graphs_path = str(data_dir / "train_graphs.pkl")
    val_graphs_path = str(data_dir / "validation_graphs.pkl")
    
    # Default embeddings paths
    if args.train_emb is None:
        # Cherche le fichier ChemBERT
        candidates = [
            data_dir / "train_embeddings_ChemBERT.csv",
            data_dir / "train_embeddings.recobo_chemical-bert-uncased.cls.csv",
        ]
        for c in candidates:
            if c.exists():
                args.train_emb = str(c)
                break
    
    if args.val_emb is None:
        candidates = [
            data_dir / "validation_embeddings_ChemBERT.csv",
            data_dir / "validation_embeddings.recobo_chemical-bert-uncased.cls.csv",
        ]
        for c in candidates:
            if c.exists():
                args.val_emb = str(c)
                break
    
    print(f"Train embeddings: {args.train_emb}")
    print(f"Val embeddings: {args.val_emb}")
    
    # Load embeddings
    train_emb = load_id2emb(args.train_emb)
    val_emb = load_id2emb(args.val_emb)
    emb_dim = len(next(iter(train_emb.values())))
    
    # Combined dataset
    dataset = CombinedGraphDataset(
        graph_paths=[train_graphs_path, val_graphs_path],
        emb_dicts=[train_emb, val_emb]
    )
    
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        drop_last=(args.loss != 'mse')  # Pour InfoNCE/Triplet
    )
    
    # Model
    model = MolGNN(
        hidden_dim=args.hidden_dim,
        out_dim=emb_dim,
        num_layers=args.num_layers
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Loss function
    loss_fn = LOSS_FUNCTIONS[args.loss]
    loss_kwargs = {'temperature': args.temperature, 'margin': args.margin}
    
    # Training loop - PAS d'early stopping car pas de validation set
    LOG_FREQ = 5
    
    for epoch in range(args.epochs):
        loss = train_epoch(model, loader, optimizer, device, loss_fn, loss_kwargs)
        scheduler.step()
        
        if (epoch + 1) % LOG_FREQ == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.epochs} | loss={loss:.6f}")
    
    # Save final model
    torch.save(model.state_dict(), args.out_ckpt)
    print(f"\n✅ Model saved to: {args.out_ckpt}")
    print("Ready for test set prediction!")


if __name__ == "__main__":
    main()
