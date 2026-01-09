"""
train_gps_chembert_mse.py - GPS + ChemicalBERT + MSE Loss

Utilise les embeddings pré-calculés de ChemicalBERT (recobo/chemical-bert-uncased)
C'est le meilleur Text Encoder selon le benchmark d'Oualid.
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch_geometric.data import Batch
from torch_geometric.nn import GPSConv, GINEConv, global_add_pool

from data_utils import load_id2emb, PreprocessedGraphDataset, collate_fn, x_map, e_map


# =========================================================
# CONFIG (peut être overridé par argparse)
# =========================================================
DEFAULT_CONFIG = {
    'batch_size': 32,
    'epochs': 50,
    'lr': 5e-4,
    'hidden_dim': 256,
    'num_layers': 4,
    'num_heads': 4,
    'dropout': 0.1,
    'patience': 5,
}

# Feature dimensions
ATOM_FEATURE_DIMS = [
    len(x_map['atomic_num']), len(x_map['chirality']), len(x_map['degree']),
    len(x_map['formal_charge']), len(x_map['num_hs']), len(x_map['num_radical_electrons']),
    len(x_map['hybridization']), len(x_map['is_aromatic']), len(x_map['is_in_ring']),
]
BOND_FEATURE_DIMS = [
    len(e_map['bond_type']), len(e_map['stereo']), len(e_map['is_conjugated']),
]


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
        g = self.proj(g)
        return F.normalize(g, dim=-1)


# =========================================================
# TRAINING
# =========================================================
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, total = 0.0, 0

    pbar = tqdm(loader, desc="Training", leave=False)
    
    for graphs, text_emb in pbar:
        graphs = graphs.to(device)
        text_emb = text_emb.to(device)
        
        graph_emb = model(graphs)
        text_emb = F.normalize(text_emb, dim=-1)
        loss = F.mse_loss(graph_emb, text_emb)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * graphs.num_graphs
        total += graphs.num_graphs
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / total


@torch.no_grad()
def eval_retrieval(model, data_path, emb_dict, device):
    model.eval()
    ds = PreprocessedGraphDataset(data_path, emb_dict)
    dl = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
    
    all_mol, all_txt = [], []
    for graphs, text_emb in dl:
        graphs, text_emb = graphs.to(device), text_emb.to(device)
        all_mol.append(model(graphs))
        all_txt.append(F.normalize(text_emb, dim=-1))
    
    all_mol, all_txt = torch.cat(all_mol), torch.cat(all_txt)
    sims = all_txt @ all_mol.t()
    ranks = sims.argsort(dim=-1, descending=True)
    
    N = all_txt.size(0)
    correct = torch.arange(N, device=sims.device)
    pos = (ranks == correct.unsqueeze(1)).nonzero()[:, 1] + 1
    
    mrr = (1.0 / pos.float()).mean().item()
    results = {"MRR": mrr}
    for k in (1, 5, 10):
        results[f"R@{k}"] = (pos <= k).float().mean().item()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--train_emb', type=str, default=None,
                        help='Path to train embeddings CSV (ChemicalBERT)')
    parser.add_argument('--val_emb', type=str, default=None,
                        help='Path to validation embeddings CSV')
    parser.add_argument('--epochs', type=int, default=DEFAULT_CONFIG['epochs'])
    parser.add_argument('--batch_size', type=int, default=DEFAULT_CONFIG['batch_size'])
    parser.add_argument('--lr', type=float, default=DEFAULT_CONFIG['lr'])
    parser.add_argument('--hidden_dim', type=int, default=DEFAULT_CONFIG['hidden_dim'])
    parser.add_argument('--num_layers', type=int, default=DEFAULT_CONFIG['num_layers'])
    parser.add_argument('--out_ckpt', type=str, default='model_gps_chembert_mse.pt')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 70)
    print("GPS + ChemicalBERT + MSE Loss")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Config: epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}")
    print(f"Architecture: hidden={args.hidden_dim}, layers={args.num_layers}")
    
    # Paths
    data_dir = Path(args.data_dir)
    train_graphs = str(data_dir / "train_graphs.pkl")
    val_graphs = str(data_dir / "validation_graphs.pkl")
    
    # Default to ChemicalBERT embeddings if not specified
    if args.train_emb is None:
        args.train_emb = str(data_dir / "train_embeddings.recobo_chemical-bert-uncased.cls.csv")
    if args.val_emb is None:
        args.val_emb = str(data_dir / "validation_embeddings.recobo_chemical-bert-uncased.cls.csv")
    
    print(f"Train embeddings: {args.train_emb}")
    print(f"Val embeddings: {args.val_emb}")
    
    # Load data
    train_emb = load_id2emb(args.train_emb)
    val_emb = load_id2emb(args.val_emb) if os.path.exists(args.val_emb) else None
    emb_dim = len(next(iter(train_emb.values())))
    
    train_ds = PreprocessedGraphDataset(train_graphs, train_emb)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    # Model
    model = MolGNN(
        hidden_dim=args.hidden_dim,
        out_dim=emb_dim,
        num_layers=args.num_layers
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_mrr = 0.0
    patience_counter = 0
    patience = DEFAULT_CONFIG['patience']

    # Fréquence d'affichage (toutes les 5 époques)
    LOG_FREQ = 5
    
    for epoch in range(args.epochs):
        loss = train_epoch(model, train_dl, optimizer, device)
        val_scores = eval_retrieval(model, val_graphs, val_emb, device) if val_emb else {}
        print(f"Epoch {epoch+1}/{args.epochs} | loss={loss:.6f} | {val_scores}")
        scheduler.step()

        should_log = (epoch == 0) or ((epoch + 1) % LOG_FREQ == 0)
        if should_log:
            print(f"Epoch {epoch+1}/{args.epochs} | loss={loss:.6f} | {val_scores}")
        
        current_mrr = val_scores.get('MRR', 0)
        if current_mrr > best_mrr:
            best_mrr = current_mrr
            patience_counter = 0
            torch.save(model.state_dict(), args.out_ckpt)
            print(f"  → New best MRR: {best_mrr:.4f} | saved: {args.out_ckpt}")
        else:
            patience_counter += 1
            print(f"  → No improvement ({patience_counter}/{patience})")
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    print(f"\nDone. Best MRR: {best_mrr:.4f}")


if __name__ == "__main__":
    main()
