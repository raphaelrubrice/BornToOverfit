"""
train_gps_chembert_triplet.py - GPS + ChemicalBERT + Triplet Loss (Compatible .pt et .csv)

Triplet Loss = Pour chaque (ancre, positif, négatif), on veut :
    distance(ancre, positif) + margin < distance(ancre, négatif)

On utilise des hard negatives : les textes les plus proches mais incorrects.
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
# CONFIG
# =========================================================
DEFAULT_CONFIG = {
    'batch_size': 64,
    'epochs': 50,
    'lr': 5e-4,
    'hidden_dim': 256,
    'num_layers': 4,
    'num_heads': 4,
    'dropout': 0.1,
    'patience': 5,
    'margin': 0.3,  # Marge pour triplet loss
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
# HELPER: CHARGEMENT INTELLIGENT (.pt ou .csv)
# =========================================================
def smart_load_embeddings(path):
    """Charge les embeddings depuis un CSV (lent) ou un PT (rapide)."""
    path = str(path)
    if path.endswith('.pt'):
        print(f"⚡ Chargement rapide depuis {path}...")
        data = torch.load(path)
        ids = data['ids']
        embs = data['embeddings']
        return {i: embs[idx] for idx, i in enumerate(ids)}
    else:
        # Fallback sur la méthode CSV classique
        return load_id2emb(path)

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
# TRIPLET LOSS with Hard Negative Mining
# =========================================================
def triplet_loss_hard_negative(graph_emb, text_emb, margin=0.3):
    """
    Triplet Loss avec hard negative mining.
    """
    # Normaliser
    graph_emb = F.normalize(graph_emb, dim=-1)
    text_emb = F.normalize(text_emb, dim=-1)
    
    # Matrice de similarités [B, B]
    sims = graph_emb @ text_emb.T
    
    B = graph_emb.size(0)
    device = graph_emb.device
    
    # Pour chaque graphe i, le positif est texte i
    positive_sims = sims.diag()  # [B]
    
    # Pour les négatifs, on prend le texte le plus similaire qui n'est pas le bon
    mask = torch.eye(B, device=device).bool()
    sims_masked = sims.masked_fill(mask, float('-inf'))
    
    # Hard negative = le plus similaire parmi les mauvais
    hard_negative_sims, _ = sims_masked.max(dim=1)  # [B]
    
    # Triplet loss
    losses = F.relu(hard_negative_sims - positive_sims + margin)
    
    return losses.mean()


# =========================================================
# TRAINING
# =========================================================
def train_epoch(model, loader, optimizer, device, margin):
    model.train()
    total_loss, total = 0.0, 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    
    for graphs, text_emb in pbar:
        graphs = graphs.to(device)
        text_emb = text_emb.to(device)
        
        graph_emb = model(graphs)
        loss = triplet_loss_hard_negative(graph_emb, text_emb, margin)
        
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
    parser.add_argument('--train_emb', type=str, default=None)
    parser.add_argument('--val_emb', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=DEFAULT_CONFIG['epochs'])
    parser.add_argument('--batch_size', type=int, default=DEFAULT_CONFIG['batch_size'])
    parser.add_argument('--lr', type=float, default=DEFAULT_CONFIG['lr'])
    parser.add_argument('--margin', type=float, default=DEFAULT_CONFIG['margin'])
    parser.add_argument('--hidden_dim', type=int, default=DEFAULT_CONFIG['hidden_dim'])
    parser.add_argument('--num_layers', type=int, default=DEFAULT_CONFIG['num_layers'])
    parser.add_argument('--out_ckpt', type=str, default='model_gps_chembert_triplet.pt')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 70)
    print("GPS + ChemicalBERT + Triplet Loss (Hard Negatives)")
    print("=" * 70)
    print(f"Device: {device}")
    
    # Paths
    data_dir = Path(args.data_dir)
    train_graphs = str(data_dir / "train_graphs.pkl")
    val_graphs = str(data_dir / "validation_graphs.pkl")
    
    if args.train_emb is None:
        args.train_emb = str(data_dir / "train_embeddings.recobo_chemical-bert-uncased.cls.csv")
    if args.val_emb is None:
        args.val_emb = str(data_dir / "validation_embeddings.recobo_chemical-bert-uncased.cls.csv")
    
    print(f"Train embeddings: {args.train_emb}")
    print(f"Val embeddings: {args.val_emb}")
    
    # --- CHARGEMENT INTELLIGENT ---
    train_emb = smart_load_embeddings(args.train_emb)
    val_emb = smart_load_embeddings(args.val_emb) if os.path.exists(args.val_emb) else None
    
    emb_dim = len(next(iter(train_emb.values())))
    print(f"Embedding dimension: {emb_dim}")
    
    train_ds = PreprocessedGraphDataset(train_graphs, train_emb)
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, drop_last=True
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
    
    # Training loop
    best_mrr = 0.0
    patience_counter = 0
    patience = DEFAULT_CONFIG['patience']
    LOG_FREQ = 5
    
    for epoch in range(args.epochs):
        loss = train_epoch(model, train_dl, optimizer, device, args.margin)
        val_scores = eval_retrieval(model, val_graphs, val_emb, device) if val_emb else {}
        print(f"Epoch {epoch+1}/{args.epochs} | loss={loss:.4f} | {val_scores}")
        scheduler.step()

        should_log = (epoch == 0) or ((epoch + 1) % LOG_FREQ == 0)
        if should_log:
            print(f"Epoch {epoch+1}/{args.epochs} | loss={loss:.6f} | {val_scores}")
        
        current_mrr = val_scores.get('MRR', 0)
        if current_mrr > best_mrr:
            best_mrr = current_mrr
            patience_counter = 0
            
            # Create parent dir if needed
            Path(args.out_ckpt).parent.mkdir(parents=True, exist_ok=True)
            
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