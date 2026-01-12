#!/usr/bin/env python3
"""
train_gcn_v3_gps_PT_args.py - GPS (GINE + Transformer) for molecule-text retrieval
Version: adds --train_on_trainval to train on (train + validation)

Notes:
- This is retrieval (contrastive / alignment), NOT generative.
- Recommended: train on train+val (NOT test). Training on test is data leakage.
"""

import os
import argparse
from pathlib import Path
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

from torch_geometric.data import Batch
from torch_geometric.nn import GPSConv, GINEConv, global_add_pool, global_mean_pool, global_max_pool

try:
    from data_utils import (
        load_id2emb,
        PreprocessedGraphDataset, collate_fn,
        x_map, e_map
    )
except ImportError:
    from .data_utils import (
        load_id2emb,
        PreprocessedGraphDataset, collate_fn,
        x_map, e_map
    )

# =========================================================
# HELPER: CHARGEMENT INTELLIGENT (.pt ou .csv)
# =========================================================
def smart_load_embeddings(path):
    """Charge les embeddings depuis un CSV (lent) ou un PT (rapide)."""
    path = str(path)
    if path.endswith('.pt'):
        print(f"⚡ Chargement rapide depuis {path}...")
        data = torch.load(path, map_location="cpu")
        ids = data['ids']
        embs = data['embeddings']
        return {str(i): embs[idx].float() for idx, i in enumerate(ids)}
    else:
        return load_id2emb(path)

# =========================================================
# LOSS FUNCTIONS
# =========================================================
def infonce_loss(mol_vec, txt_vec, temperature=0.07):
    """InfoNCE loss (contrastive learning)."""
    bs = mol_vec.size(0)
    logits = mol_vec @ txt_vec.t() / temperature
    labels = torch.arange(bs, device=mol_vec.device)
    loss_m2t = F.cross_entropy(logits, labels)
    loss_t2m = F.cross_entropy(logits.t(), labels)
    return (loss_m2t + loss_t2m) / 2

def triplet_loss(mol_vec, txt_vec, margin=0.2):
    """Triplet loss avec hard negative mining."""
    bs = mol_vec.size(0)
    sims = mol_vec @ txt_vec.t()
    pos = sims.diag().unsqueeze(1)
    mask = torch.eye(bs, device=mol_vec.device).bool()
    neg = sims.masked_fill(mask, -float('inf'))
    hard_m2t = neg.max(dim=1, keepdim=True)[0]
    hard_t2m = neg.max(dim=0, keepdim=True)[0].t()
    loss_m2t = F.relu(margin + hard_m2t - pos).mean()
    loss_t2m = F.relu(margin + hard_t2m - pos).mean()
    return (loss_m2t + loss_t2m) / 2

# =========================================================
# Feature dimensions
# =========================================================
ATOM_FEATURE_DIMS = [
    len(x_map['atomic_num']), len(x_map['chirality']), len(x_map['degree']),
    len(x_map['formal_charge']), len(x_map['num_hs']), len(x_map['num_radical_electrons']),
    len(x_map['hybridization']), len(x_map['is_aromatic']), len(x_map['is_in_ring']),
]
BOND_FEATURE_DIMS = [
    len(e_map['bond_type']), len(e_map['stereo']), len(e_map['is_conjugated']),
]

# =========================================================
# Encoders
# =========================================================
class AtomEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(dim, hidden_dim) for dim in ATOM_FEATURE_DIMS])

    def forward(self, x):
        return sum(emb(x[:, i]) for i, emb in enumerate(self.embeddings))

class BondEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(dim, hidden_dim) for dim in BOND_FEATURE_DIMS])

    def forward(self, edge_attr):
        return sum(emb(edge_attr[:, i]) for i, emb in enumerate(self.embeddings))

# =========================================================
# MODEL: GPS Transformer avec Pooling Variable
# =========================================================
class MolGNN_GPS_pooling(nn.Module):
    def __init__(self, hidden_dim=256, out_dim=768,
                 num_layers=4, num_heads=4, dropout=0.1,
                 pooling='sum'):
        super().__init__()
        self.pooling = pooling
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

        proj_in_dim = hidden_dim * 3 if self.pooling == 'all' else hidden_dim
        self.proj = nn.Sequential(
            nn.Linear(proj_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, batch: Batch):
        h = self.atom_encoder(batch.x)
        edge_attr = self.bond_encoder(batch.edge_attr)

        for conv in self.convs:
            h = conv(h, batch.edge_index, batch.batch, edge_attr=edge_attr)

        if self.pooling == 'sum':
            g = global_add_pool(h, batch.batch)
        elif self.pooling == 'mean':
            g = global_mean_pool(h, batch.batch)
        elif self.pooling == 'max':
            g = global_max_pool(h, batch.batch)
        elif self.pooling == 'all':
            g1 = global_add_pool(h, batch.batch)
            g2 = global_mean_pool(h, batch.batch)
            g3 = global_max_pool(h, batch.batch)
            g = torch.cat([g1, g2, g3], dim=1)
        else:
            raise ValueError(f"Pooling non supporté : {self.pooling}")

        g = self.proj(g)
        return F.normalize(g, dim=-1)

# =========================================================
# Training and Evaluation
# =========================================================
def train_epoch(mol_enc, loader, optimizer, device, loss_type='mse', **loss_kwargs):
    mol_enc.train()
    total_loss, total = 0.0, 0

    for graphs, text_emb in loader:
        graphs, text_emb = graphs.to(device), text_emb.to(device)
        mol_vec = mol_enc(graphs)
        txt_vec = F.normalize(text_emb, dim=-1)

        if loss_type == 'mse':
            loss = F.mse_loss(mol_vec, txt_vec)
        elif loss_type == 'infonce':
            loss = infonce_loss(mol_vec, txt_vec, **loss_kwargs)
        elif loss_type == 'triplet':
            loss = triplet_loss(mol_vec, txt_vec, **loss_kwargs)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mol_enc.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * graphs.num_graphs
        total += graphs.num_graphs

    return total_loss / max(total, 1)

@torch.no_grad()
def eval_retrieval(graph_pkl_path, emb_dict, mol_enc, device):
    mol_enc.eval()
    ds = PreprocessedGraphDataset(graph_pkl_path, emb_dict)
    dl = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    all_mol, all_txt = [], []
    for graphs, text_emb in dl:
        graphs, text_emb = graphs.to(device), text_emb.to(device)
        all_mol.append(mol_enc(graphs))
        all_txt.append(F.normalize(text_emb, dim=-1))

    all_mol = torch.cat(all_mol, dim=0)
    all_txt = torch.cat(all_txt, dim=0)

    sims = all_txt @ all_mol.t()
    ranks = sims.argsort(dim=-1, descending=True)
    N = all_txt.size(0)
    correct = torch.arange(N, device=sims.device)
    pos = (ranks == correct.unsqueeze(1)).nonzero()[:, 1] + 1

    mrr = (1.0 / pos.float()).mean().item()
    out = {"MRR": mrr}
    for k in (1, 5, 10):
        out[f"R@{k}"] = (pos <= k).float().mean().item()
    return out

# =========================================================
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser(description='Train MolGNN with different losses & poolings')

    # Pooling
    parser.add_argument('--pooling', type=str, default='sum',
                        choices=['sum', 'mean', 'max', 'all'],
                        help='Graph pooling strategy: sum, mean, max or all')

    # Loss
    parser.add_argument('--loss', type=str, default='mse',
                        choices=['mse', 'infonce', 'triplet'],
                        help='Loss function')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='Temperature for InfoNCE')
    parser.add_argument('--margin', type=float, default=0.2,
                        help='Margin for Triplet loss')

    # Training hyperparams
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)

    # Data
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Data directory containing *_graphs.pkl and embeddings')
    parser.add_argument('--train_on_trainval', action='store_true',
                        help='Train on train + validation (no proper val MRR during training).')

    args = parser.parse_args()

    file_path = Path(os.path.abspath(__file__))
    base_path = file_path.parent / args.data_dir

    TRAIN_GRAPHS = str(base_path / "train_graphs.pkl")
    VAL_GRAPHS   = str(base_path / "validation_graphs.pkl")

    # IMPORTANT: adjust these to your actual embedding files
    TRAIN_EMB_PATH = str(base_path / "train_embeddings_RealChemBERT.pt")
    VAL_EMB_PATH   = str(base_path / "validation_embeddings_RealChemBERT.pt")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    LOSS_PARAMS = {
        'infonce': {'temperature': args.temperature},
        'triplet': {'margin': args.margin},
        'mse': {}
    }

    print("=" * 70)
    print("TRAINING MolGNN v3 - GPS (Retrieval)")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Loss: {args.loss}")
    print(f"Pooling: {args.pooling}")
    print(f"Data dir: {base_path}")
    print(f"Train+Val mode: {args.train_on_trainval}")

    # Sanity checks
    if not os.path.exists(TRAIN_GRAPHS):
        raise FileNotFoundError(f"Missing: {TRAIN_GRAPHS}")
    if args.train_on_trainval and not os.path.exists(VAL_GRAPHS):
        raise FileNotFoundError(f"Missing: {VAL_GRAPHS}")

    # Load embeddings
    if not os.path.exists(TRAIN_EMB_PATH):
        raise FileNotFoundError(f"Missing train embeddings: {TRAIN_EMB_PATH}")
    train_emb = smart_load_embeddings(TRAIN_EMB_PATH)

    val_emb = None
    if os.path.exists(VAL_EMB_PATH):
        val_emb = smart_load_embeddings(VAL_EMB_PATH)

    emb_dim = len(next(iter(train_emb.values())))
    print(f"Text embedding dim: {emb_dim}")

    # Build train dataset
    if args.train_on_trainval:
        if val_emb is None:
            raise FileNotFoundError(f"train_on_trainval requires val embeddings: {VAL_EMB_PATH}")

        merged_emb = {**train_emb, **val_emb}  # IDs should be disjoint
        train_ds = ConcatDataset([
            PreprocessedGraphDataset(TRAIN_GRAPHS, merged_emb),
            PreprocessedGraphDataset(VAL_GRAPHS, merged_emb),
        ])
        print("✅ Training on: TRAIN + VALIDATION")
    else:
        train_ds = PreprocessedGraphDataset(TRAIN_GRAPHS, train_emb)
        print("✅ Training on: TRAIN only")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # Model
    mol_enc = MolGNN_GPS_pooling(
        hidden_dim=args.hidden_dim,
        out_dim=emb_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        pooling=args.pooling
    ).to(DEVICE)

    n_params = sum(p.numel() for p in mol_enc.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(mol_enc.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_score = -1e9
    patience = 15
    patience_counter = 0

    for ep in range(args.epochs):
        loss = train_epoch(
            mol_enc, train_dl, optimizer, DEVICE,
            loss_type=args.loss,
            **LOSS_PARAMS[args.loss]
        )

        # If training on train+val, we don't have a clean validation set anymore
        if args.train_on_trainval:
            print(f"Epoch {ep+1}/{args.epochs} | loss={loss:.6f} | (train+val mode: no val MRR)")
            current_score = -loss  # model selection proxy
        else:
            if val_emb is None:
                print(f"Epoch {ep+1}/{args.epochs} | loss={loss:.6f} | (no val embeddings available)")
                current_score = -loss
            else:
                val_scores = eval_retrieval(VAL_GRAPHS, val_emb, mol_enc, DEVICE)
                print(f"Epoch {ep+1}/{args.epochs} | loss={loss:.6f} | {val_scores}")
                current_score = val_scores.get("MRR", 0.0)

        scheduler.step()

        if current_score > best_score:
            best_score = current_score
            patience_counter = 0

            filename_parts = [f"model_v3_gps_{args.loss}", f"pool_{args.pooling}"]
            if args.loss == 'infonce':
                filename_parts.append(f"temp{args.temperature}")
            if args.loss == 'triplet':
                filename_parts.append(f"margin{args.margin}")
            filename_parts.append(f"hd{args.hidden_dim}")
            filename_parts.append(f"nl{args.num_layers}")
            filename_parts.append(f"heads{args.num_heads}")
            filename_parts.append(f"drop{args.dropout}")

            filename = "_".join(filename_parts) + ".pt"
            save_path = str(base_path / filename)
            torch.save(mol_enc.state_dict(), save_path)
            print(f"  → ✅ Saved best checkpoint: {filename}")
        else:
            patience_counter += 1
            print(f"  → No improvement ({patience_counter}/{patience})")

        # Early stopping only meaningful when we still validate on val
        if (not args.train_on_trainval) and patience_counter >= patience:
            print(f"\nEarly stopping at epoch {ep+1}")
            break

    print("\n" + "=" * 70)
    if args.train_on_trainval:
        print("Done. (train+val) Best proxy score = -loss:", best_score)
    else:
        print("Done. Best val MRR:", best_score)
    print("=" * 70)

if __name__ == "__main__":
    main()
