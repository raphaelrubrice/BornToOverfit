"""
train_gcn_v3_gps.py - GPS (General, Powerful, Scalable Graph Transformer)
Now supports choosing which text-embedding file to train against.
"""

import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch_geometric.data import Batch
from torch_geometric.nn import GPSConv, GINEConv, global_add_pool

from data_utils import load_id2emb, PreprocessedGraphDataset, collate_fn, x_map, e_map


# =========================================================
# Default CONFIG
# =========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HIDDEN_DIM = 256
NUM_LAYERS = 4
NUM_HEADS = 4
DROPOUT = 0.1


ATOM_FEATURE_DIMS = [
    len(x_map['atomic_num']), len(x_map['chirality']), len(x_map['degree']),
    len(x_map['formal_charge']), len(x_map['num_hs']), len(x_map['num_radical_electrons']),
    len(x_map['hybridization']), len(x_map['is_aromatic']), len(x_map['is_in_ring']),
]
BOND_FEATURE_DIMS = [
    len(e_map['bond_type']), len(e_map['stereo']), len(e_map['is_conjugated']),
]


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


class MolGNN(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM, out_dim=768, num_layers=NUM_LAYERS, num_heads=NUM_HEADS, dropout=DROPOUT):
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

        g = global_add_pool(h, batch.batch)
        g = self.proj(g)
        return F.normalize(g, dim=-1)


def train_epoch(mol_enc, loader, optimizer, device):
    mol_enc.train()
    total_loss, total = 0.0, 0
    for graphs, text_emb in loader:
        graphs, text_emb = graphs.to(device), text_emb.to(device)
        mol_vec = mol_enc(graphs)
        txt_vec = F.normalize(text_emb, dim=-1)
        loss = F.mse_loss(mol_vec, txt_vec)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mol_enc.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * graphs.num_graphs
        total += graphs.num_graphs
    return total_loss / total


@torch.no_grad()
def eval_retrieval(data_path, emb_dict, mol_enc, device):
    mol_enc.eval()
    ds = PreprocessedGraphDataset(data_path, emb_dict)
    dl = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    all_mol, all_txt = [], []
    for graphs, text_emb in dl:
        graphs, text_emb = graphs.to(device), text_emb.to(device)
        all_mol.append(mol_enc(graphs))
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--train_emb", type=str, required=True)
    ap.add_argument("--val_emb", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--out_ckpt", type=str, default=None)
    args = ap.parse_args()

    base = Path(args.data_dir)
    TRAIN_GRAPHS = str(base / "train_graphs.pkl")
    VAL_GRAPHS = str(base / "validation_graphs.pkl")

    print("=" * 70)
    print("TRAINING MolGNN - GPS Transformer")
    print("=" * 70)
    print(f"Device: {DEVICE} | epochs={args.epochs} | batch={args.batch_size} | lr={args.lr}")
    print(f"train_emb: {args.train_emb}")
    print(f"val_emb:   {args.val_emb}")

    train_emb = load_id2emb(args.train_emb)
    val_emb = load_id2emb(args.val_emb)

    emb_dim = len(next(iter(train_emb.values())))

    train_ds = PreprocessedGraphDataset(TRAIN_GRAPHS, train_emb)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    mol_enc = MolGNN(hidden_dim=HIDDEN_DIM, out_dim=emb_dim, num_layers=NUM_LAYERS).to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in mol_enc.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.AdamW(mol_enc.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt_path = args.out_ckpt or str(base / "model_checkpoint_gps.pt")

    best_mrr = 0.0
    patience = 5
    patience_counter = 0

    for ep in range(args.epochs):
        loss = train_epoch(mol_enc, train_dl, optimizer, DEVICE)
        val_scores = eval_retrieval(VAL_GRAPHS, val_emb, mol_enc, DEVICE)
        print(f"Epoch {ep+1}/{args.epochs} | loss={loss:.4f} | {val_scores}")
        scheduler.step()

        current_mrr = val_scores.get("MRR", 0.0)
        if current_mrr > best_mrr:
            best_mrr = current_mrr
            patience_counter = 0
            torch.save(mol_enc.state_dict(), ckpt_path)
            print(f"  → New best MRR: {best_mrr:.4f} | saved: {ckpt_path}")
        else:
            patience_counter += 1
            print(f"  → No improvement ({patience_counter}/{patience})")

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {ep+1}")
            break

    print(f"\nDone. Best MRR: {best_mrr:.4f}")


if __name__ == "__main__":
    main()
