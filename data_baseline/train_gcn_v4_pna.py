"""
train_gcn_v4_pna.py - Version avec PNA (Principal Neighbourhood Aggregation)

PNA utilise plusieurs agrégateurs (mean, max, min, std) + scalers.
Très expressif, particulièrement bon sur les graphes moléculaires.
"""

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch_geometric.data import Batch
from torch_geometric.nn import PNAConv, global_add_pool, BatchNorm

from data_utils import (
    load_id2emb,
    PreprocessedGraphDataset, collate_fn,
    x_map, e_map
)


# =========================================================
# CONFIG
# =========================================================
file_path = Path(os.path.abspath(__file__))
parent_folder = file_path.parent
base_path = parent_folder / "data"

TRAIN_GRAPHS = str(base_path / "train_graphs.pkl")
VAL_GRAPHS   = str(base_path / "validation_graphs.pkl")
TEST_GRAPHS  = str(base_path / "test_graphs.pkl")
TRAIN_EMB_CSV = str(base_path / "train_embeddings.csv")
VAL_EMB_CSV   = str(base_path / "validation_embeddings.csv")

BATCH_SIZE = 32
EPOCHS = 50
LR = 5e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HIDDEN_DIM = 256
NUM_LAYERS = 4
DROPOUT = 0.1


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
# Compute degree histogram (required for PNA)
# =========================================================
def compute_deg_histogram(data_path, max_deg=10):
    """Calcule l'histogramme des degrés sur le dataset d'entraînement."""
    import pickle
    with open(data_path, 'rb') as f:
        graphs = pickle.load(f)
    
    deg_hist = torch.zeros(max_deg + 1, dtype=torch.long)
    for g in graphs:
        edge_index = g.edge_index
        num_nodes = g.x.size(0)
        # Compute degree
        row = edge_index[0]
        deg = torch.bincount(row, minlength=num_nodes)
        deg = deg.clamp(max=max_deg)
        for d in deg:
            deg_hist[d] += 1
    return deg_hist


# =========================================================
# MODEL: PNA
# =========================================================
class MolGNN(nn.Module):
    """
    PNA utilise multiple aggregators (mean, min, max, std) et 
    scalers (identity, amplification, attenuation) pour une expressivité maximale.
    """
    def __init__(self, hidden_dim=HIDDEN_DIM, out_dim=768, 
                 num_layers=NUM_LAYERS, dropout=DROPOUT, deg=None):
        super().__init__()
        
        self.atom_encoder = AtomEncoder(hidden_dim)
        self.bond_encoder = BondEncoder(hidden_dim)
        
        # PNA configuration
        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for _ in range(num_layers):
            conv = PNAConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                aggregators=aggregators,
                scalers=scalers,
                deg=deg,
                edge_dim=hidden_dim,
                towers=4,
                pre_layers=1,
                post_layers=1,
                divide_input=False,
            )
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hidden_dim))
        
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
        
        for conv, bn in zip(self.convs, self.batch_norms):
            h_new = conv(h, batch.edge_index, edge_attr)
            h_new = bn(h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=0.1, training=self.training)
            h = h + h_new  # Skip connection
        
        g = self.pool(h, batch.batch)
        g = self.proj(g)
        return F.normalize(g, dim=-1)


# =========================================================
# Training and Evaluation
# =========================================================
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
    print("=" * 60)
    print("TRAINING MolGNN v4 - PNA (Principal Neighbourhood Aggregation)")
    print("=" * 60)
    print(f"Device: {DEVICE}, Hidden: {HIDDEN_DIM}, Layers: {NUM_LAYERS}")

    # Compute degree histogram (required for PNA)
    print("Computing degree histogram...")
    deg = compute_deg_histogram(TRAIN_GRAPHS)
    print(f"Degree histogram: {deg}")

    train_emb = load_id2emb(TRAIN_EMB_CSV)
    val_emb = load_id2emb(VAL_EMB_CSV) if os.path.exists(VAL_EMB_CSV) else None
    emb_dim = len(next(iter(train_emb.values())))

    if not os.path.exists(TRAIN_GRAPHS):
        print(f"Error: {TRAIN_GRAPHS} not found"); return
    
    train_ds = PreprocessedGraphDataset(TRAIN_GRAPHS, train_emb)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    mol_enc = MolGNN(hidden_dim=HIDDEN_DIM, out_dim=emb_dim, num_layers=NUM_LAYERS, deg=deg).to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in mol_enc.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.AdamW(mol_enc.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    #best_mrr = 0.0
    #for ep in range(EPOCHS):
    #    loss = train_epoch(mol_enc, train_dl, optimizer, DEVICE)
    #    val_scores = eval_retrieval(VAL_GRAPHS, val_emb, mol_enc, DEVICE) if val_emb else {}
    #    print(f"Epoch {ep+1}/{EPOCHS} | loss={loss:.4f} | {val_scores}")
    #    scheduler.step()
    #    if val_scores.get('MRR', 0) > best_mrr:
    #        best_mrr = val_scores['MRR']
    #        torch.save(mol_enc.state_dict(), str(base_path / "model_checkpoint_v4_pna.pt"))
    #        print(f"  → New best MRR: {best_mrr:.4f}")
#
    #print(f"\nDone. Best MRR: {best_mrr:.4f}")

    best_mrr = 0.0
    patience = 5  # Nombre d'epochs sans amélioration avant d'arrêter
    patience_counter = 0

    for ep in range(EPOCHS):
        loss = train_epoch(mol_enc, train_dl, optimizer, DEVICE)
        val_scores = eval_retrieval(VAL_GRAPHS, val_emb, mol_enc, DEVICE) if val_emb else {}
        print(f"Epoch {ep+1}/{EPOCHS} | loss={loss:.4f} | {val_scores}")
        scheduler.step()

        current_mrr = val_scores.get('MRR', 0)
        if current_mrr > best_mrr:
            best_mrr = current_mrr
            patience_counter = 0
            torch.save(mol_enc.state_dict(), str(base_path / "model_checkpoint_v3_gps.pt"))
            print(f"  → New best MRR: {best_mrr:.4f}")
        else:
            patience_counter += 1
            print(f"  → No improvement ({patience_counter}/{patience})")

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {ep+1}")
            break

    print(f"\nDone. Best MRR: {best_mrr:.4f}")


if __name__ == "__main__":
    main()
