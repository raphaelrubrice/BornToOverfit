"""
train_gcn_v3_gps_infonce.py - GPS avec InfoNCE Loss

CHANGEMENTS vs v3_gps:
1. MSE Loss → InfoNCE Loss (contrastive)
2. Ajout du paramètre température τ
3. Loss symétrique (graph→text + text→graph)
"""

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch_geometric.data import Batch
from torch_geometric.nn import GPSConv, GINEConv, global_add_pool

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

BATCH_SIZE = 64  # Plus grand batch = plus de négatifs = meilleur pour InfoNCE
EPOCHS = 100
LR = 5e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HIDDEN_DIM = 256
NUM_LAYERS = 4
NUM_HEADS = 4
DROPOUT = 0.1

# InfoNCE hyperparameters
TEMPERATURE = 0.07  # Température (0.05-0.1 typique)


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
# MODEL: GPS Transformer
# =========================================================
class MolGNN(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM, out_dim=768, 
                 num_layers=NUM_LAYERS, num_heads=NUM_HEADS, dropout=DROPOUT):
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
# InfoNCE Loss
# =========================================================
def info_nce_loss(graph_emb: torch.Tensor, text_emb: torch.Tensor, 
                  temperature: float = TEMPERATURE) -> torch.Tensor:
    """
    Calcule la loss InfoNCE symétrique.
    
    Args:
        graph_emb: [B, D] embeddings de graphes (normalisés L2)
        text_emb: [B, D] embeddings de texte (normalisés L2)
        temperature: τ, contrôle la "dureté" du softmax
    
    Returns:
        Loss scalaire
    
    Explication:
    - logits[i,j] = similarité entre graphe i et texte j
    - Pour chaque ligne i, le label correct est i (la diagonale)
    - cross_entropy fait le softmax et compare avec le label
    """
    # Similarités : [B, B]
    # graph_emb @ text_emb.T donne cos(g_i, t_j) car déjà normalisés
    logits = graph_emb @ text_emb.T / temperature
    
    # Labels : la diagonale (graphe i doit matcher texte i)
    labels = torch.arange(len(logits), device=logits.device)
    
    # Loss symétrique : graph→text + text→graph
    loss_g2t = F.cross_entropy(logits, labels)      # Pour chaque graphe, trouver son texte
    loss_t2g = F.cross_entropy(logits.T, labels)    # Pour chaque texte, trouver son graphe
    
    return (loss_g2t + loss_t2g) / 2


# =========================================================
# Training with InfoNCE
# =========================================================
def train_epoch(mol_enc, loader, optimizer, device, temperature):
    mol_enc.train()
    total_loss, total = 0.0, 0
    
    for graphs, text_emb in loader:
        graphs = graphs.to(device)
        text_emb = text_emb.to(device)
        
        # Forward
        graph_emb = mol_enc(graphs)  # Déjà normalisé dans le modèle
        text_emb = F.normalize(text_emb, dim=-1)
        
        # InfoNCE loss (remplace MSE)
        loss = info_nce_loss(graph_emb, text_emb, temperature)
        
        # Backward
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


# =========================================================
# Main
# =========================================================
def main():
    print("=" * 60)
    print("TRAINING MolGNN - GPS + InfoNCE Loss")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Temperature τ: {TEMPERATURE}")
    print(f"Batch size: {BATCH_SIZE} (plus grand = plus de négatifs)")
    print()

    train_emb = load_id2emb(TRAIN_EMB_CSV)
    val_emb = load_id2emb(VAL_EMB_CSV) if os.path.exists(VAL_EMB_CSV) else None
    emb_dim = len(next(iter(train_emb.values())))

    if not os.path.exists(TRAIN_GRAPHS):
        print(f"Error: {TRAIN_GRAPHS} not found")
        return
    
    train_ds = PreprocessedGraphDataset(TRAIN_GRAPHS, train_emb)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                          collate_fn=collate_fn, drop_last=True)  # drop_last pour batch complet

    mol_enc = MolGNN(hidden_dim=HIDDEN_DIM, out_dim=emb_dim, num_layers=NUM_LAYERS).to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in mol_enc.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.AdamW(mol_enc.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_mrr = 0.0
    patience = 5
    patience_counter = 0

    for ep in range(EPOCHS):
        loss = train_epoch(mol_enc, train_dl, optimizer, DEVICE, TEMPERATURE)
        val_scores = eval_retrieval(VAL_GRAPHS, val_emb, mol_enc, DEVICE) if val_emb else {}
        print(f"Epoch {ep+1}/{EPOCHS} | loss={loss:.4f} | {val_scores}")
        scheduler.step()
        
        current_mrr = val_scores.get('MRR', 0)
        if current_mrr > best_mrr:
            best_mrr = current_mrr
            patience_counter = 0
            torch.save(mol_enc.state_dict(), str(base_path / "model_checkpoint_gps_infonce.pt"))
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
