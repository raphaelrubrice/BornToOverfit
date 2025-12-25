"""
train_gcn_v1.py - Version améliorée avec Node et Edge Features

MODIFICATIONS PAR RAPPORT AU BASELINE:
1. Utilisation des 9 features atomiques via nn.Embedding
2. Utilisation des 3 features de liaison via nn.Embedding  
3. Passage de GCNConv à GINEConv (supporte edge_attr)
4. Architecture plus profonde avec skip connections
"""

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch_geometric.data import Batch
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool

from data_utils import (
    load_id2emb,
    PreprocessedGraphDataset, collate_fn,
    x_map, e_map  # On importe les mappings pour connaître les cardinalités
)


# =========================================================
# CONFIG
# =========================================================
file_path = Path(os.path.abspath(__file__))
parent_folder = file_path.parent
base_path = parent_folder / "data"

# Data paths
TRAIN_GRAPHS = str(base_path / "train_graphs.pkl")
VAL_GRAPHS   = str(base_path / "validation_graphs.pkl")
TEST_GRAPHS  = str(base_path / "test_graphs.pkl")

TRAIN_EMB_CSV = str(base_path / "train_embeddings.csv")
VAL_EMB_CSV   = str(base_path / "validation_embeddings.csv")

# Training parameters
BATCH_SIZE = 32
EPOCHS = 10  # Augmenté car modèle plus complexe
LR = 5e-4    # Réduit pour stabilité
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Architecture parameters
HIDDEN_DIM = 256
NUM_LAYERS = 4
DROPOUT = 0.1


# =========================================================
# Cardinalités des features (depuis x_map et e_map)
# =========================================================
ATOM_FEATURE_DIMS = [
    len(x_map['atomic_num']),           # 119
    len(x_map['chirality']),            # 9
    len(x_map['degree']),               # 11
    len(x_map['formal_charge']),        # 12
    len(x_map['num_hs']),               # 9
    len(x_map['num_radical_electrons']),# 5
    len(x_map['hybridization']),        # 8
    len(x_map['is_aromatic']),          # 2
    len(x_map['is_in_ring']),           # 2
]

BOND_FEATURE_DIMS = [
    len(e_map['bond_type']),            # 22
    len(e_map['stereo']),               # 6
    len(e_map['is_conjugated']),        # 2
]


# =========================================================
# Atom and Bond Encoders
# =========================================================
class AtomEncoder(nn.Module):
    """
    Encode les 9 features atomiques via des embeddings séparés,
    puis les combine par somme.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_classes, hidden_dim) 
            for num_classes in ATOM_FEATURE_DIMS
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [num_nodes, 9] tensor d'indices catégoriels
        Returns:
            [num_nodes, hidden_dim] embeddings des atomes
        """
        out = 0
        for i, emb in enumerate(self.embeddings):
            out = out + emb(x[:, i])
        return out


class BondEncoder(nn.Module):
    """
    Encode les 3 features de liaison via des embeddings séparés,
    puis les combine par somme.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_classes, hidden_dim) 
            for num_classes in BOND_FEATURE_DIMS
        ])
        
    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_attr: [num_edges, 3] tensor d'indices catégoriels
        Returns:
            [num_edges, hidden_dim] embeddings des liaisons
        """
        out = 0
        for i, emb in enumerate(self.embeddings):
            out = out + emb(edge_attr[:, i])
        return out


# =========================================================
# MODEL: GNN avec Node et Edge Features
# =========================================================
class MolGNN(nn.Module):
    """
    GNN amélioré utilisant:
    - AtomEncoder pour les features atomiques
    - BondEncoder pour les features de liaison
    - GINEConv (GIN with Edge features) pour la propagation
    - Skip connections (residual)
    - Layer normalization
    """
    def __init__(self, hidden_dim: int = HIDDEN_DIM, out_dim: int = 768, 
                 num_layers: int = NUM_LAYERS, dropout: float = DROPOUT):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Encoders
        self.atom_encoder = AtomEncoder(hidden_dim)
        self.bond_encoder = BondEncoder(hidden_dim)
        
        # GNN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for _ in range(num_layers):
            # GINEConv utilise un MLP interne
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, 2 * hidden_dim),
                nn.BatchNorm1d(2 * hidden_dim),
                nn.ReLU(),
                nn.Linear(2 * hidden_dim, hidden_dim),
            )
            conv = GINEConv(mlp, train_eps=True)
            self.convs.append(conv)
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Pooling et projection finale
        self.pool = global_add_pool
        
        # MLP de projection vers l'espace d'embedding textuel
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, batch: Batch) -> torch.Tensor:
        """
        Args:
            batch: Batch de graphes PyG
        Returns:
            [batch_size, out_dim] embeddings normalisés L2
        """
        # Encode atoms et bonds
        h = self.atom_encoder(batch.x)
        edge_attr = self.bond_encoder(batch.edge_attr)
        
        # Message passing avec skip connections
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            h_new = conv(h, batch.edge_index, edge_attr)
            h_new = bn(h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            
            # Skip connection (residual)
            h = h + h_new
        
        # Pooling global
        g = self.pool(h, batch.batch)
        
        # Projection et normalisation
        g = self.proj(g)
        g = F.normalize(g, dim=-1)
        
        return g


# =========================================================
# Training and Evaluation
# =========================================================
def train_epoch(mol_enc: nn.Module, loader: DataLoader, 
                optimizer: torch.optim.Optimizer, device: str) -> float:
    """Une epoch d'entraînement avec MSE loss."""
    mol_enc.train()
    total_loss, total = 0.0, 0
    
    for graphs, text_emb in loader:
        graphs = graphs.to(device)
        text_emb = text_emb.to(device)

        mol_vec = mol_enc(graphs)
        txt_vec = F.normalize(text_emb, dim=-1)

        loss = F.mse_loss(mol_vec, txt_vec)

        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping pour stabilité
        torch.nn.utils.clip_grad_norm_(mol_enc.parameters(), max_norm=1.0)
        
        optimizer.step()

        bs = graphs.num_graphs
        total_loss += loss.item() * bs
        total += bs

    return total_loss / total


@torch.no_grad()
def eval_retrieval(data_path: str, emb_dict: dict, mol_enc: nn.Module, 
                   device: str) -> dict:
    """Évalue les métriques de retrieval (MRR, Recall@k)."""
    mol_enc.eval()
    
    ds = PreprocessedGraphDataset(data_path, emb_dict)
    dl = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    all_mol, all_txt = [], []
    for graphs, text_emb in dl:
        graphs = graphs.to(device)
        text_emb = text_emb.to(device)
        all_mol.append(mol_enc(graphs))
        all_txt.append(F.normalize(text_emb, dim=-1))
    
    all_mol = torch.cat(all_mol, dim=0)
    all_txt = torch.cat(all_txt, dim=0)

    # Similarité cosinus (text query → graph candidates)
    sims = all_txt @ all_mol.t()
    ranks = sims.argsort(dim=-1, descending=True)

    N = all_txt.size(0)
    correct = torch.arange(N, device=sims.device)

    # Position du bon match pour chaque query
    pos = (ranks == correct.unsqueeze(1)).nonzero()[:, 1] + 1

    mrr = (1.0 / pos.float()).mean().item()

    results = {"MRR": mrr}
    for k in (1, 5, 10):
        hitk = (pos <= k).float().mean().item()
        results[f"R@{k}"] = hitk

    return results


# =========================================================
# Main Training Loop
# =========================================================
def main():
    print("=" * 60)
    print("TRAINING MolGNN v1 - With Node & Edge Features")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Hidden dim: {HIDDEN_DIM}, Layers: {NUM_LAYERS}")
    print(f"Batch size: {BATCH_SIZE}, LR: {LR}, Epochs: {EPOCHS}")
    print()

    # Load embeddings
    train_emb = load_id2emb(TRAIN_EMB_CSV)
    val_emb = load_id2emb(VAL_EMB_CSV) if os.path.exists(VAL_EMB_CSV) else None
    emb_dim = len(next(iter(train_emb.values())))
    print(f"Text embedding dim: {emb_dim}")

    # Check data
    if not os.path.exists(TRAIN_GRAPHS):
        print(f"Error: {TRAIN_GRAPHS} not found")
        return
    
    # Load dataset
    train_ds = PreprocessedGraphDataset(TRAIN_GRAPHS, train_emb)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                          collate_fn=collate_fn, num_workers=0)

    # Model
    mol_enc = MolGNN(hidden_dim=HIDDEN_DIM, out_dim=emb_dim, 
                     num_layers=NUM_LAYERS, dropout=DROPOUT).to(DEVICE)
    
    num_params = sum(p.numel() for p in mol_enc.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    print()

    # Optimizer avec weight decay
    optimizer = torch.optim.AdamW(mol_enc.parameters(), lr=LR, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Training loop
    best_mrr = 0.0
    for ep in range(EPOCHS):
        train_loss = train_epoch(mol_enc, train_dl, optimizer, DEVICE)
        
        # Evaluation
        val_scores = {}
        if val_emb is not None and os.path.exists(VAL_GRAPHS):
            val_scores = eval_retrieval(VAL_GRAPHS, val_emb, mol_enc, DEVICE)
        
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        print(f"Epoch {ep+1}/{EPOCHS} | loss={train_loss:.4f} | lr={current_lr:.2e} | {val_scores}")
        
        # Save best model
        if val_scores.get('MRR', 0) > best_mrr:
            best_mrr = val_scores['MRR']
            model_path = str(base_path / "model_checkpoint_v1.pt")
            torch.save(mol_enc.state_dict(), model_path)
            print(f"  → New best MRR: {best_mrr:.4f}, saved to {model_path}")
    
    print()
    print("=" * 60)
    print(f"Training complete. Best MRR: {best_mrr:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()