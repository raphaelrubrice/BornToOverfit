"""
train_gcn_v3_gps.py - Version with GPS (General, Powerful, Scalable Graph Transformer)
Updated with Robust Early Stopping and Dynamic Pathing.
"""

import os
import json
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from pathlib import Path

from torch_geometric.data import Batch
from torch_geometric.nn import GPSConv, GINEConv, global_add_pool

try:
    from data_utils import (
        load_id2emb,
        PreprocessedGraphDataset, collate_fn,
        x_map, e_map
    )
    from training_utils import EarlyStopping

except ImportError:
    from .data_utils import (
        load_id2emb,
        PreprocessedGraphDataset, collate_fn,
        x_map, e_map
    )
    from.training_utils import EarlyStopping

# =========================================================
# GLOBAL DEFAULTS
# =========================================================
HIDDEN_DIM = 256
NUM_LAYERS = 4
NUM_HEADS = 4
DROPOUT = 0.1

BATCH_SIZE = 32
EPOCHS = 50
PATIENCE = 5
LR = 5e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
class MolGNN_GPS(nn.Module):
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
# Training and Evaluation
# =========================================================
def clip_infonce_loss(mol_vec, txt_vec, temperature=0.07):
    # mol_vec, txt_vec already normalized
    logits = (txt_vec @ mol_vec.t()) / temperature  # [B,B]
    labels = torch.arange(logits.size(0), device=logits.device)

    loss_t2m = F.cross_entropy(logits, labels)
    loss_m2t = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_t2m + loss_m2t)

def train_epoch(mol_enc, loader, optimizer, device, loss_func='mse', loss_kwargs={}):
    mol_enc.train()
    total_loss, total = 0.0, 0

    print(f"\nUsing {loss_func.upper()} loss")
    loss_func = F.mse_loss if loss_func == 'mse' else clip_infonce_loss
    
    for graphs, text_emb in loader:
        graphs, text_emb = graphs.to(device), text_emb.to(device)
        mol_vec = mol_enc(graphs)
        txt_vec = F.normalize(text_emb, dim=-1)
        
        loss = loss_func(mol_vec, txt_vec, **loss_kwargs)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mol_enc.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * graphs.num_graphs
        total += graphs.num_graphs
    return total_loss / total


@torch.no_grad()
def eval_retrieval(data_path, emb_dict, mol_enc, device, dl=None):
    mol_enc.eval()
    if dl is None:
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

    sims = all_txt @ all_mol.t()
    ranks = sims.argsort(dim=-1, descending=True)

    N = all_txt.size(0)
    device = sims.device
    correct = torch.arange(N, device=device)

    pos = (ranks == correct.unsqueeze(1)).nonzero()[:, 1] + 1
    mrr = (1.0 / pos.float()).mean().item()

    results = {"MRR": mrr}
    for k in (1, 5, 10):
        hitk = (pos <= k).float().mean().item()
        results[f"R@{k}"] = hitk

    return results

def load_molgnn_gps_from_checkpoint(
    model_path: str,
    device: str,
    **kwargs, # NOT USED, for compatibility
):
    """
    Load MolGNN using a saved config + state_dict.
    """
    model_dir = Path(model_path).parent
    config_path = model_dir / "model_config.json"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing GNN config file: {config_path}"
        )

    with open(config_path, "r") as f:
        cfg = json.load(f)

    model_class = cfg.get("model_class", "MolGNN_GPS")

    # Ensure this loader only loads GPS models
    if model_class != "MolGNN_GPS":
        raise ValueError(f"Unsupported GNN class: {model_class}")

    gnn = MolGNN_GPS(
        hidden_dim=cfg.get("hidden_dim", HIDDEN_DIM),
        out_dim=cfg.get("out_dim", 768),
        num_layers=cfg.get("num_layers", NUM_LAYERS),
        num_heads=cfg.get("num_heads", NUM_HEADS),
        dropout=cfg.get("dropout", DROPOUT)
    ).to(device)

    state = torch.load(model_path, map_location=device)
    gnn.load_state_dict(state)
    gnn.eval()

    return gnn

def main(data_folder, output_folder, loss_func, epochs):
    loss_func = loss_func.lower()

    # Setup Paths
    file_path = Path(os.path.abspath(__file__))
    parent_folder = file_path.parent
    
    data_path = parent_folder.parent / data_folder
    save_path = parent_folder.parent / output_folder
    os.makedirs(save_path, exist_ok=True)

    TRAIN_GRAPHS = str(data_path / "train_graphs.pkl")
    VAL_GRAPHS   = str(data_path / "validation_graphs.pkl")

    # FIXED: Load embeddings from source DATA folder, not output folder
    TRAIN_EMB_CSV = str(save_path / "train_embeddings.csv")
    VAL_EMB_CSV   = str(save_path / "validation_embeddings.csv")

    print("=" * 60)
    print("TRAINING MolGNN v3 - GPS Transformer")
    print(f"Data Source: {data_path}")
    print(f"Embed Source: {TRAIN_EMB_CSV}")
    print("=" * 60)

    if not os.path.exists(TRAIN_GRAPHS):
        print(f"Error: {TRAIN_GRAPHS} not found.")
        return

    # Load Embeddings & Datasets
    train_emb = load_id2emb(TRAIN_EMB_CSV)
    val_emb = load_id2emb(VAL_EMB_CSV) if os.path.exists(VAL_EMB_CSV) else None
    emb_dim = len(next(iter(train_emb.values())))

    train_ds = PreprocessedGraphDataset(TRAIN_GRAPHS, train_emb)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    val_ds = PreprocessedGraphDataset(VAL_GRAPHS, val_emb)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Initialize Model
    mol_enc = MolGNN_GPS(
        hidden_dim=HIDDEN_DIM, 
        out_dim=emb_dim, 
        num_layers=NUM_LAYERS, 
        num_heads=NUM_HEADS, 
        dropout=DROPOUT
    ).to(DEVICE)
    
    print(f"Parameters: {sum(p.numel() for p in mol_enc.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.AdamW(mol_enc.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Initialize Early Stopping (Mode='max' for MRR)
    early_stopper = EarlyStopping(patience=PATIENCE, mode='max')

    # Training Loop
    for ep in range(epochs):
        loss = train_epoch(mol_enc, train_dl, optimizer, DEVICE, loss_func=loss_func)
        val_scores = eval_retrieval(VAL_GRAPHS, val_emb, mol_enc, DEVICE, dl=val_dl) if val_emb else {}
        
        str_val = " | ".join([f"{k}: {v:.4f}" for k, v in val_scores.items()])
        print(f"Epoch {ep+1}/{epochs} | loss={loss:.4f} | {str_val}")
        
        current_mrr = val_scores.get('MRR', 0)
        
        # Check Early Stopping
        stop_signal, is_best = early_stopper.check_stop(mol_enc, current_mrr, ep)
        
        if is_best:
            print(f"  → New best MRR: {current_mrr:.4f}")
        else:
            print(f"  → No improvement ({early_stopper.patience_count}/{early_stopper.patience})")

        if stop_signal:
            print(f"\nEarly stopping triggered at epoch {ep+1}")
            break
            
        scheduler.step()

    print(f"\nTraining Finished.")
    
    # Load the best weights back before saving final checkpoint
    early_stopper.load_best_weights(mol_enc)
    print(f"Best MRR: {early_stopper.best_score:.4f}")

    # =========================================================
    # SAVING LOGIC
    # =========================================================
    config = {
        "model_class": "MolGNN_GPS",
        "hidden_dim": HIDDEN_DIM,
        "out_dim": emb_dim,
        "num_layers": NUM_LAYERS,
        "num_heads": NUM_HEADS,
        "dropout": DROPOUT,
        "uses_edge_features": True
    }
    
    config_path = save_path / "model_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    model_path = save_path / "model_checkpoint.pt"
    torch.save(mol_enc.state_dict(), model_path)

    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-f_data", default="data_baseline/data", type=str)
    parser.add_argument("-f", default="data_baseline/data", type=str)
    parser.add_argument("-loss", default="mse", type=str)
    parser.add_argument("-epochs", default=50, type=int)

    args = parser.parse_args()
    main(data_folder=args.f_data, output_folder=args.f, loss_func=args.loss, epochs=args.epochs)