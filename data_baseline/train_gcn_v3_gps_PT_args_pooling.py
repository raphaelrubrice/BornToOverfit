"""
train_gcn_v3_gps_PT_args.py - Version avec choix de Pooling via CLI
"""

import os
import argparse
from pathlib import Path
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

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
        data = torch.load(path)
        ids = data['ids']
        embs = data['embeddings']
        return {i: embs[idx] for idx, i in enumerate(ids)}
    else:
        return load_id2emb(path)


# =========================================================
# LOSS FUNCTIONS
# =========================================================
def infonce_loss(mol_vec, txt_vec, temperature=0.07):
    """InfoNCE loss (contrastive learning)."""
    batch_size = mol_vec.size(0)
    logits = mol_vec @ txt_vec.t() / temperature
    labels = torch.arange(batch_size, device=mol_vec.device)
    loss_m2t = F.cross_entropy(logits, labels)
    loss_t2m = F.cross_entropy(logits.t(), labels)
    return (loss_m2t + loss_t2m) / 2


def triplet_loss(mol_vec, txt_vec, margin=0.2):
    """Triplet loss avec hard negative mining."""
    batch_size = mol_vec.size(0)
    sims = mol_vec @ txt_vec.t()
    pos_sims = sims.diag().unsqueeze(1)
    mask = torch.eye(batch_size, device=mol_vec.device).bool()
    neg_sims = sims.masked_fill(mask, -float('inf'))
    
    hard_neg_m2t = neg_sims.max(dim=1, keepdim=True)[0]
    hard_neg_t2m = neg_sims.max(dim=0, keepdim=True)[0].t()
    
    loss_m2t = F.relu(margin + hard_neg_m2t - pos_sims).mean()
    loss_t2m = F.relu(margin + hard_neg_t2m - pos_sims).mean()
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
        
        # Détermination de la dimension d'entrée de la projection selon le pooling
        if self.pooling == 'all':
            # Concaténation de sum + mean + max
            proj_in_dim = hidden_dim * 3
        else:
            proj_in_dim = hidden_dim

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
        
        # Gestion du pooling
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
        
        # Normalisation des embeddings texte
        txt_vec = F.normalize(text_emb, dim=-1)
        
        # Sélection de la loss
        if loss_type == 'mse':
            loss = F.mse_loss(mol_vec, txt_vec)
        elif loss_type == 'infonce':
            loss = infonce_loss(mol_vec, txt_vec, **loss_kwargs)
        elif loss_type == 'triplet':
            loss = triplet_loss(mol_vec, txt_vec, **loss_kwargs)
        
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

def parse_checkpoint_name(filename):
    """
    Parse le nom du checkpoint pour extraire la config.
    
    Supporte plusieurs formats :
    - kaggle_model_v3_gps_triplet_margin0.2_bs128_lr0.0008_hd256_nl4.pt
    - model_v3_gps_triplet_pool_sum_hd256_nl4_drop0.1_heads4.pt
    
    Args:
        filename: Nom du fichier .pt
    
    Returns:
        dict: Configuration du modèle
    """
    config = {
        'pooling': 'sum',        # Default (si absent)
        'hidden_dim': 256,       # Default
        'num_layers': 4,         # Default
        'num_heads': 4,          # Default
        'dropout': 0.1,          # Default
        'out_dim': 768,          # Fixe (ChemBERT)
    }
    
    # ===== POOLING =====
    if 'pool_all' in filename:
        config['pooling'] = 'all'
    elif 'pool_mean' in filename:
        config['pooling'] = 'mean'
    elif 'pool_max' in filename:
        config['pooling'] = 'max'
    elif 'pool_sum' in filename:
        config['pooling'] = 'sum'
    # Sinon garde le default 'sum'
    
    # ===== HIDDEN DIM =====
    match = re.search(r'hd(\d+)', filename)
    if match:
        config['hidden_dim'] = int(match.group(1))
    
    # ===== NUM LAYERS =====
    match = re.search(r'nl(\d+)', filename)
    if match:
        config['num_layers'] = int(match.group(1))
    
    # ===== DROPOUT =====
    match = re.search(r'drop([\d.]+)', filename)
    if match:
        config['dropout'] = float(match.group(1))
    # Sinon garde le default 0.1
    
    # ===== NUM HEADS =====
    match = re.search(r'heads(\d+)', filename)
    if match:
        config['num_heads'] = int(match.group(1))
    # Sinon garde le default 4
    
    return config

def load_molgnn_gps_pooling_from_checkpoint(
    model_path: str,
    device: str,
    **kwargs, # NOT USED, for compatibility
):
    """
    Load MolGNN using a saved config + state_dict.
    """
    model_filename = model_path.split("/")[-1]
    
    # FIX: Correct variable name
    print("FILENAME", model_filename)
    
    cfg = parse_checkpoint_name(model_filename)
    print("CFG", cfg)
    
    model_class = cfg.get("model_class", "MolGNN_GPS_pooling")
    print("MODEL CLASS", model_class)
    
    if model_class not in ["MolGNN", "MolGNN_GPS_pooling"]:
        raise ValueError(f"Unsupported GNN class: {model_class}")
    print("Instantiating..")
    # FIX: Provide defaults for missing constants
    gnn = MolGNN_GPS_pooling(
        hidden_dim=cfg.get("hidden_dim", 256), 
        out_dim=cfg.get("out_dim", 768), 
        num_layers=cfg.get("num_layers", 4),
        num_heads=cfg.get("num_heads", 4),
        dropout=cfg.get("dropout", 0.1),
        pooling=cfg.get("pooling", "sum")
    ).to(device)
    print("Loading..")
    state = torch.load(model_path, map_location=device, weights_only=False)
    print("State loaded", state.keys())
    print("Moving into class..")
    try:
        gnn.load_state_dict(state)
    except Exception as e:
        print(e)
        raise Exception("Error occured")
    gnn.eval()

    return gnn

def main():
    # =========================================================
    # ARGUMENT PARSING
    # =========================================================
    parser = argparse.ArgumentParser(description='Train MolGNN with different losses & poolings')
    
    # Model configuration
    parser.add_argument('--pooling', type=str, default='sum',
                        choices=['sum', 'mean', 'max', 'all'],
                        help='Graph pooling strategy: sum, mean, max or all (concatenation)')
    
    # Loss configuration
    parser.add_argument('--loss', type=str, default='mse', 
                        choices=['mse', 'infonce', 'triplet'],
                        help='Loss function')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='Temperature for InfoNCE')
    parser.add_argument('--margin', type=float, default=0.2,
                        help='Margin for Triplet loss')
    
    # Training hyperparameters
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--num-layers', type=int, default=4, help='Number of GPS layers')
    parser.add_argument('--num-heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Data paths
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    
    args = parser.parse_args()
    
    # =========================================================
    # CONFIG
    # =========================================================
    file_path = Path(os.path.abspath(__file__))
    parent_folder = file_path.parent
    base_path = parent_folder / args.data_dir

    TRAIN_GRAPHS = str(base_path / "train_graphs.pkl")
    VAL_GRAPHS   = str(base_path / "validation_graphs.pkl")

    TRAIN_EMB_PATH = str(base_path / "train_embeddings_RealChemBERT.pt")
    VAL_EMB_PATH   = str(base_path / "validation_embeddings_RealChemBERT.pt")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    LOSS_PARAMS = {
        'infonce': {'temperature': args.temperature},
        'triplet': {'margin': args.margin},
        'mse': {}
    }

    print("=" * 60)
    print(f"TRAINING MolGNN v3 - GPS")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Loss: {args.loss}")
    print(f"Pooling: {args.pooling.upper()}")
    
    # Load embeddings
    train_emb = smart_load_embeddings(TRAIN_EMB_PATH)
    val_emb = smart_load_embeddings(VAL_EMB_PATH) if os.path.exists(VAL_EMB_PATH) else None
    
    emb_dim = len(next(iter(train_emb.values())))

    if not os.path.exists(TRAIN_GRAPHS):
        print(f"Error: {TRAIN_GRAPHS} not found")
        return
    
    train_ds = PreprocessedGraphDataset(TRAIN_GRAPHS, train_emb)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # Initialisation du modèle avec l'argument pooling
    mol_enc = MolGNN_GPS_pooling(
        hidden_dim=args.hidden_dim, 
        out_dim=emb_dim, 
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        pooling=args.pooling  # <-- Passage de l'argument ici
    ).to(DEVICE)
    
    print(f"Model parameters: {sum(p.numel() for p in mol_enc.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.AdamW(mol_enc.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_mrr = 0.0
    patience = 15
    patience_counter = 0

    for ep in range(args.epochs):
        loss = train_epoch(
            mol_enc, train_dl, optimizer, DEVICE, 
            loss_type=args.loss, 
            **LOSS_PARAMS[args.loss]
        )
        val_scores = eval_retrieval(VAL_GRAPHS, val_emb, mol_enc, DEVICE) if val_emb else {}
        print(f"Epoch {ep+1}/{args.epochs} | loss={loss:.4f} | {val_scores}")
        scheduler.step()

        current_mrr = val_scores.get('MRR', 0)
        if current_mrr > best_mrr:
            best_mrr = current_mrr
            patience_counter = 0
            
            # Nom de fichier incluant le POOLING
            filename_parts = [f"model_v3_gps_{args.loss}"]
            filename_parts.append(f"pool_{args.pooling}")  # Ajout ici
            
            if args.loss == 'infonce':
                filename_parts.append(f"temp{args.temperature}")
            
            filename_parts.append(f"hd{args.hidden_dim}")
            
            filename = "_".join(filename_parts) + ".pt"
            save_path = str(base_path / filename)
            
            torch.save(mol_enc.state_dict(), save_path)
            print(f"  → New best MRR: {best_mrr:.4f} | Saved: {filename}")
        else:
            patience_counter += 1
            print(f"  → No improvement ({patience_counter}/{patience})")

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {ep+1}")
            break

    print(f"\nDone. Best MRR: {best_mrr:.4f}")


if __name__ == "__main__":
    main()