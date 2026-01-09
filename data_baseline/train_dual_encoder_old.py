"""
train_dual_encoder.py - Dual Encoder (GNN + ChemBERT entraînés ensemble)

Différence clé vs baseline :
- Baseline : embeddings ChemBERT figés (.pt) → GNN apprend seul
- Dual Encoder : GNN + ChemBERT entraînés ensemble → espace commun aligné

Architecture :
    Molécule (graph) → GNN → projection → embedding normalisé
    Description (text) → ChemBERT → projection → embedding normalisé
    → Contrastive loss (InfoNCE / Triplet / MSE)
"""

import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from torch_geometric.data import Batch
from torch_geometric.nn import GPSConv, GINEConv, global_add_pool

from transformers import AutoTokenizer, AutoModel

from data_utils import x_map, e_map

import pickle


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
# DATASET POUR DUAL ENCODER
# =========================================================
class DualEncoderDataset(Dataset):
    """Dataset qui retourne (graph, description_text)."""
    
    def __init__(self, graphs_path):
        with open(graphs_path, 'rb') as f:
            self.graphs = pickle.load(f)
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        graph = self.graphs[idx]
        return graph, graph.description  # (PyG Data, str)


def collate_dual_encoder(batch):
    """Collate pour Dual Encoder : (graphs batched, texts list)."""
    graphs, texts = zip(*batch)
    batched_graphs = Batch.from_data_list(graphs)
    return batched_graphs, list(texts)


# =========================================================
# ENCODERS
# =========================================================
ATOM_FEATURE_DIMS = [
    len(x_map['atomic_num']), len(x_map['chirality']), len(x_map['degree']),
    len(x_map['formal_charge']), len(x_map['num_hs']), 
    len(x_map['num_radical_electrons']), len(x_map['hybridization']), 
    len(x_map['is_aromatic']), len(x_map['is_in_ring']),
]
BOND_FEATURE_DIMS = [
    len(e_map['bond_type']), len(e_map['stereo']), len(e_map['is_conjugated']),
]


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
# GRAPH ENCODER (MolGNN)
# =========================================================
class MolGNN(nn.Module):
    """Graph Neural Network avec GPS layers."""
    
    def __init__(self, hidden_dim=256, out_dim=768, num_layers=4, 
                 num_heads=4, dropout=0.1):
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
# TEXT ENCODER (ChemBERT)
# =========================================================
class TextEncoder(nn.Module):
    """Text encoder avec ChemBERT + projection."""
    
    def __init__(self, model_name='DeepChem/ChemBERTa-77M-MLM', 
                 out_dim=768, dropout=0.1, freeze_bert=False, 
                 pooling='cls'):
        super().__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        self.pooling = pooling
        
        # Option : freeze BERT (pour économiser compute)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        bert_dim = self.bert.config.hidden_size
        
        self.proj = nn.Sequential(
            nn.Linear(bert_dim, bert_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bert_dim, out_dim),
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        if self.pooling == 'mean':
            # Mean pooling
            mask = attention_mask.unsqueeze(-1).expand(
                outputs.last_hidden_state.size()
            ).float()
            sum_embeddings = torch.sum(outputs.last_hidden_state * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            pooled = sum_embeddings / sum_mask
        else:
            # CLS pooling
            pooled = outputs.last_hidden_state[:, 0, :]
        
        projected = self.proj(pooled)
        return F.normalize(projected, dim=-1)


# =========================================================
# DUAL ENCODER COMPLET
# =========================================================
class DualEncoder(nn.Module):
    """Dual Encoder : GNN + ChemBERT entraînés ensemble."""
    
    def __init__(self, graph_encoder, text_encoder):
        super().__init__()
        self.graph_encoder = graph_encoder
        self.text_encoder = text_encoder
    
    def forward(self, graphs, input_ids, attention_mask):
        mol_emb = self.graph_encoder(graphs)
        txt_emb = self.text_encoder(input_ids, attention_mask)
        return mol_emb, txt_emb


# =========================================================
# TRAINING
# =========================================================
def train_epoch(dual_encoder, loader, tokenizer, optimizer, device, 
                loss_type='infonce', max_length=128, **loss_kwargs):
    """Entraîne une époque du Dual Encoder."""
    dual_encoder.train()
    total_loss, total = 0.0, 0
    
    for graphs, texts in loader:
        graphs = graphs.to(device)
        
        # Tokenize texts
        inputs = tokenizer(
            texts, 
            return_tensors='pt', 
            truncation=True, 
            max_length=max_length, 
            padding=True
        )
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        # Forward pass
        mol_vec, txt_vec = dual_encoder(graphs, input_ids, attention_mask)
        
        # Loss
        if loss_type == 'mse':
            loss = F.mse_loss(mol_vec, txt_vec)
        elif loss_type == 'infonce':
            loss = infonce_loss(mol_vec, txt_vec, **loss_kwargs)
        elif loss_type == 'triplet':
            loss = triplet_loss(mol_vec, txt_vec, **loss_kwargs)
        else:
            raise ValueError(f"Loss inconnue: {loss_type}")
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dual_encoder.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * graphs.num_graphs
        total += graphs.num_graphs
    
    return total_loss / total


# =========================================================
# EVALUATION
# =========================================================
@torch.no_grad()
def eval_retrieval(data_path, dual_encoder, tokenizer, device, max_length=128):
    """Évalue sur la tâche de retrieval (MRR, R@k)."""
    dual_encoder.eval()
    
    dataset = DualEncoderDataset(data_path)
    loader = DataLoader(
        dataset, batch_size=64, shuffle=False, 
        collate_fn=collate_dual_encoder
    )
    
    all_mol, all_txt = [], []
    
    for graphs, texts in loader:
        graphs = graphs.to(device)
        inputs = tokenizer(
            texts, return_tensors='pt', truncation=True, 
            max_length=max_length, padding=True
        )
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        mol_emb, txt_emb = dual_encoder(graphs, input_ids, attention_mask)
        all_mol.append(mol_emb)
        all_txt.append(txt_emb)
    
    all_mol = torch.cat(all_mol)
    all_txt = torch.cat(all_txt)
    
    # Similarités text->mol (retrieval)
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
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description='Train Dual Encoder (GNN + ChemBERT)'
    )
    
    # === Loss configuration ===
    parser.add_argument('--loss', type=str, default='infonce', 
                        choices=['mse', 'infonce', 'triplet'])
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--margin', type=float, default=0.2)
    
    # === Model architecture ===
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Hidden dim du GNN')
    parser.add_argument('--emb-dim', type=int, default=768,
                        help='Dimension finale des embeddings (GNN et BERT)')
    parser.add_argument('--num-layers', type=int, default=4,
                        help='Nombre de layers GPS')
    parser.add_argument('--num-heads', type=int, default=4,
                        help='Nombre de attention heads')
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # === Text encoder ===
    parser.add_argument('--text-model', type=str, 
                        default='DeepChem/ChemBERTa-77M-MLM',
                        help='Nom du modèle HuggingFace (ChemBERT, BERT, etc.)')
    parser.add_argument('--freeze-bert', action='store_true',
                        help='Freeze BERT (entraîne seulement projection)')
    parser.add_argument('--pooling', type=str, default='cls',
                        choices=['cls', 'mean'],
                        help='Pooling strategy pour BERT')
    parser.add_argument('--max-length', type=int, default=128,
                        help='Max token length pour tokenizer')
    
    # === Training hyperparameters ===
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate global')
    parser.add_argument('--lr-bert', type=float, default=1e-5,
                        help='Learning rate pour BERT (plus petit)')
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    
    # === Data paths ===
    parser.add_argument('--data-dir', type=str, default='data')
    
    # === Checkpointing ===
    parser.add_argument('--save-prefix', type=str, default='dual_encoder',
                        help='Préfixe pour sauvegarder les checkpoints')
    
    args = parser.parse_args()
    
    # =========================================================
    # SETUP
    # =========================================================
    file_path = Path(os.path.abspath(__file__))
    parent_folder = file_path.parent
    base_path = parent_folder / args.data_dir
    
    TRAIN_GRAPHS = str(base_path / "train_graphs.pkl")
    VAL_GRAPHS = str(base_path / "validation_graphs.pkl")
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 70)
    print(f"DUAL ENCODER TRAINING - {args.loss.upper()} Loss")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"\n=== Architecture ===")
    print(f"GNN Hidden Dim: {args.hidden_dim}")
    print(f"Embedding Dim: {args.emb_dim}")
    print(f"GNN Layers: {args.num_layers}, Heads: {args.num_heads}")
    print(f"Text Model: {args.text_model}")
    print(f"BERT Frozen: {args.freeze_bert}")
    print(f"Pooling: {args.pooling}")
    print(f"\n=== Loss ===")
    print(f"Loss Type: {args.loss}")
    if args.loss == 'infonce':
        print(f"  Temperature: {args.temperature}")
    elif args.loss == 'triplet':
        print(f"  Margin: {args.margin}")
    print(f"\n=== Training ===")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"LR (GNN): {args.lr}, LR (BERT): {args.lr_bert}")
    print(f"Weight Decay: {args.weight_decay}")
    print("=" * 70)
    
    # =========================================================
    # DATA LOADING
    # =========================================================
    print("\nLoading data...")
    train_ds = DualEncoderDataset(TRAIN_GRAPHS)
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_dual_encoder
    )
    print(f"Train set: {len(train_ds)} molecules")
    
    # =========================================================
    # MODEL
    # =========================================================
    print("\nInitializing models...")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    
    # Graph encoder
    graph_encoder = MolGNN(
        hidden_dim=args.hidden_dim,
        out_dim=args.emb_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout
    )
    
    # Text encoder
    text_encoder = TextEncoder(
        model_name=args.text_model,
        out_dim=args.emb_dim,
        dropout=args.dropout,
        freeze_bert=args.freeze_bert,
        pooling=args.pooling
    )
    
    # Dual encoder
    dual_encoder = DualEncoder(graph_encoder, text_encoder).to(DEVICE)
    
    # Comptage des paramètres
    total_params = sum(p.numel() for p in dual_encoder.parameters())
    trainable_params = sum(
        p.numel() for p in dual_encoder.parameters() if p.requires_grad
    )
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    
    # =========================================================
    # OPTIMIZER (Learning rates différenciés)
    # =========================================================
    # GNN : LR normal, BERT : LR plus petit (fine-tuning)
    if args.freeze_bert:
        optimizer = torch.optim.AdamW(
            dual_encoder.parameters(), 
            lr=args.lr, 
            weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.AdamW([
            {'params': dual_encoder.graph_encoder.parameters(), 'lr': args.lr},
            {'params': dual_encoder.text_encoder.parameters(), 'lr': args.lr_bert},
        ], weight_decay=args.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    
    # Loss params
    LOSS_PARAMS = {
        'infonce': {'temperature': args.temperature},
        'triplet': {'margin': args.margin},
        'mse': {}
    }
    
    # =========================================================
    # TRAINING LOOP
    # =========================================================
    print("\nStarting training...\n")
    
    best_mrr = 0.0
    patience = 5
    patience_counter = 0
    
    for ep in range(args.epochs):
        # Train
        train_loss = train_epoch(
            dual_encoder, train_dl, tokenizer, optimizer, DEVICE,
            loss_type=args.loss,
            max_length=args.max_length,
            **LOSS_PARAMS[args.loss]
        )
        
        # Eval
        if os.path.exists(VAL_GRAPHS):
            val_scores = eval_retrieval(
                VAL_GRAPHS, dual_encoder, tokenizer, DEVICE, 
                max_length=args.max_length
            )
        else:
            val_scores = {}
        
        print(f"Epoch {ep+1}/{args.epochs} | loss={train_loss:.4f} | {val_scores}")
        
        scheduler.step()
        
        # Checkpointing
        current_mrr = val_scores.get('MRR', 0)
        if current_mrr > best_mrr:
            best_mrr = current_mrr
            patience_counter = 0
            
            # Nom de fichier unique
            filename_parts = [args.save_prefix, args.loss]
            if args.loss == 'infonce':
                filename_parts.append(f"temp{args.temperature}")
            elif args.loss == 'triplet':
                filename_parts.append(f"margin{args.margin}")
            filename_parts.extend([
                f"bs{args.batch_size}",
                f"lr{args.lr}",
                f"hd{args.hidden_dim}",
                f"emb{args.emb_dim}",
                f"nl{args.num_layers}"
            ])
            if args.freeze_bert:
                filename_parts.append("frozen")
            
            filename = "_".join(filename_parts) + ".pt"
            save_path = str(base_path / filename)
            
            # Sauvegarde complète (dual encoder + tokenizer)
            torch.save({
                'dual_encoder': dual_encoder.state_dict(),
                'args': vars(args),
                'epoch': ep + 1,
                'best_mrr': best_mrr,
            }, save_path)
            
            print(f"  → New best MRR: {best_mrr:.4f} | Saved: {filename}")
        else:
            patience_counter += 1
            print(f"  → No improvement ({patience_counter}/{patience})")
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {ep+1}")
            break
    
    print(f"\nTraining done. Best MRR: {best_mrr:.4f}")


if __name__ == "__main__":
    main()
