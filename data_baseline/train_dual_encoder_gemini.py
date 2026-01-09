"""
train_dual_encoder.py - Fine-tuning conjoint (GPS + Recobo ChemBERT)
Adapté pour l'environnement BornToOverfit/data_baseline
"""

import os
import argparse
import pickle
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler

from torch_geometric.data import Batch
from torch_geometric.nn import GPSConv, GINEConv, global_add_pool
from transformers import AutoModel, AutoTokenizer

# Import depuis data_utils si présent, sinon on définit les maps ici
try:
    from data_utils import x_map, e_map
except ImportError:
    # Fallback si data_utils n'est pas accessible directement
    print("⚠️ data_utils non trouvé, utilisation des maps par défaut.")
    # (Je garde les longueurs standards basées sur ton PDF si besoin)
    pass

# =========================================================
# 1. DATASET SPÉCIAL DUAL ENCODER
# =========================================================

class RawTextGraphDataset(Dataset):
    """
    Charge les graphes et les descriptions brutes depuis les .pkl.
    Ne charge PAS les embeddings .pt (car on va les apprendre).
    """
    def __init__(self, pkl_path):
        self.pkl_path = Path(pkl_path)
        if not self.pkl_path.exists():
            raise FileNotFoundError(f"Fichier non trouvé : {self.pkl_path}")
            
        print(f"📥 Chargement de {self.pkl_path.name}...")
        with open(self.pkl_path, 'rb') as f:
            self.data_list = pickle.load(f)
        print(f"✅ Chargé {len(self.data_list)} molécules.")

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data = self.data_list[idx]
        # On extrait le texte brut
        text = data.description if hasattr(data, 'description') else ""
        return data, text

class DualCollate:
    """Batch les graphes (PyG) et tokenise le texte (HF) à la volée."""
    def __init__(self, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, batch):
        graphs, texts = zip(*batch)
        
        # 1. Batcher les graphes
        batched_graphs = Batch.from_data_list(graphs)
        
        # 2. Tokenizer les textes
        text_inputs = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        
        return batched_graphs, text_inputs

# =========================================================
# 2. MODÈLES (GPS + ChemBERT)
# =========================================================

# Définition des dimensions (hardcodé pour éviter les dépendances circulaires si data_utils manque)
ATOM_DIMS = [119, 4, 11, 12, 9, 5, 8, 2, 2] # Basé sur x_map
BOND_DIMS = [22, 6, 2] # Basé sur e_map

class AtomEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(dim, hidden_dim) for dim in ATOM_DIMS])
    def forward(self, x):
        return sum(emb(x[:, i]) for i, emb in enumerate(self.embeddings))

class BondEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(dim, hidden_dim) for dim in BOND_DIMS])
    def forward(self, edge_attr):
        return sum(emb(edge_attr[:, i]) for i, emb in enumerate(self.embeddings))

class MolGNN(nn.Module):
    """Encodeur de Graphe (GPS)"""
    def __init__(self, hidden_dim=256, out_dim=768, num_layers=4, num_heads=4, dropout=0.1):
        super().__init__()
        self.atom_encoder = AtomEncoder(hidden_dim)
        self.bond_encoder = BondEncoder(hidden_dim)
        self.convs = nn.ModuleList()
        
        for _ in range(num_layers):
            local_nn = nn.Sequential(
                nn.Linear(hidden_dim, 2 * hidden_dim), nn.BatchNorm1d(2 * hidden_dim), nn.ReLU(),
                nn.Linear(2 * hidden_dim, hidden_dim),
            )
            conv = GPSConv(hidden_dim, GINEConv(local_nn, train_eps=True, edge_dim=hidden_dim), 
                           heads=num_heads, dropout=dropout, attn_type='multihead')
            self.convs.append(conv)
            
        self.pool = global_add_pool
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, batch):
        h = self.atom_encoder(batch.x)
        edge_attr = self.bond_encoder(batch.edge_attr)
        for conv in self.convs:
            h = conv(h, batch.edge_index, batch.batch, edge_attr=edge_attr)
        return self.proj(self.pool(h, batch.batch))

class DualEncoder(nn.Module):
    """Modèle Fusionné : GNN + ChemBERT"""
    def __init__(self, model_name, gnn_args):
        super().__init__()
        self.graph_encoder = MolGNN(**gnn_args)
        
        print(f"⚛️ Chargement de {model_name}...")
        self.text_encoder = AutoModel.from_pretrained(model_name)
        
        # Projection si la dimension de BERT != dimension de sortie commune (768)
        bert_dim = self.text_encoder.config.hidden_size
        out_dim = gnn_args['out_dim']
        
        self.text_proj = nn.Linear(bert_dim, out_dim) if bert_dim != out_dim else nn.Identity()

    def forward(self, batch_graphs, text_inputs):
        # --- Graph Branch ---
        g_emb = self.graph_encoder(batch_graphs) 
        
        # --- Text Branch ---
        t_out = self.text_encoder(**text_inputs)
        # On prend le token [CLS] (premier token) pour la représentation de la phrase
        t_emb = self.text_proj(t_out.last_hidden_state[:, 0, :])
        
        # Normalisation L2 (Crucial pour la Cosine Similarity)
        return F.normalize(g_emb, dim=-1), F.normalize(t_emb, dim=-1)

# =========================================================
# 3. LOSS & TRAINING
# =========================================================

def triplet_loss(mol_vec, txt_vec, margin=0.2):
    """Triplet loss bidirectionnelle avec Hard Negative Mining"""
    sims = mol_vec @ txt_vec.t()
    pos_sims = sims.diag().unsqueeze(1)
    
    mask = torch.eye(sims.size(0), device=sims.device).bool()
    neg_sims = sims.masked_fill(mask, -float('inf'))
    
    # On prend le négatif le plus difficile (celui qui ressemble le plus mais qui est faux)
    hard_neg_m2t = neg_sims.max(dim=1, keepdim=True)[0]
    hard_neg_t2m = neg_sims.max(dim=0, keepdim=True)[0].t()
    
    loss = F.relu(margin + hard_neg_m2t - pos_sims).mean() + \
           F.relu(margin + hard_neg_t2m - pos_sims).mean()
    return loss / 2

def main():
    parser = argparse.ArgumentParser(description="Entraînement Dual Encoder (GNN + ChemBERT)")
    
    # Arguments alignés avec ton notebook
    parser.add_argument('--data_dir', type=str, default='data_baseline/data', 
                        help="Dossier contenant les .pkl")
    parser.add_argument('--model_name', type=str, default='recobo/chemical-bert-uncased',
                        help="Modèle HF à utiliser")
    
    # Hyperparamètres d'entraînement
    parser.add_argument('--batch_size', type=int, default=16, 
                        help="Batch size physique (ce qui rentre en VRAM)")
    parser.add_argument('--grad_accum', type=int, default=8, 
                        help="Accumulation de gradients (BS effectif = batch_size * grad_accum)")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr_gnn', type=float, default=5e-4)
    parser.add_argument('--lr_bert', type=float, default=1e-5, 
                        help="LR très faible pour ne pas casser le pré-entraînement BERT")
    parser.add_argument('--margin', type=float, default=0.2)
    
    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Démarrage sur {device}")
    print(f"📦 Batch Size Effectif : {args.batch_size * args.grad_accum}")
    
    # Paths
    base_path = Path(args.data_dir)
    train_pkl = base_path / "train_graphs.pkl"
    
    # Data Loading
    print("⏳ Préparation du Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    collate_fn = DualCollate(tokenizer)
    
    train_ds = RawTextGraphDataset(train_pkl)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)

    # Model
    gnn_config = {'hidden_dim': 256, 'out_dim': 768, 'num_layers': 4, 'num_heads': 4}
    model = DualEncoder(args.model_name, gnn_config).to(device)

    # Optimizer (Séparation des learning rates)
    optimizer = torch.optim.AdamW([
        {'params': model.graph_encoder.parameters(), 'lr': args.lr_gnn},
        {'params': model.text_encoder.parameters(), 'lr': args.lr_bert}
    ], weight_decay=1e-4)
    
    scaler = GradScaler() # Pour AMP (Mixed Precision)

    # Loop
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        optimizer.zero_grad()
        
        for step, (graphs, text_inputs) in enumerate(train_loader):
            # Move to device
            graphs = graphs.to(device)
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            
            # Forward & Loss (in mixed precision)
            with autocast():
                g_emb, t_emb = model(graphs, text_inputs)
                loss = triplet_loss(g_emb, t_emb, margin=args.margin)
                loss = loss / args.grad_accum # Normalisation pour accumulation

            # Backward
            scaler.scale(loss).backward()
            
            # Step (si accumulation finie)
            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            total_loss += loss.item() * args.grad_accum
            
            if step % 50 == 0:
                print(f"Epoch {epoch+1} | Step {step} | Loss: {loss.item() * args.grad_accum:.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"=== Epoch {epoch+1} Terminé | Avg Loss: {avg_loss:.4f} ===")
        
        # Save
        save_name = f"dual_encoder_recobo_ep{epoch+1}.pt"
        torch.save(model.state_dict(), base_path / save_name)
        print(f"💾 Sauvegardé : {save_name}")

if __name__ == "__main__":
    main()