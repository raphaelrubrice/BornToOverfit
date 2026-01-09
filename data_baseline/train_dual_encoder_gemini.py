"""
train_dual_encoder.py - Fine-tuning conjoint (GPS + ChemBERT)
"""

import os
import argparse
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler  # Pour Mixed Precision

from torch_geometric.data import Batch
from torch_geometric.nn import GPSConv, GINEConv, global_add_pool
from transformers import AutoModel, AutoTokenizer

from data_utils import PreprocessedGraphDataset, x_map, e_map

# =========================================================
# 1. DATASET & COLLATE (Adapté pour Texte Brut)
# =========================================================

class TextGraphDataset(PreprocessedGraphDataset):
    """Surcharge pour renvoyer le texte brut au lieu d'un embedding lookup."""
    def __init__(self, file_path):
        # On n'a plus besoin du dictionnaire d'embeddings ici
        super().__init__(file_path, embedding_dict=None)
    
    def get(self, idx):
        data = self.data_list[idx]
        # On renvoie le graphe et sa description textuelle
        # Note: Pour le set de test, description peut être vide, à gérer si besoin
        text = data.description if hasattr(data, 'description') else ""
        return data, text

class DualCollate:
    """Gère le batching des graphes ET la tokenisation du texte."""
    def __init__(self, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, batch):
        graphs, texts = zip(*batch)
        
        # 1. Batcher les graphes (PyTorch Geometric)
        batched_graphs = Batch.from_data_list(graphs)
        
        # 2. Tokenizer les textes (HuggingFace)
        text_inputs = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        
        return batched_graphs, text_inputs

# =========================================================
# 2. MODELS (GPS + BERT Wrapper)
# =========================================================

# --- Copie de ton Atom/Bond Encoder ---
ATOM_FEATURE_DIMS = [len(x_map[k]) for k in ['atomic_num', 'chirality', 'degree', 'formal_charge', 'num_hs', 'num_radical_electrons', 'hybridization', 'is_aromatic', 'is_in_ring']]
BOND_FEATURE_DIMS = [len(e_map[k]) for k in ['bond_type', 'stereo', 'is_conjugated']]

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
    """Ton modèle GPS existant"""
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
            self.convs.append(GPSConv(hidden_dim, GINEConv(local_nn, train_eps=True, edge_dim=hidden_dim), heads=num_heads, dropout=dropout, attn_type='multihead'))
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
        g = self.pool(h, batch.batch)
        return self.proj(g)

class DualEncoder(nn.Module):
    """Le modèle complet qui contient GNN et BERT"""
    def __init__(self, model_name, gnn_args):
        super().__init__()
        # 1. Graph Encoder
        self.graph_encoder = MolGNN(**gnn_args)
        
        # 2. Text Encoder (BERT)
        print(f"Loading HF Model: {model_name}")
        self.text_encoder = AutoModel.from_pretrained(model_name)
        
        # Projection si la dimension de BERT != dimension de sortie commune
        bert_dim = self.text_encoder.config.hidden_size
        out_dim = gnn_args['out_dim']
        
        if bert_dim != out_dim:
            self.text_proj = nn.Linear(bert_dim, out_dim)
        else:
            self.text_proj = nn.Identity()

    def forward(self, batch_graphs, text_inputs):
        # --- Graph Branch ---
        g_emb = self.graph_encoder(batch_graphs) # [B, dim]
        
        # --- Text Branch ---
        # input_ids, attention_mask sont dans text_inputs
        t_outputs = self.text_encoder(**text_inputs)
        
        # Strategy: CLS Token (premier token)
        cls_token = t_outputs.last_hidden_state[:, 0, :] 
        t_emb = self.text_proj(cls_token) # [B, dim]
        
        # Normalisation L2 (Essentiel pour Cosine Similarity / Triplet)
        return F.normalize(g_emb, dim=-1), F.normalize(t_emb, dim=-1)

# =========================================================
# 3. LOSS (Triplet)
# =========================================================
def triplet_loss(mol_vec, txt_vec, margin=0.2):
    """Triplet loss bidirectionnelle avec Hard Negative Mining"""
    sims = mol_vec @ txt_vec.t() # [B, B]
    pos_sims = sims.diag().unsqueeze(1) # [B, 1]
    
    # Mask diagonale pour trouver les negatifs
    mask = torch.eye(sims.size(0), device=sims.device).bool()
    neg_sims = sims.masked_fill(mask, -float('inf'))
    
    # Hardest negatives
    hard_neg_m2t = neg_sims.max(dim=1, keepdim=True)[0]
    hard_neg_t2m = neg_sims.max(dim=0, keepdim=True)[0].t()
    
    loss_m2t = F.relu(margin + hard_neg_m2t - pos_sims).mean()
    loss_t2m = F.relu(margin + hard_neg_t2m - pos_sims).mean()
    
    return (loss_m2t + loss_t2m) / 2

# =========================================================
# 4. TRAINING LOOP
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32, help="Physical batch size (limité par VRAM)")
    parser.add_argument('--grad-accum', type=int, default=4, help="Accumulate gradients to simulate larger batch (ex: 32*4=128)")
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr-gnn', type=float, default=5e-4)
    parser.add_argument('--lr-bert', type=float, default=2e-5, help="Lower LR for BERT to avoid forgetting")
    parser.add_argument('--margin', type=float, default=0.2)
    parser.add_argument('--model-name', type=str, default='bert-base-uncased') # Ou ton ChemBERT path
    args = parser.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {DEVICE} with simulated batch size {args.batch_size * args.grad_accum}")

    # --- Data ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    collate = DualCollate(tokenizer)
    
    # Chemins (A adapter selon ton dossier)
    base_path = Path('data') 
    train_ds = TextGraphDataset(base_path / "train_graphs.pkl")
    # Pour la val, on peut garder l'ancienne méthode (GNN vs Text Embeddings fixes) 
    # ou tout re-calculer. Pour simplifier ici, on entraîne juste.
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                              collate_fn=collate, num_workers=4, pin_memory=True)

    # --- Model ---
    gnn_config = {'hidden_dim': 256, 'out_dim': 768, 'num_layers': 4, 'num_heads': 4}
    model = DualEncoder(args.model_name, gnn_config).to(DEVICE)

    # --- Optimizer avec param groups ---
    # On sépare les paramètres pour appliquer des LR différents
    bert_params = list(map(id, model.text_encoder.parameters()))
    gnn_params = filter(lambda p: id(p) not in bert_params, model.parameters())
    
    optimizer = torch.optim.AdamW([
        {'params': gnn_params, 'lr': args.lr_gnn},
        {'params': model.text_encoder.parameters(), 'lr': args.lr_bert}
    ], weight_decay=1e-4)
    
    scaler = GradScaler() # Pour Mixed Precision

    # --- Loop ---
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        optimizer.zero_grad()
        
        for step, (graphs, text_inputs) in enumerate(train_loader):
            graphs = graphs.to(DEVICE)
            text_inputs = {k: v.to(DEVICE) for k, v in text_inputs.items()}
            
            # Mixed Precision Context
            with autocast():
                g_emb, t_emb = model(graphs, text_inputs)
                loss = triplet_loss(g_emb, t_emb, margin=args.margin)
                loss = loss / args.grad_accum # Normaliser la loss pour l'accumulation

            # Backward pass avec Scaler
            scaler.scale(loss).backward()
            
            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            total_loss += loss.item() * args.grad_accum # Pour l'affichage
            
            if step % 50 == 0:
                print(f"Epoch {epoch+1} | Step {step} | Loss: {loss.item() * args.grad_accum:.4f}")

        print(f"=== Epoch {epoch+1} Avg Loss: {total_loss / len(train_loader):.4f} ===")
        
        # Save checkpoint
        torch.save(model.state_dict(), f"dual_encoder_ep{epoch+1}.pt")

if __name__ == "__main__":
    main()