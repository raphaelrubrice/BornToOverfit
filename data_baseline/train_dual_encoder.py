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
from torch.optim.lr_scheduler import OneCycleLR

from torch_geometric.data import Batch
from torch_geometric.nn import GPSConv, GINEConv, global_add_pool
from transformers import AutoModel, AutoTokenizer

# --- DATASET ---
class RawTextGraphDataset(Dataset):
    def __init__(self, pkl_path):
        self.pkl_path = Path(pkl_path)
        with open(self.pkl_path, 'rb') as f:
            self.data_list = pickle.load(f)
    def __len__(self): return len(self.data_list)
    def __getitem__(self, idx):
        data = self.data_list[idx]
        text = data.description if hasattr(data, 'description') else ""
        return data, text

class DualCollate:
    def __init__(self, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __call__(self, batch):
        graphs, texts = zip(*batch)
        batched_graphs = Batch.from_data_list(graphs)
        text_inputs = self.tokenizer(list(texts), padding=True, truncation=True, 
                                     max_length=self.max_len, return_tensors="pt")
        return batched_graphs, text_inputs

# --- MODELS ---
ATOM_DIMS = [119, 4, 11, 12, 9, 5, 8, 2, 2]
BOND_DIMS = [22, 6, 2]

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
    def __init__(self, hidden_dim=256, out_dim=768, num_layers=4, num_heads=4, dropout=0.1):
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
            self.convs.append(GPSConv(
                hidden_dim, 
                GINEConv(local_nn, train_eps=True, edge_dim=hidden_dim), 
                heads=num_heads, 
                dropout=dropout, 
                attn_type='multihead'
            ))
        self.pool = global_add_pool
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, batch):
        h = self.atom_encoder(batch.x)
        edge_attr = self.bond_encoder(batch.edge_attr)
        for conv in self.convs: 
            h = conv(h, batch.edge_index, batch.batch, edge_attr=edge_attr)
        return self.proj(self.pool(h, batch.batch))

class DualEncoder(nn.Module):
    def __init__(self, model_name, gnn_args, freeze_layers=0):
        super().__init__()
        self.graph_encoder = MolGNN(**gnn_args)
        self.text_encoder = AutoModel.from_pretrained(model_name)
        
        if freeze_layers > 0:
            print(f"❄️ Gel des {freeze_layers} premières couches de BERT")
            for param in self.text_encoder.embeddings.parameters(): 
                param.requires_grad = False
            for i in range(freeze_layers):
                if i < len(self.text_encoder.encoder.layer):
                    for param in self.text_encoder.encoder.layer[i].parameters(): 
                        param.requires_grad = False
        
        bert_dim = self.text_encoder.config.hidden_size
        out_dim = gnn_args['out_dim']
        self.text_proj = nn.Linear(bert_dim, out_dim) if bert_dim != out_dim else nn.Identity()

    def forward(self, batch_graphs, text_inputs):
        g_emb = self.graph_encoder(batch_graphs) 
        t_out = self.text_encoder(**text_inputs)
        t_emb = self.text_proj(t_out.last_hidden_state[:, 0, :])
        return F.normalize(g_emb, dim=-1), F.normalize(t_emb, dim=-1)

# --- LOSS ---
def triplet_loss(mol_vec, txt_vec, margin=0.2):
    sims = mol_vec @ txt_vec.t()
    mask = torch.eye(sims.size(0), device=sims.device).bool()
    neg_sims = sims.masked_fill(mask, -1e9)  # ← Changé de -inf à -1e9
    hard_neg_m2t = neg_sims.max(dim=1, keepdim=True)[0]
    hard_neg_t2m = neg_sims.max(dim=0, keepdim=True)[0].t()
    pos_sims = sims.diag().unsqueeze(1)
    loss = (F.relu(margin + hard_neg_m2t - pos_sims).mean() + 
            F.relu(margin + hard_neg_t2m - pos_sims).mean())
    return loss / 2

# --- EVALUATION ---
@torch.no_grad()
def evaluate_retrieval(model, val_loader, device):
    """Évalue le MRR sur le validation set."""
    model.eval()
    all_g_emb, all_t_emb = [], []
    
    for graphs, text_inputs in val_loader:
        graphs = graphs.to(device)
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        
        g_emb, t_emb = model(graphs, text_inputs)
        all_g_emb.append(g_emb)
        all_t_emb.append(t_emb)
    
    all_g_emb = torch.cat(all_g_emb, dim=0)
    all_t_emb = torch.cat(all_t_emb, dim=0)
    
    # Text-to-Molecule retrieval
    sims = all_t_emb @ all_g_emb.t()  # [N, N]
    ranks = sims.argsort(dim=-1, descending=True)
    
    N = sims.size(0)
    correct = torch.arange(N, device=sims.device)
    positions = (ranks == correct.unsqueeze(1)).nonzero()[:, 1] + 1
    
    mrr = (1.0 / positions.float()).mean().item()
    r1 = (positions <= 1).float().mean().item()
    r5 = (positions <= 5).float().mean().item()
    r10 = (positions <= 10).float().mean().item()
    
    return {'MRR': mrr, 'R@1': r1, 'R@5': r5, 'R@10': r10}

# --- MAIN ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data_baseline/data')
    parser.add_argument('--model_name', type=str, default='recobo/chemical-bert-uncased')
    
    # Paramètres à varier
    parser.add_argument('--lr_gnn', type=float, default=8e-4)
    parser.add_argument('--lr_bert', type=float, default=3e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--freeze_layers', type=int, default=0)
    parser.add_argument('--margin', type=float, default=0.2)
    
    # Training
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--grad_accum', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--patience', type=int, default=15)
    
    args = parser.parse_args()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 70)
    print("🚀 DUAL ENCODER TRAINING - OPTIMIZED")
    print("=" * 70)
    print(f"LR GNN      : {args.lr_gnn}")
    print(f"LR BERT     : {args.lr_bert}")
    print(f"Weight Decay: {args.weight_decay}")
    print(f"Freeze Layers: {args.freeze_layers}")
    print(f"Margin      : {args.margin}")
    print(f"Batch Size  : {args.batch_size} × {args.grad_accum} = {args.batch_size * args.grad_accum} (effective)")
    print("=" * 70)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Train loader
    train_loader = DataLoader(
        RawTextGraphDataset(Path(args.data_dir) / "train_graphs.pkl"), 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=DualCollate(tokenizer), 
        num_workers=2, 
        pin_memory=True
    )
    
    # Validation loader
    val_loader = DataLoader(
        RawTextGraphDataset(Path(args.data_dir) / "validation_graphs.pkl"), 
        batch_size=64,  # Plus gros batch pour eval
        shuffle=False, 
        collate_fn=DualCollate(tokenizer), 
        num_workers=2, 
        pin_memory=True
    )
    
    gnn_config = {
        'hidden_dim': 256, 
        'out_dim': 768, 
        'num_layers': 4, 
        'num_heads': 4, 
        'dropout': 0.1
    }
    model = DualEncoder(args.model_name, gnn_config, freeze_layers=args.freeze_layers).to(DEVICE)
    
    # Optimizer avec weight decay différencié
    optimizer = torch.optim.AdamW([
        {'params': model.graph_encoder.parameters(), 
         'lr': args.lr_gnn, 
         'weight_decay': args.weight_decay},
        {'params': model.text_encoder.parameters(), 
         'lr': args.lr_bert, 
         'weight_decay': args.weight_decay / 10},  # ← Plus faible pour BERT
        {'params': model.text_proj.parameters(), 
         'lr': args.lr_bert, 
         'weight_decay': args.weight_decay}
    ])
    
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=[args.lr_gnn, args.lr_bert, args.lr_bert], 
        steps_per_epoch=len(train_loader) // args.grad_accum, 
        epochs=args.epochs, 
        pct_start=0.1
    )

    scaler = GradScaler()
    best_mrr = 0.0  # ← Changé de best_loss à best_mrr
    patience_counter = 0

    for epoch in range(args.epochs):
        # === TRAINING ===
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        for step, (graphs, text_inputs) in enumerate(train_loader):
            graphs = graphs.to(DEVICE)
            text_inputs = {k: v.to(DEVICE) for k, v in text_inputs.items()}
            
            with autocast():
                g_emb, t_emb = model(graphs, text_inputs)
                loss = triplet_loss(g_emb, t_emb, margin=args.margin) / args.grad_accum
            
            scaler.scale(loss).backward()
            
            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            
            total_loss += loss.item() * args.grad_accum

        avg_loss = total_loss / len(train_loader)
        
        # === VALIDATION ===
        val_metrics = evaluate_retrieval(model, val_loader, DEVICE)
        
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {avg_loss:.4f} | "
              f"Val MRR: {val_metrics['MRR']:.4f} | "
              f"R@1: {val_metrics['R@1']:.4f} | "
              f"R@5: {val_metrics['R@5']:.4f}")
        
        # === EARLY STOPPING (basé sur VAL MRR) ===
        if val_metrics['MRR'] > best_mrr:
            best_mrr = val_metrics['MRR']
            patience_counter = 0
            
            # Nom de fichier complet
            effective_bs = args.batch_size * args.grad_accum
            save_name = (f"dual_lrGNN{args.lr_gnn}_lrBERT{args.lr_bert}_"
                        f"wd{args.weight_decay}_frz{args.freeze_layers}_"
                        f"margin{args.margin}_bs{effective_bs}.pt")
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_mrr': best_mrr,
                'args': vars(args),
                'val_metrics': val_metrics
            }, Path(args.data_dir) / save_name)
            
            print(f"  💾 New Best MRR: {best_mrr:.4f} | Saved: {save_name}")
        else:
            patience_counter += 1
            print(f"  → No improvement ({patience_counter}/{args.patience})")
            
            if patience_counter >= args.patience:
                print("🛑 Early stopping")
                break
    
    print("\n" + "=" * 70)
    print(f"✅ Training done. Best Val MRR: {best_mrr:.4f}")
    print("=" * 70)

if __name__ == "__main__":
    main()