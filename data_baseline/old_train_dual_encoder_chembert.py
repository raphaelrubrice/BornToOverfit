"""
train_dual_encoder_chembert.py - Dual Encoder (GPS + ChemicalBERT entraînés ensemble)

═══════════════════════════════════════════════════════════════════════════════
POURQUOI LE DUAL ENCODER ?
═══════════════════════════════════════════════════════════════════════════════

Problème avec les embeddings figés:
- BERT génère des embeddings UNE SEULE FOIS (CSV)
- Le GNN s'adapte à cet espace figé
- Avec InfoNCE, le GNN crée un espace bien séparé mais DIFFÉRENT de BERT
- Résultat : bon MRR mais mauvais BLEU sur le retrieval final

Solution Dual Encoder:
- GNN et BERT s'entraînent ENSEMBLE
- Les deux encodeurs s'adaptent mutuellement
- Ils créent un espace commun bien structuré
- Meilleur alignement pour le retrieval final

═══════════════════════════════════════════════════════════════════════════════
CHANGEMENTS vs SCRIPTS PRÉCÉDENTS
═══════════════════════════════════════════════════════════════════════════════

1. PLUS DE FICHIERS CSV - Les embeddings sont calculés à chaque batch
2. BERT ENTRAÎNABLE - Fait partie du modèle
3. DEUX LEARNING RATES - LR faible pour BERT (fine-tuning), LR normal pour GNN
4. NOUVEAU DATASET - Retourne (graph, texte_brut) au lieu de (graph, embedding)
"""

import os
import argparse
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch_geometric.data import Batch
from torch_geometric.nn import GPSConv, GINEConv, global_add_pool

from transformers import AutoTokenizer, AutoModel

from data_utils import x_map, e_map


# =========================================================
# CONFIG
# =========================================================
DEFAULT_CONFIG = {
    'batch_size': 32,      # Réduit car BERT consomme beaucoup de mémoire
    'epochs': 30,
    'lr_gnn': 5e-4,        # Learning rate pour GNN
    'lr_bert': 2e-5,       # Learning rate faible pour BERT (fine-tuning)
    'hidden_dim': 256,
    'num_layers': 4,
    'num_heads': 4,
    'dropout': 0.1,
    'patience': 5,
    'temperature': 0.07,
    'emb_dim': 256,        # Dimension de l'espace commun
    'max_length': 128,
}

# ChemicalBERT - le meilleur selon le benchmark
BERT_MODEL_NAME = "recobo/chemical-bert-uncased"

ATOM_FEATURE_DIMS = [
    len(x_map['atomic_num']), len(x_map['chirality']), len(x_map['degree']),
    len(x_map['formal_charge']), len(x_map['num_hs']), len(x_map['num_radical_electrons']),
    len(x_map['hybridization']), len(x_map['is_aromatic']), len(x_map['is_in_ring']),
]
BOND_FEATURE_DIMS = [
    len(e_map['bond_type']), len(e_map['stereo']), len(e_map['is_conjugated']),
]


# =========================================================
# DATASET : retourne (graphe, texte_brut)
# =========================================================
class GraphTextDataset(Dataset):
    """
    Dataset qui retourne les graphes avec leurs descriptions brutes.
    PAS d'embeddings pré-calculés !
    """
    def __init__(self, graph_path: str):
        print(f"Loading graphs from: {graph_path}")
        with open(graph_path, 'rb') as f:
            self.graphs = pickle.load(f)
        self.descriptions = [g.description for g in self.graphs]
        self.ids = [g.id for g in self.graphs]
        print(f"Loaded {len(self.graphs)} graphs with descriptions")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.descriptions[idx]


def collate_fn_dual(batch, tokenizer, max_length):
    """Collate function qui tokenize les textes."""
    graphs, texts = zip(*batch)
    batch_graphs = Batch.from_data_list(list(graphs))
    
    encoded = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    return batch_graphs, encoded["input_ids"], encoded["attention_mask"]


# =========================================================
# ENCODERS
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


class GraphEncoder(nn.Module):
    """GPS encoder pour les graphes."""
    def __init__(self, hidden_dim=256, out_dim=256, num_layers=4, num_heads=4, dropout=0.1):
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


class TextEncoder(nn.Module):
    """ChemicalBERT encoder pour les textes."""
    def __init__(self, bert_model_name=BERT_MODEL_NAME, out_dim=256, dropout=0.1):
        super().__init__()
        
        self.bert = AutoModel.from_pretrained(bert_model_name)
        bert_dim = self.bert.config.hidden_size
        
        self.proj = nn.Sequential(
            nn.Linear(bert_dim, bert_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bert_dim, out_dim),
        )
    
    def mean_pooling(self, token_embeddings, attention_mask):
        """Mean pooling sur tous les tokens (meilleur que [CLS])."""
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / sum_mask
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.mean_pooling(outputs.last_hidden_state, attention_mask)
        projected = self.proj(pooled)
        return F.normalize(projected, dim=-1)


class DualEncoder(nn.Module):
    """Modèle complet : GNN + BERT."""
    def __init__(self, hidden_dim=256, emb_dim=256, num_layers=4, num_heads=4, dropout=0.1):
        super().__init__()
        self.graph_encoder = GraphEncoder(
            hidden_dim=hidden_dim, out_dim=emb_dim, 
            num_layers=num_layers, num_heads=num_heads, dropout=dropout
        )
        self.text_encoder = TextEncoder(out_dim=emb_dim, dropout=dropout)
    
    def forward(self, batch_graphs, input_ids, attention_mask):
        graph_emb = self.graph_encoder(batch_graphs)
        text_emb = self.text_encoder(input_ids, attention_mask)
        return graph_emb, text_emb


# =========================================================
# INFONCE LOSS
# =========================================================
def info_nce_loss(graph_emb, text_emb, temperature=0.07):
    logits = graph_emb @ text_emb.T / temperature
    labels = torch.arange(len(logits), device=logits.device)
    loss_g2t = F.cross_entropy(logits, labels)
    loss_t2g = F.cross_entropy(logits.T, labels)
    return (loss_g2t + loss_t2g) / 2


# =========================================================
# TRAINING
# =========================================================
def train_epoch(model, loader, optimizer, device, temperature):
    model.train()
    total_loss, total = 0.0, 0
    
    for batch_graphs, input_ids, attention_mask in loader:
        batch_graphs = batch_graphs.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        graph_emb, text_emb = model(batch_graphs, input_ids, attention_mask)
        loss = info_nce_loss(graph_emb, text_emb, temperature)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * batch_graphs.num_graphs
        total += batch_graphs.num_graphs
    
    return total_loss / total


@torch.no_grad()
def eval_retrieval(model, data_path, tokenizer, device, max_length):
    model.eval()
    dataset = GraphTextDataset(data_path)
    
    all_graph_emb = []
    all_text_emb = []
    
    batch_size = 64
    for i in range(0, len(dataset), batch_size):
        batch_data = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
        graphs, texts = zip(*batch_data)
        
        batch_graphs = Batch.from_data_list(list(graphs)).to(device)
        encoded = tokenizer(
            list(texts), padding=True, truncation=True,
            max_length=max_length, return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        
        graph_emb, text_emb = model(batch_graphs, input_ids, attention_mask)
        all_graph_emb.append(graph_emb)
        all_text_emb.append(text_emb)
    
    all_graph_emb = torch.cat(all_graph_emb, dim=0)
    all_text_emb = torch.cat(all_text_emb, dim=0)
    
    sims = all_text_emb @ all_graph_emb.T
    ranks = sims.argsort(dim=-1, descending=True)
    N = all_text_emb.size(0)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--epochs', type=int, default=DEFAULT_CONFIG['epochs'])
    parser.add_argument('--batch_size', type=int, default=DEFAULT_CONFIG['batch_size'])
    parser.add_argument('--lr_gnn', type=float, default=DEFAULT_CONFIG['lr_gnn'])
    parser.add_argument('--lr_bert', type=float, default=DEFAULT_CONFIG['lr_bert'])
    parser.add_argument('--temperature', type=float, default=DEFAULT_CONFIG['temperature'])
    parser.add_argument('--hidden_dim', type=int, default=DEFAULT_CONFIG['hidden_dim'])
    parser.add_argument('--emb_dim', type=int, default=DEFAULT_CONFIG['emb_dim'])
    parser.add_argument('--num_layers', type=int, default=DEFAULT_CONFIG['num_layers'])
    parser.add_argument('--out_ckpt', type=str, default='model_dual_encoder_chembert.pt')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 70)
    print("DUAL ENCODER (GPS + ChemicalBERT) with InfoNCE")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"BERT model: {BERT_MODEL_NAME}")
    print(f"Config: epochs={args.epochs}, batch={args.batch_size}")
    print(f"LR GNN: {args.lr_gnn}, LR BERT: {args.lr_bert}")
    print(f"Temperature: {args.temperature}")
    print(f"Embedding dim: {args.emb_dim}")
    
    # Paths
    data_dir = Path(args.data_dir)
    train_graphs = str(data_dir / "train_graphs.pkl")
    val_graphs = str(data_dir / "validation_graphs.pkl")
    
    # Tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    
    # Dataset
    train_dataset = GraphTextDataset(train_graphs)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn_dual(b, tokenizer, DEFAULT_CONFIG['max_length']),
        num_workers=0,
        drop_last=True
    )
    
    # Model
    print("\nCreating model...")
    model = DualEncoder(
        hidden_dim=args.hidden_dim,
        emb_dim=args.emb_dim,
        num_layers=args.num_layers
    ).to(device)
    
    gnn_params = sum(p.numel() for p in model.graph_encoder.parameters())
    bert_params = sum(p.numel() for p in model.text_encoder.parameters())
    print(f"GNN parameters: {gnn_params:,}")
    print(f"BERT parameters: {bert_params:,}")
    print(f"Total: {gnn_params + bert_params:,}")
    
    # Optimizer avec deux LR différents
    optimizer = torch.optim.AdamW([
        {"params": model.graph_encoder.parameters(), "lr": args.lr_gnn},
        {"params": model.text_encoder.parameters(), "lr": args.lr_bert},
    ], weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training
    best_mrr = 0.0
    patience_counter = 0
    patience = DEFAULT_CONFIG['patience']
    
    for epoch in range(args.epochs):
        loss = train_epoch(model, train_loader, optimizer, device, args.temperature)
        
        val_scores = {}
        if os.path.exists(val_graphs):
            val_scores = eval_retrieval(model, val_graphs, tokenizer, device, DEFAULT_CONFIG['max_length'])
        
        print(f"Epoch {epoch+1}/{args.epochs} | loss={loss:.4f} | {val_scores}")
        scheduler.step()
        
        current_mrr = val_scores.get('MRR', 0)
        if current_mrr > best_mrr:
            best_mrr = current_mrr
            patience_counter = 0
            torch.save({
                'graph_encoder': model.graph_encoder.state_dict(),
                'text_encoder': model.text_encoder.state_dict(),
            }, args.out_ckpt)
            print(f"  → New best MRR: {best_mrr:.4f} | saved: {args.out_ckpt}")
        else:
            patience_counter += 1
            print(f"  → No improvement ({patience_counter}/{patience})")
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    print(f"\nDone. Best MRR: {best_mrr:.4f}")


if __name__ == "__main__":
    main()
