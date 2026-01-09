"""
train_dual_encoder.py - Dual Encoder (GNN + BERT entraînés ensemble)

═══════════════════════════════════════════════════════════════════════════════
DIFFÉRENCES MAJEURES AVEC LES VERSIONS PRÉCÉDENTES:
═══════════════════════════════════════════════════════════════════════════════

1. PLUS DE FICHIERS CSV D'EMBEDDINGS
   - Avant : on chargeait train_embeddings.csv (embeddings BERT pré-calculés)
   - Maintenant : on calcule les embeddings BERT à chaque batch

2. BERT EST ENTRAÎNABLE
   - Avant : BERT était utilisé une fois pour générer les CSV, puis jeté
   - Maintenant : BERT fait partie du modèle et ses poids sont mis à jour

3. NOUVEAU DATASET
   - Avant : PreprocessedGraphDataset retournait (graph, embedding_tensor)
   - Maintenant : On crée un dataset qui retourne (graph, texte_brut)

4. TOKENIZATION À LA VOLÉE
   - À chaque batch, on tokenize les textes avec le tokenizer BERT
   - Cela permet à BERT de s'adapter pendant l'entraînement

═══════════════════════════════════════════════════════════════════════════════
"""

import os
from pathlib import Path
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch_geometric.data import Batch
from torch_geometric.nn import GPSConv, GINEConv, global_add_pool

# Transformers pour BERT
from transformers import AutoTokenizer, AutoModel

from data_utils import x_map, e_map


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
file_path = Path(os.path.abspath(__file__))
parent_folder = file_path.parent
base_path = parent_folder / "data"

TRAIN_GRAPHS = str(base_path / "train_graphs.pkl")
VAL_GRAPHS   = str(base_path / "validation_graphs.pkl")
TEST_GRAPHS  = str(base_path / "test_graphs.pkl")

# On n'utilise PLUS les CSV d'embeddings !
# TRAIN_EMB_CSV = str(base_path / "train_embeddings.csv")  # SUPPRIMÉ

# Hyperparamètres
BATCH_SIZE = 32        # Plus petit car BERT consomme beaucoup de mémoire
EPOCHS = 30
LR_GNN = 5e-4          # Learning rate pour le GNN
LR_BERT = 2e-5         # Learning rate plus faible pour BERT (fine-tuning)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Architecture GNN
HIDDEN_DIM = 256
NUM_LAYERS = 4
NUM_HEADS = 4
DROPOUT = 0.1

# Embedding dimension (doit matcher BERT)
EMB_DIM = 256          # Dimension de l'espace commun (projection depuis 768)

# InfoNCE
TEMPERATURE = 0.07

# BERT model (tu peux changer pour SciBERT)
BERT_MODEL_NAME = "allenai/scibert_scivocab_uncased"  # Meilleur pour la chimie
# BERT_MODEL_NAME = "bert-base-uncased"  # Alternative généraliste

MAX_LENGTH = 128       # Longueur max des textes tokenizés


# ═══════════════════════════════════════════════════════════════════════════════
# Feature dimensions (identique aux versions précédentes)
# ═══════════════════════════════════════════════════════════════════════════════
ATOM_FEATURE_DIMS = [
    len(x_map['atomic_num']), len(x_map['chirality']), len(x_map['degree']),
    len(x_map['formal_charge']), len(x_map['num_hs']), len(x_map['num_radical_electrons']),
    len(x_map['hybridization']), len(x_map['is_aromatic']), len(x_map['is_in_ring']),
]
BOND_FEATURE_DIMS = [
    len(e_map['bond_type']), len(e_map['stereo']), len(e_map['is_conjugated']),
]


# ═══════════════════════════════════════════════════════════════════════════════
# NOUVEAU DATASET : retourne (graphe, texte_brut) au lieu de (graphe, embedding)
# ═══════════════════════════════════════════════════════════════════════════════
class GraphTextDataset(Dataset):
    """
    Dataset qui charge les graphes avec leurs descriptions textuelles BRUTES.
    
    DIFFÉRENCE avec PreprocessedGraphDataset:
    - Avant : retournait (graph, embedding_tensor) où embedding venait du CSV
    - Maintenant : retourne (graph, description_string) 
    
    Le texte sera tokenizé dans la fonction collate_fn, pas ici.
    """
    def __init__(self, graph_path: str):
        print(f"Loading graphs from: {graph_path}")
        with open(graph_path, 'rb') as f:
            self.graphs = pickle.load(f)
        
        # Extraire les descriptions (elles sont stockées dans chaque graph.description)
        self.descriptions = [g.description for g in self.graphs]
        self.ids = [g.id for g in self.graphs]
        
        print(f"Loaded {len(self.graphs)} graphs with descriptions")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        # Retourne le graphe ET le texte brut (string)
        return self.graphs[idx], self.descriptions[idx]


def collate_fn_dual(batch, tokenizer):
    """
    Fonction de collation pour le Dual Encoder.
    
    Prend une liste de (graph, text_string) et retourne:
    - batch_graphs: graphes batchés avec PyG
    - input_ids: tokens des textes [B, seq_len]
    - attention_mask: masque d'attention [B, seq_len]
    
    C'est ici qu'on tokenize les textes, pas dans le dataset.
    Pourquoi ? Parce que le tokenizer a besoin de tous les textes du batch
    pour faire du padding efficace.
    """
    graphs, texts = zip(*batch)
    
    # Batch les graphes avec PyTorch Geometric
    batch_graphs = Batch.from_data_list(list(graphs))
    
    # Tokenize tous les textes du batch
    # padding=True : ajoute des [PAD] pour que tous aient la même longueur
    # truncation=True : coupe si trop long
    # return_tensors="pt" : retourne des tenseurs PyTorch
    encoded = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    
    return batch_graphs, encoded["input_ids"], encoded["attention_mask"]


# ═══════════════════════════════════════════════════════════════════════════════
# ENCODEURS
# ═══════════════════════════════════════════════════════════════════════════════

class AtomEncoder(nn.Module):
    """Encode les 9 features atomiques (identique aux versions précédentes)"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, hidden_dim) for dim in ATOM_FEATURE_DIMS
        ])
    def forward(self, x):
        return sum(emb(x[:, i]) for i, emb in enumerate(self.embeddings))


class BondEncoder(nn.Module):
    """Encode les 3 features de liaison (identique aux versions précédentes)"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, hidden_dim) for dim in BOND_FEATURE_DIMS
        ])
    def forward(self, edge_attr):
        return sum(emb(edge_attr[:, i]) for i, emb in enumerate(self.embeddings))


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH ENCODER (GPS) - Similaire aux versions précédentes
# ═══════════════════════════════════════════════════════════════════════════════
class GraphEncoder(nn.Module):
    """
    Encode un graphe moléculaire en vecteur.
    Architecture GPS (identique à v3).
    
    Sortie: vecteur de dimension EMB_DIM (pas 768 comme avant!)
    """
    def __init__(self, hidden_dim=HIDDEN_DIM, out_dim=EMB_DIM, 
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
        
        # Projection vers l'espace commun
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
        return F.normalize(g, dim=-1)  # Normalisation L2


# ═══════════════════════════════════════════════════════════════════════════════
# TEXT ENCODER (BERT) - NOUVEAU !
# ═══════════════════════════════════════════════════════════════════════════════
class TextEncoder(nn.Module):
    """
    Encode un texte en vecteur via BERT.
    
    DIFFÉRENCE MAJEURE avec avant:
    - Avant : BERT était utilisé offline pour créer des CSV, puis jeté
    - Maintenant : BERT fait partie du modèle et s'entraîne !
    
    Architecture:
    1. BERT transforme les tokens en embeddings contextuels
    2. Mean pooling sur tous les tokens (meilleur que juste [CLS])
    3. Projection linéaire vers l'espace commun (EMB_DIM)
    """
    def __init__(self, bert_model_name=BERT_MODEL_NAME, out_dim=EMB_DIM, dropout=DROPOUT):
        super().__init__()
        
        # Charger BERT pré-entraîné
        self.bert = AutoModel.from_pretrained(bert_model_name)
        
        # Dimension de sortie de BERT (768 pour base, 1024 pour large)
        bert_dim = self.bert.config.hidden_size
        
        # Projection vers l'espace commun
        # On passe de 768 (BERT) à EMB_DIM (256)
        self.proj = nn.Sequential(
            nn.Linear(bert_dim, bert_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bert_dim, out_dim),
        )
    
    def mean_pooling(self, token_embeddings, attention_mask):
        """
        Mean pooling : moyenne des embeddings de tokens (en ignorant le padding).
        
        Pourquoi pas juste le token [CLS] ?
        - [CLS] est optimisé pour la classification, pas pour la similarité
        - Mean pooling capture mieux le sens global de la phrase
        - C'est ce que font SBERT et autres modèles de similarité
        
        Args:
            token_embeddings: [B, seq_len, 768] - embeddings de chaque token
            attention_mask: [B, seq_len] - 1 pour vrais tokens, 0 pour padding
        
        Returns:
            [B, 768] - un embedding par texte
        """
        # Étendre le mask pour matcher les dimensions des embeddings
        mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, seq_len, 1]
        
        # Somme pondérée (les tokens padding ont mask=0 donc ne comptent pas)
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)  # [B, 768]
        
        # Nombre de vrais tokens par exemple
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)  # [B, 1], évite division par 0
        
        # Moyenne
        return sum_embeddings / sum_mask
    
    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: [B, seq_len] - IDs des tokens
            attention_mask: [B, seq_len] - masque d'attention
        
        Returns:
            [B, EMB_DIM] - embeddings normalisés L2
        """
        # Passer dans BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # outputs.last_hidden_state : [B, seq_len, 768]
        token_embeddings = outputs.last_hidden_state
        
        # Mean pooling
        sentence_embedding = self.mean_pooling(token_embeddings, attention_mask)
        
        # Projection vers espace commun
        projected = self.proj(sentence_embedding)
        
        # Normalisation L2
        return F.normalize(projected, dim=-1)


# ═══════════════════════════════════════════════════════════════════════════════
# DUAL ENCODER : combine GraphEncoder + TextEncoder
# ═══════════════════════════════════════════════════════════════════════════════
class DualEncoder(nn.Module):
    """
    Modèle complet qui encode graphes ET textes dans le même espace.
    
    Les deux encodeurs sont entraînés ENSEMBLE avec InfoNCE.
    C'est la différence fondamentale avec les versions précédentes.
    """
    def __init__(self):
        super().__init__()
        self.graph_encoder = GraphEncoder(out_dim=EMB_DIM)
        self.text_encoder = TextEncoder(out_dim=EMB_DIM)
    
    def encode_graphs(self, batch_graphs):
        """Encode une batch de graphes"""
        return self.graph_encoder(batch_graphs)
    
    def encode_texts(self, input_ids, attention_mask):
        """Encode une batch de textes"""
        return self.text_encoder(input_ids, attention_mask)
    
    def forward(self, batch_graphs, input_ids, attention_mask):
        """
        Encode graphes et textes, retourne les deux embeddings.
        """
        graph_emb = self.encode_graphs(batch_graphs)
        text_emb = self.encode_texts(input_ids, attention_mask)
        return graph_emb, text_emb


# ═══════════════════════════════════════════════════════════════════════════════
# INFONCE LOSS (identique à avant)
# ═══════════════════════════════════════════════════════════════════════════════
def info_nce_loss(graph_emb, text_emb, temperature=TEMPERATURE):
    """
    Loss contrastive symétrique.
    Identique à la version précédente.
    """
    logits = graph_emb @ text_emb.T / temperature
    labels = torch.arange(len(logits), device=logits.device)
    loss_g2t = F.cross_entropy(logits, labels)
    loss_t2g = F.cross_entropy(logits.T, labels)
    return (loss_g2t + loss_t2g) / 2


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════
def train_epoch(model, loader, optimizer, device):
    """
    Une epoch d'entraînement.
    
    DIFFÉRENCE avec avant:
    - On reçoit (graphs, input_ids, attention_mask) au lieu de (graphs, embeddings)
    - On encode les textes À LA VOLÉE avec BERT
    """
    model.train()
    total_loss, total = 0.0, 0
    
    for batch_graphs, input_ids, attention_mask in loader:
        # Tout sur GPU
        batch_graphs = batch_graphs.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # Forward : encode graphes ET textes
        graph_emb, text_emb = model(batch_graphs, input_ids, attention_mask)
        
        # Loss InfoNCE
        loss = info_nce_loss(graph_emb, text_emb)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * batch_graphs.num_graphs
        total += batch_graphs.num_graphs
    
    return total_loss / total


@torch.no_grad()
def eval_retrieval(model, data_path, tokenizer, device):
    """
    Évaluation du retrieval.
    
    DIFFÉRENCE avec avant:
    - On doit aussi encoder les textes du validation set à la volée
    """
    model.eval()
    
    # Charger le dataset
    dataset = GraphTextDataset(data_path)
    
    # On ne peut pas utiliser le DataLoader directement car on a besoin du tokenizer
    # Donc on fait par petits batchs manuellement
    all_graph_emb = []
    all_text_emb = []
    
    batch_size = 64
    for i in range(0, len(dataset), batch_size):
        # Extraire un batch
        batch_data = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
        graphs, texts = zip(*batch_data)
        
        # Préparer les graphes
        batch_graphs = Batch.from_data_list(list(graphs)).to(device)
        
        # Tokenizer les textes
        encoded = tokenizer(
            list(texts), padding=True, truncation=True, 
            max_length=MAX_LENGTH, return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        
        # Encoder
        graph_emb, text_emb = model(batch_graphs, input_ids, attention_mask)
        all_graph_emb.append(graph_emb)
        all_text_emb.append(text_emb)
    
    # Concaténer tous les embeddings
    all_graph_emb = torch.cat(all_graph_emb, dim=0)
    all_text_emb = torch.cat(all_text_emb, dim=0)
    
    # Calculer MRR et Recall@k
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


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("DUAL ENCODER TRAINING (GNN + BERT ensemble)")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"BERT model: {BERT_MODEL_NAME}")
    print(f"Embedding dimension: {EMB_DIM}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"LR GNN: {LR_GNN}, LR BERT: {LR_BERT}")
    print()

    # Vérifier que les données existent
    if not os.path.exists(TRAIN_GRAPHS):
        print(f"Error: {TRAIN_GRAPHS} not found")
        return
    
    # Charger le tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    
    # Créer le dataset
    train_dataset = GraphTextDataset(TRAIN_GRAPHS)
    
    # DataLoader avec notre collate_fn personnalisée
    # On utilise une lambda pour passer le tokenizer
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: collate_fn_dual(batch, tokenizer),
        num_workers=0,
        drop_last=True  # Important pour InfoNCE
    )
    
    # Créer le modèle
    print("Creating Dual Encoder model...")
    model = DualEncoder().to(DEVICE)
    
    # Compter les paramètres
    gnn_params = sum(p.numel() for p in model.graph_encoder.parameters())
    bert_params = sum(p.numel() for p in model.text_encoder.parameters())
    print(f"GNN parameters: {gnn_params:,}")
    print(f"BERT parameters: {bert_params:,}")
    print(f"Total parameters: {gnn_params + bert_params:,}")
    print()
    
    # Optimizer avec learning rates différents pour GNN et BERT
    # BERT a un LR plus faible car il est déjà pré-entraîné
    optimizer = torch.optim.AdamW([
        {"params": model.graph_encoder.parameters(), "lr": LR_GNN},
        {"params": model.text_encoder.parameters(), "lr": LR_BERT},
    ], weight_decay=1e-4)
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Training loop avec early stopping
    best_mrr = 0.0
    patience = 5
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        # Train
        loss = train_epoch(model, train_loader, optimizer, DEVICE)
        
        # Eval
        val_scores = {}
        if os.path.exists(VAL_GRAPHS):
            val_scores = eval_retrieval(model, VAL_GRAPHS, tokenizer, DEVICE)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | loss={loss:.4f} | {val_scores}")
        scheduler.step()
        
        # Early stopping
        current_mrr = val_scores.get('MRR', 0)
        if current_mrr > best_mrr:
            best_mrr = current_mrr
            patience_counter = 0
            
            # Sauvegarder le modèle complet
            save_path = str(base_path / "model_dual_encoder.pt")
            torch.save({
                'graph_encoder': model.graph_encoder.state_dict(),
                'text_encoder': model.text_encoder.state_dict(),
            }, save_path)
            print(f"  → New best MRR: {best_mrr:.4f}, saved to {save_path}")
        else:
            patience_counter += 1
            print(f"  → No improvement ({patience_counter}/{patience})")
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    print(f"\nDone. Best MRR: {best_mrr:.4f}")


if __name__ == "__main__":
    main()
