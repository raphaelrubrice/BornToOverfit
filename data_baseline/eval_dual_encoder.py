"""
eval_dual_encoder.py - Évaluation du Dual Encoder pour le retrieval final

═══════════════════════════════════════════════════════════════════════════════
CE QUI CHANGE PAR RAPPORT À L'ÉVALUATION PRÉCÉDENTE:
═══════════════════════════════════════════════════════════════════════════════

AVANT (embeddings figés):
- On chargeait train_embeddings.csv (embeddings BERT pré-calculés)
- Ces embeddings ne correspondaient pas bien à l'espace appris par le GNN

MAINTENANT (Dual Encoder):
- On encode les textes du train AVEC LE MÊME BERT entraîné
- Les embeddings texte sont dans le même espace que les embeddings graphe
- Meilleure correspondance pour le retrieval

═══════════════════════════════════════════════════════════════════════════════
"""

import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

import evaluate
from transformers import AutoTokenizer

# Importer les classes du training
from train_dual_encoder import (
    DualEncoder, GraphTextDataset, 
    BERT_MODEL_NAME, MAX_LENGTH, EMB_DIM, DEVICE
)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
file_path = Path(os.path.abspath(__file__))
parent_folder = file_path.parent
base_path = parent_folder / "data"

TRAIN_GRAPHS = str(base_path / "train_graphs.pkl")
VAL_GRAPHS = str(base_path / "validation_graphs.pkl")
MODEL_PATH = str(base_path / "model_dual_encoder.pt")


# ═══════════════════════════════════════════════════════════════════════════════
# FONCTIONS D'ENCODAGE
# ═══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def encode_all_texts(model, texts, tokenizer, device, batch_size=64):
    """
    Encode tous les textes avec le TextEncoder entraîné.
    
    C'est LA différence clé : on utilise le BERT qui a été entraîné
    avec le GNN, pas un BERT séparé.
    """
    model.eval()
    all_embs = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize
        encoded = tokenizer(
            batch_texts, padding=True, truncation=True,
            max_length=MAX_LENGTH, return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        
        # Encode avec le TextEncoder entraîné
        text_emb = model.text_encoder(input_ids, attention_mask)
        all_embs.append(text_emb.cpu())
    
    return torch.cat(all_embs, dim=0)


@torch.no_grad()
def encode_all_graphs(model, graphs, device, batch_size=64):
    """
    Encode tous les graphes avec le GraphEncoder entraîné.
    """
    model.eval()
    all_embs = []
    
    for i in range(0, len(graphs), batch_size):
        batch_graphs = graphs[i:i + batch_size]
        batch = Batch.from_data_list(batch_graphs).to(device)
        
        graph_emb = model.graph_encoder(batch)
        all_embs.append(graph_emb.cpu())
    
    return torch.cat(all_embs, dim=0)


# ═══════════════════════════════════════════════════════════════════════════════
# RETRIEVAL
# ═══════════════════════════════════════════════════════════════════════════════
def predict_val_retrieval(model, tokenizer, device):
    """
    Pour chaque graphe du validation set, trouve le texte le plus proche
    parmi tous les textes du training set.
    
    Retourne les descriptions prédites et les références.
    """
    print("Loading training data...")
    with open(TRAIN_GRAPHS, 'rb') as f:
        train_graphs_data = pickle.load(f)
    
    train_ids = [g.id for g in train_graphs_data]
    train_descriptions = [g.description for g in train_graphs_data]
    train_id2desc = {g.id: g.description for g in train_graphs_data}
    
    print("Loading validation data...")
    with open(VAL_GRAPHS, 'rb') as f:
        val_graphs_data = pickle.load(f)
    
    val_ids = [g.id for g in val_graphs_data]
    val_ref = {g.id: g.description for g in val_graphs_data}
    
    # ═══════════════════════════════════════════════════════════════════════
    # ÉTAPE CLÉ : Encoder les textes du train avec le BERT entraîné
    # ═══════════════════════════════════════════════════════════════════════
    print(f"Encoding {len(train_descriptions)} training texts with trained BERT...")
    train_text_embs = encode_all_texts(model, train_descriptions, tokenizer, device)
    train_text_embs = train_text_embs.to(device)
    print(f"Train text embeddings shape: {train_text_embs.shape}")
    
    # Encoder les graphes du validation set
    print(f"Encoding {len(val_graphs_data)} validation graphs...")
    val_graph_embs = encode_all_graphs(model, val_graphs_data, device)
    val_graph_embs = val_graph_embs.to(device)
    print(f"Val graph embeddings shape: {val_graph_embs.shape}")
    
    # Calculer les similarités
    print("Computing similarities...")
    sims = val_graph_embs @ train_text_embs.T  # [n_val, n_train]
    
    # Pour chaque graphe val, trouver le texte train le plus proche
    best_indices = sims.argmax(dim=-1).cpu().tolist()
    
    # Construire les prédictions et références
    preds, refs = [], []
    for i, val_id in enumerate(val_ids):
        # Texte train le plus proche
        train_idx = best_indices[i]
        pred_desc = train_descriptions[train_idx]
        
        # Texte de référence (vérité terrain du validation)
        ref_desc = val_ref[val_id]
        
        preds.append(pred_desc)
        refs.append(ref_desc)
    
    return preds, refs


# ═══════════════════════════════════════════════════════════════════════════════
# MÉTRIQUES
# ═══════════════════════════════════════════════════════════════════════════════
def compute_metrics(preds, refs):
    """Calcule BLEU-4 et BERTScore"""
    print("Computing BLEU-4...")
    bleu = evaluate.load("bleu").compute(
        predictions=preds,
        references=[[r] for r in refs],
        max_order=4
    )["bleu"]
    
    print("Computing BERTScore...")
    bert = evaluate.load("bertscore").compute(
        predictions=preds,
        references=refs,
        model_type="roberta-base",
        lang="en"
    )
    bert_f1 = float(np.mean(bert["f1"]))
    
    # Normaliser BLEU si nécessaire
    bleu_norm = bleu / 100.0 if bleu > 1.0 else bleu
    final_proxy = 0.5 * bleu_norm + 0.5 * bert_f1
    
    return bleu, bert_f1, final_proxy


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("DUAL ENCODER EVALUATION")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print()
    
    # Vérifier que le modèle existe
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please train the model first with train_dual_encoder.py")
        return
    
    # Charger le tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    
    # Créer et charger le modèle
    print("Loading model...")
    model = DualEncoder().to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.graph_encoder.load_state_dict(checkpoint['graph_encoder'])
    model.text_encoder.load_state_dict(checkpoint['text_encoder'])
    model.eval()
    print("Model loaded successfully!")
    print()
    
    # Faire les prédictions
    preds, refs = predict_val_retrieval(model, tokenizer, DEVICE)
    
    # Calculer les métriques
    print()
    bleu4, bert_f1, final_proxy = compute_metrics(preds, refs)
    
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"BLEU-4:        {bleu4:.4f}")
    print(f"BERTScore F1:  {bert_f1:.4f}")
    print(f"Final proxy:   {final_proxy:.4f}")
    print()
    print("Example:")
    print(f"PRED: {preds[0][:200]}...")
    print(f"REF:  {refs[0][:200]}...")


if __name__ == "__main__":
    main()
