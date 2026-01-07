#!/usr/bin/env python3
"""
retrieval_answer.py - Retrieval with Dynamic Imports
Permet de choisir le modèle et le code source via arguments.

%%bash
python data_baseline/retrieval_answer_new.py \
  --code train_gps_chembert_mse \
  --model model_gps_chembert_mse.pt \
  --data_dir data_baseline/data \
  --results_dir results
"""

import os
import sys
import argparse
import importlib
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path

from data_utils import (
    load_id2emb, load_descriptions_from_graphs, PreprocessedGraphDataset, collate_fn
)

@torch.no_grad()
def retrieve_descriptions(model, train_data, test_data, train_emb_dict, device, output_csv):
    """
    Logique de retrieval inchangée.
    """
    # Chargement des descriptions d'entraînement (pour pouvoir les récupérer)
    # Note: On suppose que load_descriptions_from_graphs sait gérer le fichier pkl
    print(f"Loading descriptions from {train_data}...")
    train_id2desc = load_descriptions_from_graphs(train_data)
    
    # Préparation des embeddings d'entraînement
    train_ids = list(train_emb_dict.keys())
    # On empile les tenseurs
    train_embs = torch.stack([train_emb_dict[id_] for id_ in train_ids]).to(device)
    # Normalisation pour la similarité cosinus (optionnel mais recommandé pour le retrieval)
    train_embs = F.normalize(train_embs, dim=-1)
    
    print(f"Train set size: {len(train_ids)}")
    
    # Préparation du test set (Graphes)
    test_ds = PreprocessedGraphDataset(test_data)
    test_dl = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
    
    print(f"Test set size: {len(test_ds)}")
    
    # Encodage des molécules de test
    test_mol_embs = []
    test_ids_ordered = []
    
    for graphs in test_dl:
        graphs = graphs.to(device)
        # Inference du modèle
        mol_emb = model(graphs)
        # Normalisation aussi pour le test
        mol_emb = F.normalize(mol_emb, dim=-1)
        
        test_mol_embs.append(mol_emb)
        
        # Récupération des IDs du batch
        batch_size = graphs.num_graphs
        start_idx = len(test_ids_ordered)
        test_ids_ordered.extend(test_ds.ids[start_idx:start_idx + batch_size])
    
    test_mol_embs = torch.cat(test_mol_embs, dim=0)
    print(f"Encoded {test_mol_embs.size(0)} test molecules")
    
    # Calcul de similarité (Produit scalaire car vecteurs normalisés = Cosine Similarity)
    similarities = test_mol_embs @ train_embs.t()
    
    # Pour chaque molécule de test, on trouve l'indice de la molécule train la plus proche
    most_similar_indices = similarities.argmax(dim=-1).cpu()
    
    results = []
    for i, test_id in enumerate(test_ids_ordered):
        train_idx = most_similar_indices[i].item()
        retrieved_train_id = train_ids[train_idx]
        
        # On récupère la description textuelle associée à cet ID d'entraînement
        retrieved_desc = train_id2desc.get(retrieved_train_id, "No description found")
        
        results.append({
            'ID': test_id,
            'description': retrieved_desc
        })
        
        # Petit affichage pour vérifier les 3 premiers
        if i < 3:
            print(f"\n[Test ID {test_id}] -> Closest Train ID: {retrieved_train_id}")
            print(f"Desc: {retrieved_desc[:100]}...")
    
    # Sauvegarde
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"\n{'='*80}")
    print(f"Saved {len(results)} retrieved descriptions to: {output_csv}")
    
    return results_df


def main():
    # 1. Gestion des arguments
    parser = argparse.ArgumentParser(description="Retrieval generation")
    parser.add_argument('--code', type=str, required=True, 
                        help="Nom du fichier python contenant le modèle (ex: train_gps_chembert_mse)")
    parser.add_argument('--model', type=str, required=True, 
                        help="Nom du fichier .pt du modèle (ex: model_gps_chembert_mse.pt)")
    parser.add_argument('--data_dir', type=str, default='data_baseline/data', 
                        help="Dossier contenant les données")
    parser.add_argument('--results_dir', type=str, default='results',
                        help="Dossier contenant le modèle sauvegardé")
    
    args = parser.parse_args()

    # Chemins
    base_path = Path(args.data_dir)
    results_path = Path(args.results_dir)
    
    # Import dynamique du module spécifié
    # On ajoute le dossier courant au path pour que python trouve le fichier
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        print(f"🔄 Import dynamique du module : {args.code} ...")
        # C'est ici que la magie opère : on importe le fichier comme si on faisait 'import X'
        source_module = importlib.import_module(args.code)
        
        # On récupère la classe et la config depuis ce module
        MolGNN = source_module.MolGNN
        
        # On essaie de récupérer le DEVICE du module, sinon on le redéfinit
        if hasattr(source_module, 'DEVICE'):
            DEVICE = source_module.DEVICE
        else:
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            
    except ImportError as e:
        print(f"❌ Erreur : Impossible d'importer le module '{args.code}'.")
        print(f"Détail : {e}")
        return

    print(f"Device: {DEVICE}")

    # Définition des chemins de données (On utilise ceux passés en arguments ou déduits)
    # Note: On garde la logique que Test Graphs est dans data_dir
    test_graphs_path = base_path / "test_graphs.pkl"
    train_graphs_path = base_path / "train_graphs.pkl"
    
    # Pour les embeddings d'entraînement, on doit savoir quel fichier utiliser.
    # Soit on le demande en argument, soit on devine le fichier par défaut généré précédemment.
    # Pour faire simple ici, on cherche le fichier ChemBERT par défaut car c'est votre pipeline actuel.
    train_emb_path = base_path / "train_embeddings_ChemBERT.csv"
    
    if not train_emb_path.exists():
        # Fallback sur le nom standard si ChemBERT n'est pas trouvé
        train_emb_path = base_path / "train_embeddings.csv"

    output_csv = base_path / "test_retrieved_descriptions.csv"
    model_file_path = results_path / args.model

    # Vérifications
    if not model_file_path.exists():
        # On tente de regarder dans data_dir si jamais il a été mis là
        if (base_path / args.model).exists():
            model_file_path = base_path / args.model
        else:
            print(f"❌ Error: Model checkpoint '{model_file_path}' not found.")
            return
    
    if not test_graphs_path.exists():
        print(f"❌ Error: Preprocessed graphs not found at {test_graphs_path}")
        return

    # Chargement des embeddings (dictionnaire ID -> Tensor)
    print(f"Loading embeddings from {train_emb_path}...")
    train_emb = load_id2emb(str(train_emb_path))
    emb_dim = len(next(iter(train_emb.values())))
    
    # Initialisation du modèle
    print(f"Initializing model (dim={emb_dim})...")
    model = MolGNN(out_dim=emb_dim).to(DEVICE)
    
    print(f"Loading weights from {model_file_path}...")
    model.load_state_dict(torch.load(model_file_path, map_location=DEVICE))
    model.eval()
    
    # Lancement du retrieval
    retrieve_descriptions(
        model=model,
        train_data=str(train_graphs_path),
        test_data=str(test_graphs_path),
        train_emb_dict=train_emb,
        device=DEVICE,
        output_csv=str(output_csv)
    )

if __name__ == "__main__":
    main()