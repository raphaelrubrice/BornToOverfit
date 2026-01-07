#!/usr/bin/env python3
"""Generate BERT embeddings for molecular descriptions (Optimized with Batching)."""

import pickle
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from pathlib import Path
import os

# Configuration
MAX_TOKEN_LENGTH = 128
BATCH_SIZE = 32  # Vous pouvez monter à 64 ou 128 selon la VRAM de votre GPU

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available(): # Pour les Mac M1/M2/M3
        return torch.device('mps')
    else:
        return torch.device('cpu')

if __name__ == "__main__":
    file_path = Path(os.path.abspath(__file__))
    parent_folder = file_path.parent
    base_path = parent_folder / "data"

    # Load BERT model
    print("Loading BERT model...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased')
    
    device = get_device()
    model = model.to(device)
    model.eval()
    print(f"Model loaded on: {device}")

    # Process each split
    for split in ['train', 'validation']:
        print(f"\nProcessing {split}...")
        
        # Load graphs from pkl file
        pkl_path = str(base_path / f'{split}_graphs.pkl')
        if not os.path.exists(pkl_path):
            print(f"File not found: {pkl_path}, skipping...")
            continue

        print(f"Loading from {pkl_path}...")
        with open(pkl_path, 'rb') as f:
            graphs = pickle.load(f)
        print(f"Loaded {len(graphs)} graphs")
        
        # Prepare storage
        all_ids = []
        all_embeddings = []
        
        # Batch processing loop
        # On avance par pas de BATCH_SIZE (ex: 0, 32, 64...)
        for i in tqdm(range(0, len(graphs), BATCH_SIZE), desc=f"Encoding {split}"):
            # 1. Créer le batch
            batch_graphs = graphs[i : i + BATCH_SIZE]
            
            # Extraire les descriptions et IDs en liste
            batch_descriptions = [g.description for g in batch_graphs]
            batch_ids = [g.id for g in batch_graphs]
            
            # 2. Tokenize (Le tokenizer gère nativement les listes de chaînes)
            # padding=True va padder selon la phrase la plus longue DU BATCH (plus efficace)
            inputs = tokenizer(batch_descriptions, 
                               return_tensors='pt', 
                               truncation=True, 
                               max_length=MAX_TOKEN_LENGTH, 
                               padding=True)
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 3. Inference
            with torch.no_grad():
                outputs = model(**inputs)
            
            # 4. Récupérer les embeddings (CLS token)
            # On récupère [Batch_Size, Hidden_Dim] d'un coup
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            # 5. Stocker les résultats
            all_ids.extend(batch_ids)
            # Convertir chaque ligne de la matrice numpy en string pour le CSV
            all_embeddings.extend([','.join(map(str, emb)) for emb in batch_embeddings])
        
        # Save to CSV
        result = pd.DataFrame({
            'ID': all_ids,
            'embedding': all_embeddings
        })
        output_path = str(base_path / f'{split}_embeddings.csv')
        result.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")

    print("\nDone!")