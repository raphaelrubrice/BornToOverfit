#!/usr/bin/env python3
"""Generate embeddings for molecular descriptions using a fine-tuned model."""

import pickle
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from pathlib import Path
import os
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-f", default="data_baseline/data", type=str)

    args = parser.parse_args()
    folder = args.f
    
    file_path = Path(os.path.abspath(__file__))
    parent_folder = file_path.parent.parent

    data_path = parent_folder / "data_baseline" / "data"
    save_path = parent_folder / folder

    # Configuration
    MAX_TOKEN_LENGTH = 128

    # Load BERT model
    print("Loading BERT model...")
    tokenizer = AutoTokenizer.from_pretrained('zjunlp/llama-molinst-molecule-7b')
    model = AutoModel.from_pretrained('zjunlp/llama-molinst-molecule-7b')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    print(f"Model loaded on: {device}")

    # Process each split
    for split in ['train', 'validation']:
        print(f"\nProcessing {split}...")
        
        # Load graphs from pkl file
        pkl_path = str(data_path / f'{split}_graphs.pkl')
        print(f"Loading from {pkl_path}...")
        with open(pkl_path, 'rb') as f:
            graphs = pickle.load(f)
        print(f"Loaded {len(graphs)} graphs")
        
        # Generate embeddings
        ids = []
        embeddings = []
        
        for graph in tqdm(graphs, total=len(graphs)):
            # Get description from graph
            description = graph.description
            
            # Tokenize
            inputs = tokenizer(description, return_tensors='pt', 
                            truncation=True, max_length=MAX_TOKEN_LENGTH, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get embedding
            with torch.no_grad():
                outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
            
            ids.append(graph.id)
            embeddings.append(embedding)
        
        # Save to CSV
        result = pd.DataFrame({
            'ID': ids,
            'embedding': [','.join(map(str, emb)) for emb in embeddings]
        })
        output_path = str(save_path / f'{split}_embeddings.csv')
        result.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")

    print("\nDone!")

