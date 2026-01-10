#!/usr/bin/env python3
"""Generate BERT embeddings for molecular descriptions (Optimized)."""

import argparse
import pickle
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from pathlib import Path
import os
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--pooling', type=str, default='cls', choices=['cls', 'mean'])
    parser.add_argument('--splits', nargs='+', default=['train', 'validation'])
    parser.add_argument('--batch_size', type=int, default=32)
    return parser.parse_args()

def main():
    args = parse_args()
    base_path = Path(args.data_dir)
    MAX_TOKEN_LENGTH = 128
    BATCH_SIZE = args.batch_size
    
    print(f"Loading model: {args.model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModel.from_pretrained(args.model_name)
    except OSError:
        print(f"Error: Model '{args.model_name}' not found.")
        sys.exit(1)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    print(f"Model loaded on: {device}")

    for split in args.splits:
        print(f"\nProcessing {split}...")
        pkl_path = base_path / f'{split}_graphs.pkl'
        
        if not pkl_path.exists():
            print(f"Skipping {split}, file not found.")
            continue
            
        with open(pkl_path, 'rb') as f:
            graphs = pickle.load(f)
        print(f"Loaded {len(graphs)} graphs")
        
        all_ids = []
        all_embeddings = []
        
        # Batch Loop (Beaucoup plus rapide)
        for i in tqdm(range(0, len(graphs), BATCH_SIZE), desc=f"Encoding {split}"):
            batch_graphs = graphs[i : i + BATCH_SIZE]
            batch_descriptions = [g.description for g in batch_graphs]
            batch_ids = [g.id for g in batch_graphs]
            
            inputs = tokenizer(batch_descriptions, return_tensors='pt', 
                             truncation=True, max_length=MAX_TOKEN_LENGTH, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            if args.pooling == 'mean':
                # Mean Pooling vectorisé
                mask = inputs['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                sum_embeddings = torch.sum(outputs.last_hidden_state * mask, 1)
                sum_mask = torch.clamp(mask.sum(1), min=1e-9)
                batch_embeddings = sum_embeddings / sum_mask
            else:
                # CLS Pooling
                batch_embeddings = outputs.last_hidden_state[:, 0, :]
            
            all_ids.extend(batch_ids)
            all_embeddings.append(batch_embeddings.cpu())
            
        # Concat & Save as .pt (Binary)
        if all_embeddings:
            final_embeddings = torch.cat(all_embeddings, dim=0)
            
            # Nom de fichier propre (ex: train_embeddings_ChemBERT.pt)
            clean_name = args.model_name.split('/')[-1].replace('bert-base-uncased', 'bert')
            if 'chemical' in args.model_name: clean_name = 'ChemBERT'
            
            output_path = base_path / f'{split}_embeddings_{clean_name}.pt'
            
            torch.save({
                'ids': all_ids,
                'embeddings': final_embeddings
            }, output_path)
            
            print(f"Saved binary to {output_path}")
            print(f"Shape: {final_embeddings.shape}")

    print("\nDone!")

if __name__ == "__main__":
    main()