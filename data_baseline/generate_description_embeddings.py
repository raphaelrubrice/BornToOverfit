#!/usr/bin/env python3
"""Generate BERT embeddings for molecular descriptions."""

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
    parser.add_argument(
        "--base_model",
        type=str,
        default="bert-base-uncased",
        help="HF model name (e.g. bert-base-uncased, Qwen/Qwen2-0.5B-Instruct, mistralai/Mistral-7B-v0.1)"
    )
    parser.add_argument(
                    "--batch_size",
                    type=int,
                    default=16,
                    help="Batch size for embedding generation (default: 16)"
                )
    args = parser.parse_args()

    file_path = Path(os.path.abspath(__file__))
    parent_folder = file_path.parent

    base_path = parent_folder / "data"

    # Configuration
    MAX_TOKEN_LENGTH = 128

    print(f"Loading model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    model = AutoModel.from_pretrained(
        args.base_model,
        output_hidden_states=True,
        return_dict=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    print(f"Model loaded on: {device}")

    # Process each split
    for split in ['train', 'validation']:
        print(f"\nProcessing {split}...")
        
        # Load graphs from pkl file
        pkl_path = str(base_path / f'{split}_graphs.pkl')
        print(f"Loading from {pkl_path}...")
        with open(pkl_path, 'rb') as f:
            graphs = pickle.load(f)
        print(f"Loaded {len(graphs)} graphs")
        
        # Generate embeddings
        ids = []
        embeddings = []

        descriptions = [g.description for g in graphs]
        graph_ids = [g.id for g in graphs]

        for i in tqdm(range(0, len(graphs), args.batch_size),
                      total=(len(graphs) + args.batch_size - 1) // args.batch_size):
            batch_text = descriptions[i:i + args.batch_size]
            batch_ids = graph_ids[i:i + args.batch_size]

            inputs = tokenizer(
                batch_text,
                return_tensors='pt',
                truncation=True,
                max_length=MAX_TOKEN_LENGTH,
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            hidden = outputs.hidden_states[-1]              # (B, T, H)
            attention_mask = inputs["attention_mask"]       # (B, T)

            # Last non-pad token per sequence
            last_token_idx = attention_mask.sum(dim=1) - 1  # (B,)
            batch_idx = torch.arange(hidden.size(0), device=hidden.device)
            seq_embedding = hidden[batch_idx, last_token_idx, :]  # (B, H)

            ids.extend(batch_ids)
            embeddings.extend(seq_embedding.cpu().numpy())
        
        # Save to CSV
        result = pd.DataFrame({
            'ID': ids,
            'embedding': [','.join(map(str, emb.tolist())) for emb in embeddings]
        })
        output_path = str(base_path / f'{split}_embeddings.csv')
        result.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")

    print("\nDone!")

