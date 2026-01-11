import argparse
import os
import sys
import torch
import pandas as pd
from typing import Dict, List, Union

def load_embeddings(path: str) -> Dict[str, torch.Tensor]:
    """
    Loads embeddings from a CSV or .pt file into a dictionary {id: tensor}.
    """
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        sys.exit(1)

    print(f"Loading embeddings from {path}...")
    _, ext = os.path.splitext(path)
    
    id2emb = {}

    if ext == '.pt':
        # Load binary PyTorch file
        # Expects dict: {'ids': List[str], 'embeddings': torch.Tensor}
        data = torch.load(path, map_location='cpu')
        
        if 'ids' not in data or 'embeddings' not in data:
            print(f"Error: .pt file {path} must contain keys 'ids' and 'embeddings'")
            sys.exit(1)
            
        ids = data['ids']
        embs = data['embeddings']
        
        if len(ids) != embs.size(0):
            print(f"Error: Mismatch in {path} - {len(ids)} IDs vs {embs.size(0)} embeddings")
            sys.exit(1)
            
        for i, doc_id in enumerate(ids):
            id2emb[str(doc_id)] = embs[i]

    else:
        # Fallback to CSV
        # Expects columns: ID, embedding (comma-separated string)
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"Error reading CSV {path}: {e}")
            sys.exit(1)
            
        required_cols = {'ID', 'embedding'}
        if not required_cols.issubset(df.columns):
            print(f"Error: CSV {path} missing required columns: {required_cols - set(df.columns)}")
            sys.exit(1)

        for _, row in df.iterrows():
            doc_id = str(row['ID'])
            emb_str = row['embedding']
            try:
                # Convert "0.1, 0.2, ..." string to tensor
                emb_vals = [float(x) for x in str(emb_str).split(',')]
                id2emb[doc_id] = torch.tensor(emb_vals, dtype=torch.float32)
            except ValueError:
                print(f"Warning: Could not parse embedding for ID {doc_id}. Skipping.")
                continue

    print(f" -> Loaded {len(id2emb)} embeddings.")
    return id2emb

def save_embeddings(merged_dict: Dict[str, torch.Tensor], output_path: str, format_type: str):
    """
    Saves the merged dictionary to disk in the requested format.
    """
    ids = list(merged_dict.keys())
    # Ensure deterministic order
    ids.sort()
    
    # Stack tensors
    embs_list = [merged_dict[i] for i in ids]
    if not embs_list:
        print("Error: No embeddings to save.")
        sys.exit(1)
        
    embs_tensor = torch.stack(embs_list)

    print(f"Saving {len(ids)} embeddings to {output_path}...")

    if format_type == 'pt':
        save_data = {
            'ids': ids,
            'embeddings': embs_tensor
        }
        torch.save(save_data, output_path)
        
    elif format_type == 'csv':
        # Convert tensors back to comma-separated strings
        emb_strings = []
        for emb in embs_list:
            # .tolist() converts tensor to list of floats
            s = ",".join(map(str, emb.tolist()))
            emb_strings.append(s)
            
        df = pd.DataFrame({
            'ID': ids,
            'embedding': emb_strings
        })
        df.to_csv(output_path, index=False)
    else:
        print(f"Error: Unknown format type '{format_type}'")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Merge train and validation embedding files (CSV or PT).")
    parser.add_argument("--train_path", type=str, required=True, help="Path to training embeddings")
    parser.add_argument("--val_path", type=str, required=True, help="Path to validation embeddings")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output file")
    parser.add_argument("--output_format", type=str, choices=['pt', 'csv'], default='pt', 
                        help="Output format (default: pt)")
    
    args = parser.parse_args()

    # 1. Load Data
    train_embs = load_embeddings(args.train_path)
    val_embs = load_embeddings(args.val_path)

    # 2. Check for Overlaps
    train_ids = set(train_embs.keys())
    val_ids = set(val_embs.keys())
    
    overlap = train_ids.intersection(val_ids)
    if overlap:
        print(f"\nError: Found {len(overlap)} overlapping IDs between files.")
        print(f"Sample overlapping IDs: {list(overlap)[:5]}")
        print("Merging aborted to prevent data leakage.")
        sys.exit(1)
    else:
        print("\n -> No ID overlaps found. Safe to merge.")

    # 3. Merge
    # Dictionary unpacking merges them (since we proved disjoint keys, order doesn't impact overwrites)
    merged_embs = {**train_embs, **val_embs}
    print(f"Merged {len(train_embs)} + {len(val_embs)} = {len(merged_embs)} total embeddings.")

    # 4. Save
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine filename based on format
    filename = f"train_val_embeddings.{args.output_format}"
    output_path = os.path.join(args.output_dir, filename)
    
    save_embeddings(merged_embs, output_path, args.output_format)
    print("Done.")

if __name__ == "__main__":
    main()