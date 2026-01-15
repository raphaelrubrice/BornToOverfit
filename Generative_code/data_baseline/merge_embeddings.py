import argparse
import os
import sys
import torch
import pandas as pd
from typing import Dict

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
                emb_vals = [float(x) for x in str(emb_str).split(',')]
                id2emb[doc_id] = torch.tensor(emb_vals, dtype=torch.float32)
            except ValueError:
                continue

    print(f" -> Loaded {len(id2emb)} embeddings.")
    return id2emb

def get_max_id_from_dict(emb_dict: Dict[str, torch.Tensor]) -> int:
    """Finds the maximum integer ID in the keys of an embedding dictionary."""
    max_id = -1
    for k in emb_dict.keys():
        try:
            val = int(k)
            if val > max_id:
                max_id = val
        except ValueError:
            continue
    return max_id

def reindex_embeddings(emb_dict: Dict[str, torch.Tensor], offset: int) -> Dict[str, torch.Tensor]:
    """
    Creates a new dictionary where keys are shifted by the offset.
    Matches the logic used in the graph merge script.
    """
    print(f"Re-indexing {len(emb_dict)} validation embeddings with offset +{offset}...")
    new_dict = {}
    for k, v in emb_dict.items():
        try:
            current_id = int(k)
            new_id = current_id + offset
            new_dict[str(new_id)] = v
        except ValueError:
            # Fallback for non-numeric IDs matches graph script fallback
            new_key = f"{k}_{offset}"
            new_dict[new_key] = v
            
    return new_dict

def save_embeddings(merged_dict: Dict[str, torch.Tensor], output_path: str, format_type: str):
    """
    Saves the merged dictionary to disk in the requested format.
    """
    ids = list(merged_dict.keys())
    # Sort numerically if possible to keep things tidy
    try:
        ids.sort(key=lambda x: int(x))
    except ValueError:
        ids.sort()
    
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
        emb_strings = []
        for emb in embs_list:
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
    parser = argparse.ArgumentParser(description="Merge train and validation embedding files with re-indexing.")
    parser.add_argument("--train_path", type=str, required=True, help="Path to training embeddings")
    parser.add_argument("--val_path", type=str, required=True, help="Path to validation embeddings")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output file")
    parser.add_argument("--output_format", type=str, choices=['pt', 'csv'], default='pt', 
                        help="Output format (default: pt)")
    
    args = parser.parse_args()

    # 1. Load Data
    train_embs = load_embeddings(args.train_path)
    val_embs = load_embeddings(args.val_path)

    # 2. Calculate Offset (Must match graph script logic)
    max_train_id = get_max_id_from_dict(train_embs)
    offset = max_train_id + 1
    
    print(f"Max Train ID found: {max_train_id}. Offset calculated: {offset}")

    # 3. Re-index Validation Embeddings
    val_embs_reindexed = reindex_embeddings(val_embs, offset)

    # 4. Merge
    # We update a copy of train_embs with the new val items
    merged_embs = train_embs.copy()
    
    # Check for collisions before blind update (sanity check)
    overlaps = set(merged_embs.keys()).intersection(set(val_embs_reindexed.keys()))
    if overlaps:
        print(f"CRITICAL ERROR: {len(overlaps)} ID collisions detected even after re-indexing.")
        sys.exit(1)
        
    merged_embs.update(val_embs_reindexed)
    print(f"Merged {len(train_embs)} + {len(val_embs)} = {len(merged_embs)} total embeddings.")

    # 5. Save
    os.makedirs(args.output_dir, exist_ok=True)
    filename = f"train_val_embeddings.{args.output_format}"
    output_path = os.path.join(args.output_dir, filename)
    
    save_embeddings(merged_embs, output_path, args.output_format)
    print("Done.")

if __name__ == "__main__":
    main()