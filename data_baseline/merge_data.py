import argparse
import pickle
import os
import sys
from typing import List, Set

def load_graphs(path: str) -> List:
    """Loads a list of graph objects from a pickle file."""
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        sys.exit(1)
        
    print(f"Loading graphs from {path}...")
    with open(path, 'rb') as f:
        graphs = pickle.load(f)
        
    if not isinstance(graphs, list):
        print(f"Error: Expected a list of graphs in {path}, but got {type(graphs)}")
        sys.exit(1)
        
    print(f" -> Loaded {len(graphs)} graphs.")
    return graphs

def get_max_id(graphs: List) -> int:
    """Finds the maximum integer ID in a list of graphs."""
    max_id = -1
    for g in graphs:
        if hasattr(g, 'id'):
            try:
                # Assume IDs are strings representing integers (e.g., "102")
                val = int(g.id)
                if val > max_id:
                    max_id = val
            except ValueError:
                continue
    return max_id

def reindex_graphs(graphs: List, offset: int) -> List:
    """
    Updates graph IDs by adding an offset.
    Returns the modified list.
    """
    print(f"Re-indexing {len(graphs)} validation graphs with offset +{offset}...")
    for g in graphs:
        if hasattr(g, 'id'):
            try:
                current_id = int(g.id)
                new_id = current_id + offset
                g.id = str(new_id)
            except ValueError:
                # If ID isn't numeric, we can't mathematically shift it.
                # Fallback: Append suffix to ensure uniqueness
                g.id = f"{g.id}_{offset}"
    return graphs

def main():
    parser = argparse.ArgumentParser(description="Merge train and validation graph pickle files with re-indexing.")
    parser.add_argument("--train_path", type=str, required=True, help="Path to the training graphs .pkl file")
    parser.add_argument("--val_path", type=str, required=True, help="Path to the validation graphs .pkl file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the merged .pkl file")
    
    args = parser.parse_args()

    # 1. Load Data
    train_graphs = load_graphs(args.train_path)
    val_graphs = load_graphs(args.val_path)

    # 2. Calculate Offset from Train Data
    max_train_id = get_max_id(train_graphs)
    # The validation IDs should start after the last train ID
    # Use max_train_id + 1. If we assume IDs start at 0 and are continuous, this is effectively len(train).
    # However, using max_id is safer if there are gaps.
    offset = max_train_id + 1
    
    print(f"Max Train ID found: {max_train_id}. Offset calculated: {offset}")

    # 3. Re-index Validation Data
    # We apply the shift to ensure uniqueness while preserving internal structure
    val_graphs = reindex_graphs(val_graphs, offset)

    # 4. Merge
    merged_graphs = train_graphs + val_graphs
    
    # Verify uniqueness post-merge (Sanity Check)
    all_ids = set()
    duplicates = 0
    for g in merged_graphs:
        if hasattr(g, 'id'):
            if g.id in all_ids:
                duplicates += 1
            all_ids.add(g.id)
            
    print(f"\nMerged {len(train_graphs)} (train) + {len(val_graphs)} (val) = {len(merged_graphs)} total graphs.")
    if duplicates > 0:
        print(f"WARNING: {duplicates} duplicate IDs found after re-indexing! Check source data types.")
    else:
        print("Success: All merged IDs are unique.")

    # 5. Save
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "train_val_graphs.pkl")
    
    print(f"Saving merged dataset to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(merged_graphs, f)
        
    print("Done.")

if __name__ == "__main__":
    main()