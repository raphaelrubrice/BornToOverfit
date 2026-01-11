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

def get_graph_ids(graphs: List) -> Set[str]:
    """Extracts a set of IDs from a list of graph objects for validation."""
    ids = set()
    for i, g in enumerate(graphs):
        # Robustly handle id attribute
        if hasattr(g, 'id'):
            g_id = str(g.id)
        else:
            print(f"Warning: Graph at index {i} is missing an 'id' attribute. Skipping ID check for this item.")
            continue
            
        if g_id in ids:
            print(f"Warning: Duplicate ID found within the single file itself: {g_id}")
        ids.add(g_id)
    return ids

def main():
    parser = argparse.ArgumentParser(description="Merge train and validation graph pickle files.")
    parser.add_argument("--train_path", type=str, required=True, help="Path to the training graphs .pkl file")
    parser.add_argument("--val_path", type=str, required=True, help="Path to the validation graphs .pkl file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the merged .pkl file")
    
    args = parser.parse_args()

    # 1. Load Data
    train_graphs = load_graphs(args.train_path)
    val_graphs = load_graphs(args.val_path)

    # 2. Check for ID Overlaps
    print("\nVerifying unique IDs...")
    train_ids = get_graph_ids(train_graphs)
    val_ids = get_graph_ids(val_graphs)
    
    # Check intersection
    overlap = train_ids.intersection(val_ids)
    
    if overlap:
        print(f"Error: Found {len(overlap)} overlapping IDs between train and validation sets.")
        print(f"Sample overlapping IDs: {list(overlap)[:5]}")
        print("Merging aborted to prevent data duplication/leakage.")
        sys.exit(1)
    else:
        print(" -> No ID overlaps found. Safe to merge.")

    # 3. Merge
    merged_graphs = train_graphs + val_graphs
    print(f"\nMerged {len(train_graphs)} (train) + {len(val_graphs)} (val) = {len(merged_graphs)} total graphs.")

    # 4. Save
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "train_val_graphs.pkl")
    
    print(f"Saving merged dataset to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(merged_graphs, f)
        
    print("Done.")

if __name__ == "__main__":
    main()