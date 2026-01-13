# data_utils.py (version captioning)
from typing import Dict, List, Any, Optional
import pickle
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from transformers import AutoTokenizer

# ===== feature maps (inchangés) =====
x_map: Dict[str, List[Any]] = { ... }  # comme toi
e_map: Dict[str, List[Any]] = { ... }

X_KEYS = ["atomic_num","chirality","degree","formal_charge","num_hs",
          "num_radical_electrons","hybridization","is_aromatic","is_in_ring"]
E_KEYS = ["bond_type","stereo","is_conjugated"]

def load_graphs(pkl_path: str):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

class GraphCaptionDataset(Dataset):
    """
    Train/Val: returns (graph, text)
    Test: returns (graph, None)
    """
    def __init__(self, graph_path: str, split: str):
        self.graphs = load_graphs(graph_path)
        self.split = split
        self.ids = [g.id for g in self.graphs]

    def __len__(self): return len(self.graphs)

    def __getitem__(self, idx):
        g = self.graphs[idx]
        if self.split in ("train", "validation"):
            return g, g.description
        else:
            return g, None

def collate_caption_fn(batch, tokenizer: AutoTokenizer, max_len: int = 128):
    # batch: list of (graph, text_or_None)
    graphs, texts = zip(*batch)
    batch_graph = Batch.from_data_list(list(graphs))

    if texts[0] is None:
        return batch_graph, None, None  # test

    tok = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    # labels: pad -> -100
    labels = tok["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    return batch_graph, tok, labels
