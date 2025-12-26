"""
Data loading and processing utilities for molecule->text captioning.

- Loads PyG graphs from .pkl (train/val/test)
- Provides datasets + collate functions for:
  * captioning training (graph + description -> tokenized)
  * test inference (graph only)
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Tuple
import pickle

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch


# =========================================================
# Feature maps for atom and bond attributes (categorical indices)
# =========================================================
x_map: Dict[str, List[Any]] = {
    "atomic_num": list(range(0, 119)),
    "chirality": [
        "CHI_UNSPECIFIED","CHI_TETRAHEDRAL_CW","CHI_TETRAHEDRAL_CW","CHI_OTHER",
        "CHI_TETRAHEDRAL","CHI_ALLENE","CHI_SQUAREPLANAR","CHI_TRIGONALBIPYRAMIDAL",
        "CHI_OCTAHEDRAL",
    ],
    "degree": list(range(0, 11)),
    "formal_charge": list(range(-5, 7)),
    "num_hs": list(range(0, 9)),
    "num_radical_electrons": list(range(0, 5)),
    "hybridization": ["UNSPECIFIED","S","SP","SP2","SP3","SP3D","SP3D2","OTHER"],
    "is_aromatic": [False, True],
    "is_in_ring": [False, True],
}

e_map: Dict[str, List[Any]] = {
    "bond_type": [
        "UNSPECIFIED","SINGLE","DOUBLE","TRIPLE","QUADRUPLE","QUINTUPLE","HEXTUPLE",
        "ONEANDAHALF","TWOANDAHALF","THREEANDAHALF","FOURANDAHALF","FIVEANDAHALF",
        "AROMATIC","IONIC","HYDROGEN","THREECENTER","DATIVEONE","DATIVE","DATIVEL",
        "DATIVER","OTHER","ZERO",
    ],
    "stereo": ["STEREONONE","STEREOANY","STEREOZ","STEREOE","STEREOCIS","STEREOTRANS"],
    "is_conjugated": [False, True],
}


def load_graphs(pkl_path: str):
    with open(pkl_path, "rb") as f:
        graphs = pickle.load(f)
    return graphs


# =========================================================
# Datasets
# =========================================================
class MoleculeCaptionDataset(Dataset):
    """
    Train/Val dataset: returns (graph, description, id)
    """
    def __init__(self, graph_path: str):
        print(f"Loading graphs from: {graph_path}")
        self.graphs = load_graphs(graph_path)
        self.ids = [g.id for g in self.graphs]
        print(f"Loaded {len(self.graphs)} graphs")

        # sanity: ensure descriptions exist
        has_desc = 0
        for g in self.graphs:
            if hasattr(g, "description") and isinstance(g.description, str) and len(g.description) > 0:
                has_desc += 1
        print(f"Graphs with non-empty descriptions: {has_desc}/{len(self.graphs)}")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx: int):
        g = self.graphs[idx]
        desc = getattr(g, "description", "")
        return g, desc, g.id


class MoleculeTestDataset(Dataset):
    """
    Test dataset: returns (graph, id)
    """
    def __init__(self, graph_path: str):
        print(f"Loading test graphs from: {graph_path}")
        self.graphs = load_graphs(graph_path)
        self.ids = [g.id for g in self.graphs]
        print(f"Loaded {len(self.graphs)} graphs")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx: int):
        g = self.graphs[idx]
        return g, g.id


# =========================================================
# Collate for captioning (tokenize text in collate)
# =========================================================
def make_caption_collate_fn(tokenizer, max_length: int = 128):
    """
    Returns a collate_fn that:
      - Batches graphs into a PyG Batch
      - Tokenizes descriptions
      - Creates labels with pad tokens masked to -100 (for GPT2 loss)
    """
    pad_id = tokenizer.pad_token_id

    def collate(batch: List[Tuple[Any, str, str]]):
        graphs, descs, ids = zip(*batch)
        batch_graph = Batch.from_data_list(list(graphs))

        tok = tokenizer(
            list(descs),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        input_ids = tok["input_ids"]
        attention_mask = tok["attention_mask"]

        labels = input_ids.clone()
        labels[labels == pad_id] = -100  # ignore pad in loss

        return batch_graph, input_ids, attention_mask, labels, list(ids)

    return collate


def test_collate_fn(batch: List[Tuple[Any, str]]):
    graphs, ids = zip(*batch)
    batch_graph = Batch.from_data_list(list(graphs))
    return batch_graph, list(ids)


