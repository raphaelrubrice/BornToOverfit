# data_utils.py
from __future__ import annotations
from typing import List, Tuple, Any, Dict, Optional
import pickle
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

# feature vocab sizes
x_map = {
    'atomic_num': list(range(0, 119)),
    'chirality': [
        'CHI_UNSPECIFIED','CHI_TETRAHEDRAL_CW','CHI_TETRAHEDRAL_CCW','CHI_OTHER',
        'CHI_TETRAHEDRAL','CHI_ALLENE','CHI_SQUAREPLANAR','CHI_TRIGONALBIPYRAMIDAL',
        'CHI_OCTAHEDRAL',
    ],
    'degree': list(range(0, 11)),
    'formal_charge': list(range(-5, 7)),
    'num_hs': list(range(0, 9)),
    'num_radical_electrons': list(range(0, 5)),
    'hybridization': ['UNSPECIFIED','S','SP','SP2','SP3','SP3D','SP3D2','OTHER'],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}
e_map = {
    'bond_type': [
        'UNSPECIFIED','SINGLE','DOUBLE','TRIPLE','QUADRUPLE','QUINTUPLE','HEXTUPLE',
        'ONEANDAHALF','TWOANDAHALF','THREEANDAHALF','FOURANDAHALF','FIVEANDAHALF',
        'AROMATIC','IONIC','HYDROGEN','THREECENTER','DATIVEONE','DATIVE','DATIVEL',
        'DATIVER','OTHER','ZERO',
    ],
    'stereo': ['STEREONONE','STEREOANY','STEREOZ','STEREOE','STEREOCIS','STEREOTRANS'],
    'is_conjugated': [False, True],
}

def atom_vocab_sizes():
    return [
        len(x_map["atomic_num"]),
        len(x_map["chirality"]),
        len(x_map["degree"]),
        len(x_map["formal_charge"]),
        len(x_map["num_hs"]),
        len(x_map["num_radical_electrons"]),
        len(x_map["hybridization"]),
        len(x_map["is_aromatic"]),
        len(x_map["is_in_ring"]),
    ]

def bond_vocab_sizes():
    return [
        len(e_map["bond_type"]),
        len(e_map["stereo"]),
        len(e_map["is_conjugated"]),
    ]

def load_graphs(pkl_path: str):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

class GraphTextDataset(Dataset):
    """(graph, description, id)"""
    def __init__(self, pkl_path: str, max_items: Optional[int]=None):
        self.graphs = load_graphs(pkl_path)
        if max_items is not None:
            self.graphs = self.graphs[:max_items]
        self.ids = [g.id for g in self.graphs]

    def __len__(self): return len(self.graphs)

    def __getitem__(self, idx: int):
        g = self.graphs[idx]
        return g, getattr(g, "description", ""), g.id

class GraphOnlyDataset(Dataset):
    """(graph, id)"""
    def __init__(self, pkl_path: str):
        self.graphs = load_graphs(pkl_path)
        self.ids = [g.id for g in self.graphs]

    def __len__(self): return len(self.graphs)

    def __getitem__(self, idx: int):
        g = self.graphs[idx]
        return g, g.id

def collate_graph_text(batch: List[Tuple[Any, str, str]]):
    graphs, texts, ids = zip(*batch)
    return Batch.from_data_list(list(graphs)), list(texts), list(ids)

def collate_graph_only(batch: List[Tuple[Any, str]]):
    graphs, ids = zip(*batch)
    return Batch.from_data_list(list(graphs)), list(ids)
