"""
models_gps.py - GPS graph encoder for molecule->text embedding alignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GPSConv, GINEConv, global_add_pool

from data_utils import x_map, e_map

# =========================================================
# Feature dimensions
# =========================================================
ATOM_FEATURE_DIMS = [
    len(x_map['atomic_num']), len(x_map['chirality']), len(x_map['degree']),
    len(x_map['formal_charge']), len(x_map['num_hs']), len(x_map['num_radical_electrons']),
    len(x_map['hybridization']), len(x_map['is_aromatic']), len(x_map['is_in_ring']),
]
BOND_FEATURE_DIMS = [
    len(e_map['bond_type']), len(e_map['stereo']), len(e_map['is_conjugated']),
]


class AtomEncoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(dim, hidden_dim) for dim in ATOM_FEATURE_DIMS])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.long()
        return sum(emb(x[:, i]) for i, emb in enumerate(self.embeddings))


class BondEncoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(dim, hidden_dim) for dim in BOND_FEATURE_DIMS])

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        edge_attr = edge_attr.long()
        return sum(emb(edge_attr[:, i]) for i, emb in enumerate(self.embeddings))


class MolGPS(nn.Module):
    """
    GPS = Local MPNN (GINE) + Global Attention (Transformer-style).
    """
    def __init__(
        self,
        hidden_dim: int = 256,
        out_dim: int = 768,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        attn_type: str = "multihead",
    ):
        super().__init__()
        self.atom_encoder = AtomEncoder(hidden_dim)
        self.bond_encoder = BondEncoder(hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            local_nn = nn.Sequential(
                nn.Linear(hidden_dim, 2 * hidden_dim),
                nn.BatchNorm1d(2 * hidden_dim),
                nn.ReLU(),
                nn.Linear(2 * hidden_dim, hidden_dim),
            )
            local_conv = GINEConv(local_nn, train_eps=True, edge_dim=hidden_dim)

            gps_conv = GPSConv(
                channels=hidden_dim,
                conv=local_conv,
                heads=num_heads,
                dropout=dropout,
                attn_type=attn_type,  # 'multihead' or 'performer'
            )
            self.convs.append(gps_conv)

        self.pool = global_add_pool
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, batch: Batch) -> torch.Tensor:
        h = self.atom_encoder(batch.x)
        e = self.bond_encoder(batch.edge_attr)

        for conv in self.convs:
            h = conv(h, batch.edge_index, batch.batch, edge_attr=e)

        g = self.pool(h, batch.batch)
        g = self.proj(g)
        return F.normalize(g, dim=-1)
