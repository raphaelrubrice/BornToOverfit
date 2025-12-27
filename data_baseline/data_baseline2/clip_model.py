# clip_model.py
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GINEConv, global_mean_pool
from transformers import AutoModel, AutoTokenizer

class AtomEncoder(nn.Module):
    def __init__(self, vocab_sizes, d):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(v, d) for v in vocab_sizes])
        self.ln = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d,d), nn.GELU(), nn.Linear(d,d))
    def forward(self, x):
        h = 0
        for i, emb in enumerate(self.embs):
            h = h + emb(x[:, i])
        return self.mlp(self.ln(h))

class BondEncoder(nn.Module):
    def __init__(self, vocab_sizes, d):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(v, d) for v in vocab_sizes])
        self.ln = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d,d), nn.GELU(), nn.Linear(d,d))
    def forward(self, e):
        h = 0
        for i, emb in enumerate(self.embs):
            h = h + emb(e[:, i])
        return self.mlp(self.ln(h))

class GraphEncoder(nn.Module):
    def __init__(self, atom_vocab, bond_vocab, d=256, layers=4, dropout=0.1):
        super().__init__()
        self.atom = AtomEncoder(atom_vocab, d)
        self.bond = BondEncoder(bond_vocab, d)
        self.convs = nn.ModuleList()
        self.lns = nn.ModuleList()
        self.drop = nn.Dropout(dropout)
        for _ in range(layers):
            mlp = nn.Sequential(nn.Linear(d,d), nn.ReLU(), nn.Linear(d,d))
            self.convs.append(GINEConv(nn=mlp, edge_dim=d))
            self.lns.append(nn.LayerNorm(d))
        self.proj = nn.Linear(d, d)

    def forward(self, batch: Batch):
        x = batch.x.long()
        ea = batch.edge_attr.long()
        h = self.atom(x)
        e = self.bond(ea)
        for conv, ln in zip(self.convs, self.lns):
            h_in = h
            h = conv(h, batch.edge_index, e)
            h = F.gelu(h)
            h = self.drop(h)
            h = ln(h + h_in)
        g = global_mean_pool(h, batch.batch)
        g = self.proj(g)
        return F.normalize(g, dim=-1)

class TextEncoder(nn.Module):
    def __init__(self, name="bert-base-uncased", out_dim=256):
        super().__init__()
        self.name = name
        self.model = AutoModel.from_pretrained(name)
        self.proj = nn.Linear(self.model.config.hidden_size, out_dim)

    def mean_pool(self, last_hidden, attn_mask):
        mask = attn_mask.unsqueeze(-1).float()
        s = (last_hidden * mask).sum(dim=1)
        d = mask.sum(dim=1).clamp(min=1.0)
        return s / d

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.mean_pool(out.last_hidden_state, attention_mask)
        z = self.proj(pooled)
        return F.normalize(z, dim=-1)

class CLIPGraphText(nn.Module):
    def __init__(self, atom_vocab, bond_vocab, graph_dim=256, text_model="bert-base-uncased", tau_init=0.07):
        super().__init__()
        self.g = GraphEncoder(atom_vocab, bond_vocab, d=graph_dim)
        self.t = TextEncoder(text_model, out_dim=graph_dim)
        self.logit_scale = nn.Parameter(torch.tensor(1.0 / tau_init).log())

    def forward(self, batch_graph, input_ids, attention_mask):
        zg = self.g(batch_graph)
        zt = self.t(input_ids, attention_mask)
        # temperature
        scale = self.logit_scale.exp().clamp(1e-3, 100.0)
        logits = scale * (zg @ zt.t())
        return logits
