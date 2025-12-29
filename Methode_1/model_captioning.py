from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GINEConv

from transformers import GPT2Config, GPT2LMHeadModel


@dataclass
class GraphPadOutput:
    enc_hidden: torch.Tensor          # (B, Nmax, d_model)
    enc_attn_mask: torch.Tensor       # (B, Nmax) 1=valid,0=pad


class AtomEncoder(nn.Module):
    """
    Encodes categorical atom features x[:,0..8] using embeddings.
    """
    def __init__(self, vocab_sizes, d_model: int):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(v, d_model) for v in vocab_sizes])
        self.ln = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 9) int64
        h = 0
        for i, emb in enumerate(self.embs):
            h = h + emb(x[:, i])
        h = self.ln(h)
        h = self.mlp(h)
        return h


class BondEncoder(nn.Module):
    """
    Encodes categorical bond features edge_attr[:,0..2] using embeddings.
    """
    def __init__(self, vocab_sizes, d_model: int):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(v, d_model) for v in vocab_sizes])
        self.ln = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        # edge_attr: (E, 3) int64
        e = 0
        for i, emb in enumerate(self.embs):
            e = e + emb(edge_attr[:, i])
        e = self.ln(e)
        e = self.mlp(e)
        return e


class GraphEncoder(nn.Module):
    """
    Node-token encoder with GINEConv layers.
    Output: node embeddings per graph (padded batch) for cross-attention.
    """
    def __init__(
        self,
        atom_vocab_sizes,
        bond_vocab_sizes,
        d_model: int = 768,
        gnn_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.atom_enc = AtomEncoder(atom_vocab_sizes, d_model)
        self.bond_enc = BondEncoder(bond_vocab_sizes, d_model)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.drop = nn.Dropout(dropout)

        for _ in range(gnn_layers):
            # GINE requires an "nn" that maps node messages; edge features are added internally.
            mlp = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
            )
            self.convs.append(GINEConv(nn=mlp, edge_dim=d_model))
            self.norms.append(nn.LayerNorm(d_model))

    def forward(self, batch: Batch) -> torch.Tensor:
        # batch.x: (total_nodes, 9)
        # batch.edge_attr: (total_edges, 3)
        x = batch.x.long()
        edge_attr = batch.edge_attr.long()

        h = self.atom_enc(x)                       # (total_nodes, d_model)
        e = self.bond_enc(edge_attr)               # (total_edges, d_model)

        for conv, ln in zip(self.convs, self.norms):
            h_in = h
            h = conv(h, batch.edge_index, e)
            h = F.gelu(h)
            h = self.drop(h)
            h = ln(h + h_in)  # residual
        return h  # node embeddings (total_nodes, d_model)

    @staticmethod
    def pad_nodes(batch: Batch, node_h: torch.Tensor) -> GraphPadOutput:
        """
        Convert (total_nodes, d) + batch.batch to padded (B, Nmax, d)
        """
        device = node_h.device
        B = int(batch.num_graphs)
        batch_vec = batch.batch  # (total_nodes,) graph index per node
        counts = torch.bincount(batch_vec, minlength=B)
        nmax = int(counts.max().item())

        enc_hidden = node_h.new_zeros((B, nmax, node_h.size(-1)))
        enc_mask = torch.zeros((B, nmax), device=device, dtype=torch.long)

        # fill per graph
        start = 0
        for g in range(B):
            n = int(counts[g].item())
            if n > 0:
                enc_hidden[g, :n, :] = node_h[start:start+n, :]
                enc_mask[g, :n] = 1
            start += n

        return GraphPadOutput(enc_hidden=enc_hidden, enc_attn_mask=enc_mask)


class Graph2TextModel(nn.Module):
    """
    Graph encoder -> GPT2 decoder with cross-attention.
    """
    def __init__(
        self,
        atom_vocab_sizes,
        bond_vocab_sizes,
        d_model: int = 768,
        gnn_layers: int = 4,
        gpt2_name: str = "gpt2",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = GraphEncoder(
            atom_vocab_sizes=atom_vocab_sizes,
            bond_vocab_sizes=bond_vocab_sizes,
            d_model=d_model,
            gnn_layers=gnn_layers,
            dropout=dropout,
        )

        # GPT2 with cross-attention enabled (loads pretrained weights, cross-attn is random init)
        config = GPT2Config.from_pretrained(gpt2_name)
        config.add_cross_attention = True
        config.n_embd = d_model  # ensure dims match
        # NOTE: If you change n_embd != original, pretrained weights won't load.
        # So keep d_model = 768 to use pretrained "gpt2" properly.
        self.decoder = GPT2LMHeadModel.from_pretrained(gpt2_name, config=config)

    def forward(
        self,
        batch_graph: Batch,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        # Encode graph -> padded node tokens
        node_h = self.encoder(batch_graph)  # (total_nodes, d_model)
        pad = self.encoder.pad_nodes(batch_graph, node_h)
        # Run GPT2 with cross-attn
        out = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            encoder_hidden_states=pad.enc_hidden,
            encoder_attention_mask=pad.enc_attn_mask,
        )
        return out

    @torch.no_grad()
    def generate(
        self,
        batch_graph: Batch,
        tokenizer,
        max_new_tokens: int = 128,
        num_beams: int = 5,
        length_penalty: float = 1.0,
    ) -> torch.Tensor:
        node_h = self.encoder(batch_graph)
        pad = self.encoder.pad_nodes(batch_graph, node_h)

        B = int(batch_graph.num_graphs)
        # GPT2 usually uses eos as bos; start token = eos
        start_id = tokenizer.eos_token_id
        input_ids = torch.full((B, 1), start_id, dtype=torch.long, device=node_h.device)

        gen = self.decoder.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            length_penalty=length_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            encoder_hidden_states=pad.enc_hidden,
            encoder_attention_mask=pad.enc_attn_mask,
        )
        return gen
