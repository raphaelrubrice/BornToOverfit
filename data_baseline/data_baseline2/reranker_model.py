# reranker_model.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GINEConv

from transformers import BertModel, BertConfig


class AtomEncoder(nn.Module):
    def __init__(self, vocab_sizes, d):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(v, d) for v in vocab_sizes])
        self.ln = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))

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
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, e):
        h = 0
        for i, emb in enumerate(self.embs):
            h = h + emb(e[:, i])
        return self.mlp(self.ln(h))


class GraphTokenEncoder(nn.Module):
    """
    Encodes graph into node embeddings, then selects top-K node tokens per graph.
    """
    def __init__(self, atom_vocab, bond_vocab, d=256, layers=4, dropout=0.1):
        super().__init__()
        self.d = d
        self.atom = AtomEncoder(atom_vocab, d)
        self.bond = BondEncoder(bond_vocab, d)
        self.convs = nn.ModuleList()
        self.lns = nn.ModuleList()
        self.drop = nn.Dropout(dropout)

        for _ in range(layers):
            mlp = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
            self.convs.append(GINEConv(nn=mlp, edge_dim=d))
            self.lns.append(nn.LayerNorm(d))

        # scoring for topK
        self.score = nn.Linear(d, 1)

    def forward(self, batch: Batch) -> torch.Tensor:
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
        return h  # (total_nodes, d)

    def topk_tokens(self, batch: Batch, node_h: torch.Tensor, K: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          tokens: (B, K, d) padded with zeros if graph has <K nodes
          mask:   (B, K) 1 valid else 0
        """
        device = node_h.device
        B = int(batch.num_graphs)
        counts = torch.bincount(batch.batch, minlength=B)
        tokens = node_h.new_zeros((B, K, node_h.size(-1)))
        mask = torch.zeros((B, K), device=device, dtype=torch.long)

        start = 0
        for g in range(B):
            n = int(counts[g].item())
            h_g = node_h[start:start + n]  # (n,d)
            start += n
            if n == 0:
                continue
            scores = self.score(h_g).squeeze(-1)  # (n,)
            kk = min(K, n)
            idx = torch.topk(scores, k=kk, largest=True).indices
            sel = h_g[idx]  # (kk,d)
            tokens[g, :kk] = sel
            mask[g, :kk] = 1
        return tokens, mask


class GraphTextReranker(nn.Module):
    """
    Cross-encoder:
      sequence = [CLS] + graph_tokens(K) + [SEP] + text_tokens + [SEP]
    fed into BERT encoder, then classify with CLS.
    """
    def __init__(
        self,
        atom_vocab,
        bond_vocab,
        bert_name: str = "bert-base-uncased",
        graph_dim: int = 256,
        graph_layers: int = 4,
        max_graph_tokens: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_graph_tokens = max_graph_tokens

        self.graph_enc = GraphTokenEncoder(atom_vocab, bond_vocab, d=graph_dim, layers=graph_layers, dropout=dropout)

        self.bert = BertModel.from_pretrained(bert_name)
        self.hidden = self.bert.config.hidden_size

        # project graph tokens into BERT hidden size
        self.graph_proj = nn.Linear(graph_dim, self.hidden)
        self.drop = nn.Dropout(dropout)
        self.cls_head = nn.Linear(self.hidden, 1)

        # cache special token ids
        self.cls_id = self.bert.config.bos_token_id if self.bert.config.bos_token_id is not None else 101  # [CLS]=101
        self.sep_id = 102  # [SEP]
        # word embeddings access
        self.word_emb = self.bert.embeddings.word_embeddings

    def build_inputs_embeds(
        self,
        graph_tokens: torch.Tensor,     # (B,K,graph_dim)
        graph_mask: torch.Tensor,       # (B,K)
        text_input_ids: torch.Tensor,   # (B,T)
        text_attn: torch.Tensor,        # (B,T)
    ):
        device = text_input_ids.device
        B, T = text_input_ids.shape
        K = graph_tokens.shape[1]

        g = self.graph_proj(graph_tokens)  # (B,K,H)

        cls_emb = self.word_emb(torch.full((B, 1), 101, device=device, dtype=torch.long))  # [CLS]
        sep_emb = self.word_emb(torch.full((B, 1), 102, device=device, dtype=torch.long))  # [SEP]
        text_emb = self.word_emb(text_input_ids)  # (B,T,H)

        # [CLS] + G + [SEP] + text + [SEP]
        inputs_embeds = torch.cat([cls_emb, g, sep_emb, text_emb, sep_emb], dim=1)  # (B, 1+K+1+T+1, H)

        # attention mask
        attn = torch.cat(
            [
                torch.ones((B, 1), device=device, dtype=torch.long),  # CLS
                graph_mask,                                          # K
                torch.ones((B, 1), device=device, dtype=torch.long),  # SEP
                text_attn,                                           # T
                torch.ones((B, 1), device=device, dtype=torch.long),  # SEP
            ],
            dim=1,
        )

        # token types: 0 for graph side, 1 for text side
        token_type_ids = torch.cat(
            [
                torch.zeros((B, 1 + K + 1), device=device, dtype=torch.long),
                torch.ones((B, T + 1), device=device, dtype=torch.long),
            ],
            dim=1,
        )

        return inputs_embeds, attn, token_type_ids

    def forward(self, batch_graph: Batch, text_input_ids: torch.Tensor, text_attn: torch.Tensor):
        node_h = self.graph_enc(batch_graph)  # (total_nodes, graph_dim)
        gtok, gmask = self.graph_enc.topk_tokens(batch_graph, node_h, K=self.max_graph_tokens)  # (B,K,d),(B,K)
        gtok = self.drop(gtok)

        inputs_embeds, attn, token_type_ids = self.build_inputs_embeds(
            graph_tokens=gtok,
            graph_mask=gmask,
            text_input_ids=text_input_ids,
            text_attn=text_attn,
        )

        out = self.bert(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        cls = out.last_hidden_state[:, 0, :]  # (B,H)
        logit = self.cls_head(self.drop(cls)).squeeze(-1)  # (B,)
        return logit
