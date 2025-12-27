# editor_model.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GINEConv
from transformers import GPT2Config, GPT2LMHeadModel

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
    def __init__(self, atom_vocab, bond_vocab, d=768, layers=4, dropout=0.1):
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
        return h

    @staticmethod
    def pad_nodes(batch: Batch, node_h: torch.Tensor):
        device = node_h.device
        B = int(batch.num_graphs)
        counts = torch.bincount(batch.batch, minlength=B)
        nmax = int(counts.max().item())
        enc_hidden = node_h.new_zeros((B, nmax, node_h.size(-1)))
        enc_mask = torch.zeros((B, nmax), device=device, dtype=torch.long)
        start = 0
        for g in range(B):
            n = int(counts[g].item())
            if n:
                enc_hidden[g, :n] = node_h[start:start+n]
                enc_mask[g, :n] = 1
            start += n
        return enc_hidden, enc_mask

class GraphCaptionEditor(nn.Module):
    def __init__(self, atom_vocab, bond_vocab, gpt2_name="gpt2", d=768, gnn_layers=4):
        super().__init__()
        self.enc = GraphEncoder(atom_vocab, bond_vocab, d=d, layers=gnn_layers)
        cfg = GPT2Config.from_pretrained(gpt2_name)
        cfg.add_cross_attention = True
        self.dec = GPT2LMHeadModel.from_pretrained(gpt2_name, config=cfg)

    def forward(self, batch_graph, input_ids, attention_mask, labels):
        node_h = self.enc(batch_graph)
        enc_hidden, enc_mask = self.enc.pad_nodes(batch_graph, node_h)
        return self.dec(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            encoder_hidden_states=enc_hidden,
            encoder_attention_mask=enc_mask,
        )

    @torch.no_grad()
    def generate(self, batch_graph, input_ids, attention_mask, tokenizer,
                 max_new_tokens=96, num_beams=5, length_penalty=1.0, no_repeat_ngram_size=3):
        node_h = self.enc(batch_graph)
        enc_hidden, enc_mask = self.enc.pad_nodes(batch_graph, node_h)
        return self.dec.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            encoder_hidden_states=enc_hidden,
            encoder_attention_mask=enc_mask,
        )
