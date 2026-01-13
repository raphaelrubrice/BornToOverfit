# train_molca.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup

from data_utils import GraphCaptionDataset, collate_caption_fn
from model_molca import GraphEncoder, QFormerLite

TRAIN_GRAPHS = "data/train_graphs.pkl"
VAL_GRAPHS   = "data/validation_graphs.pkl"
TEST_GRAPHS  = "data/test_graphs.pkl"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class MolCAStyleCaptioner(nn.Module):
    def __init__(self, t5_name="google/flan-t5-base", hidden=256, num_queries=16):
        super().__init__()
        self.graph_enc = GraphEncoder(hidden=hidden, layers=4)
        self.qformer = QFormerLite(hidden=hidden, num_queries=num_queries, layers=2)

        self.t5 = AutoModelForSeq2SeqLM.from_pretrained(t5_name)
        d_model = self.t5.config.d_model
        self.proj = nn.Linear(hidden, d_model)

    def forward(self, batch_graph, labels):
        node_emb, batch_vec = self.graph_enc(batch_graph)
        q = self.qformer(node_emb, batch_vec)               # [B,Q,H]
        enc = self.proj(q)                                  # [B,Q,d_model]
        attn_mask = torch.ones(enc.shape[:2], device=enc.device, dtype=torch.long)

        out = self.t5(inputs_embeds=enc, attention_mask=attn_mask, labels=labels)
        return out.loss

@torch.no_grad()
def eval_loss(model, loader, device):
    model.eval()
    tot, n = 0.0, 0
    for graphs, tok, labels in loader:
        graphs = graphs.to(device)
        labels = labels.to(device)
        loss = model(graphs, labels)
        bs = graphs.num_graphs
        tot += loss.item() * bs
        n += bs
    return tot / max(1, n)

def main():
    t5_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(t5_name)

    train_ds = GraphCaptionDataset(TRAIN_GRAPHS, split="train")
    val_ds   = GraphCaptionDataset(VAL_GRAPHS, split="validation")

    train_dl = DataLoader(
        train_ds, batch_size=16, shuffle=True,
        collate_fn=lambda b: collate_caption_fn(b, tokenizer, max_len=128)
    )
    val_dl = DataLoader(
        val_ds, batch_size=16, shuffle=False,
        collate_fn=lambda b: collate_caption_fn(b, tokenizer, max_len=128)
    )

    model = MolCAStyleCaptioner(t5_name=t5_name, hidden=256, num_queries=16).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    steps = 3 * len(train_dl)
    sched = get_linear_schedule_with_warmup(opt, int(0.06*steps), steps)

    best = 1e9
    for ep in range(3):
        model.train()
        for graphs, tok, labels in train_dl:
            graphs = graphs.to(DEVICE)
            labels = labels.to(DEVICE)

            loss = model(graphs, labels)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

        vloss = eval_loss(model, val_dl, DEVICE)
        print(f"Epoch {ep+1}: val_loss={vloss:.4f}")

        if vloss < best:
            best = vloss
            torch.save(model.state_dict(), "molca_t5_best.pt")
            print("  -> saved molca_t5_best.pt")

if __name__ == "__main__":
    main()
