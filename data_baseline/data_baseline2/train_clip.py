# train_clip.py
import os, json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from data_utils import GraphTextDataset, collate_graph_text, atom_vocab_sizes, bond_vocab_sizes
from clip_model import CLIPGraphText

TRAIN = "data/train_graphs.pkl"
VAL   = "data/validation_graphs.pkl"
OUTDIR = "clip_ckpt"
CKPT  = os.path.join(OUTDIR, "clip.pt")
TRAIN_TEXT_EMB = os.path.join(OUTDIR, "train_text_emb.pt")  # tensors + ids + desc

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 64
EPOCHS = 3
LR = 2e-5
MAXLEN = 128
TEXT_MODEL = "bert-base-uncased"
GRAPH_DIM = 256

def main():
    os.makedirs(OUTDIR, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(TEXT_MODEL)
    ds = GraphTextDataset(TRAIN)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, collate_fn=collate_graph_text)

    model = CLIPGraphText(atom_vocab_sizes(), bond_vocab_sizes(), graph_dim=GRAPH_DIM, text_model=TEXT_MODEL).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    for ep in range(1, EPOCHS+1):
        model.train()
        pbar = tqdm(dl, desc=f"CLIP Epoch {ep}/{EPOCHS}")
        for batch_graph, texts, ids in pbar:
            batch_graph = batch_graph.to(DEVICE)
            enc = tok(texts, padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt")
            input_ids = enc["input_ids"].to(DEVICE)
            attention_mask = enc["attention_mask"].to(DEVICE)

            logits = model(batch_graph, input_ids, attention_mask)
            labels = torch.arange(logits.size(0), device=DEVICE)
            loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2

            opt.zero_grad()
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=float(loss.item()))

    torch.save({"state": model.state_dict(), "text_model": TEXT_MODEL, "graph_dim": GRAPH_DIM}, CKPT)
    print("Saved:", CKPT)

    # Precompute and save train text embeddings (for fast retrieval/rerank)
    model.eval()
    all_ids, all_desc, all_z = [], [], []
    dl2 = DataLoader(ds, batch_size=BATCH, shuffle=False, collate_fn=collate_graph_text)
    with torch.no_grad():
        for batch_graph, texts, ids in tqdm(dl2, desc="Encode train text"):
            enc = tok(texts, padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt")
            zt = model.t(enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)).cpu()
            all_z.append(zt)
            all_ids.extend(ids)
            all_desc.extend(texts)
    Z = torch.cat(all_z, dim=0)  # (N, dim)
    torch.save({"ids": all_ids, "desc": all_desc, "Z": Z}, TRAIN_TEXT_EMB)
    print("Saved:", TRAIN_TEXT_EMB)

if __name__ == "__main__":
    main()
