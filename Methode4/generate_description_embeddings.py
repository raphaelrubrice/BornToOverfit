#!/usr/bin/env python3
"""
generate_description_embeddings.py

Generate text embeddings for molecular descriptions using ChemicalBERT.
- Encoder: recobo/chemical-bert-uncased
- Pooling: CLS token
- Splits: train + validation
- Output:
    data/train_embeddings.recobo_chemical-bert-uncased.cls.csv
    data/validation_embeddings.recobo_chemical-bert-uncased.cls.csv
"""

import os
import pickle
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


# =========================
# CONFIG
# =========================
MODEL_NAME = "recobo/chemical-bert-uncased"
MAX_TOKEN_LENGTH = 128
BATCH_SIZE = 64
DATA_DIR = "data"  # contains train_graphs.pkl, validation_graphs.pkl


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # (B,T,1)
    summed = (last_hidden_state * mask).sum(dim=1)                  # (B,H)
    denom = mask.sum(dim=1).clamp(min=1e-6)                         # (B,1)
    return summed / denom


@torch.no_grad()
def encode_batch(model, tokenizer, texts, device, max_len):
    tok = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    )
    tok = {k: v.to(device) for k, v in tok.items()}
    out = model(**tok)
    # CLS pooling
    emb = out.last_hidden_state[:, 0, :]  # (B,H)
    return emb.detach().cpu()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading text encoder: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    print(f"Model loaded on: {device}")

    for split in ["train", "validation"]:
        pkl_path = os.path.join(DATA_DIR, f"{split}_graphs.pkl")
        print(f"\nProcessing {split}...")
        print(f"Loading from {pkl_path}...")

        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Missing file: {pkl_path}")

        with open(pkl_path, "rb") as f:
            graphs = pickle.load(f)

        print(f"Loaded {len(graphs)} graphs")

        ids = [g.id for g in graphs]
        descs = [g.description for g in graphs]

        all_embs = []
        for i in tqdm(range(0, len(descs), BATCH_SIZE), desc=f"Embedding {split}"):
            batch_texts = descs[i:i + BATCH_SIZE]
            batch_emb = encode_batch(
                model=model,
                tokenizer=tokenizer,
                texts=batch_texts,
                device=device,
                max_len=MAX_TOKEN_LENGTH,
            )
            all_embs.append(batch_emb)

        embs = torch.cat(all_embs, dim=0).numpy()

        out_csv = os.path.join(
            DATA_DIR,
            f"{split}_embeddings.{MODEL_NAME.replace('/','_')}.cls.csv"
        )
        df = pd.DataFrame({
            "ID": ids,
            "embedding": [",".join(map(str, row)) for row in embs]
        })
        df.to_csv(out_csv, index=False)
        print(f"Saved to {out_csv}")

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
