#!/usr/bin/env python3
"""
generate_description_embeddings.py

Generate text embeddings for descriptions using a HuggingFace encoder.

Examples:
python generate_description_embeddings.py \
  --model_name bert-base-uncased --pooling cls \
  --max_len 128 --batch_size 64 --splits train validation

Outputs:
data/train_embeddings.<model>.<pooling>.csv
data/validation_embeddings.<model>.<pooling>.csv
"""

import os
import argparse
import pickle
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


def mean_pool(last_hidden_state, attention_mask):
    # last_hidden_state: (B, T, H)
    # attention_mask: (B, T)
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # (B,T,1)
    summed = (last_hidden_state * mask).sum(dim=1)                  # (B,H)
    denom = mask.sum(dim=1).clamp(min=1e-6)                         # (B,1)
    return summed / denom


@torch.no_grad()
def encode_texts(model, tokenizer, texts, device, max_len, pooling):
    tok = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    ).to(device)

    out = model(**tok)
    if pooling == "cls":
        emb = out.last_hidden_state[:, 0, :]   # (B,H)
    else:
        emb = mean_pool(out.last_hidden_state, tok["attention_mask"])
    return emb.detach().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="bert-base-uncased")
    ap.add_argument("--pooling", type=str, default="cls", choices=["cls", "mean"])
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--splits", nargs="+", default=["train", "validation"])
    ap.add_argument("--data_dir", type=str, default="data")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Text encoder:", args.model_name, "| pooling:", args.pooling)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)
    model.eval()

    for split in args.splits:
        pkl_path = os.path.join(args.data_dir, f"{split}_graphs.pkl")
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Missing: {pkl_path}")

        with open(pkl_path, "rb") as f:
            graphs = pickle.load(f)

        ids = [g.id for g in graphs]
        descs = [g.description for g in graphs]

        embs = []
        for i in tqdm(range(0, len(descs), args.batch_size), desc=f"Embedding {split}"):
            batch_texts = descs[i:i + args.batch_size]
            batch_emb = encode_texts(
                model=model,
                tokenizer=tokenizer,
                texts=batch_texts,
                device=device,
                max_len=args.max_len,
                pooling=args.pooling,
            )
            embs.append(batch_emb)

        embs = torch.cat(embs, dim=0).numpy()

        out_csv = os.path.join(
            args.data_dir,
            f"{split}_embeddings.{args.model_name.replace('/','_')}.{args.pooling}.csv"
        )
        df = pd.DataFrame({
            "ID": ids,
            "embedding": [",".join(map(str, row)) for row in embs]
        })
        df.to_csv(out_csv, index=False)
        print("Saved:", out_csv)

    print("\nDone!")


if __name__ == "__main__":
    main()
