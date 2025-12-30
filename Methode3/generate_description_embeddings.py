#!/usr/bin/env python3
"""
Generate embeddings for molecular descriptions using various Transformer architectures.

Supports:
  - BERT/RoBERTa/DeBERTa/MPNet (Encoder-only, uses [CLS])
  - GPT/Llama/Qwen (Decoder-only, uses last non-pad token)
  - T5/mt5 encoders (Mean pooling)

IMPORTANT for your benchmark:
- This script can now save embeddings with a unique suffix/tag so you can compare many text encoders
  without overwriting previous CSVs.
"""

import os
import re
import pickle
import pandas as pd
import torch
import transformers
from transformers import AutoTokenizer, AutoModel, T5EncoderModel
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser

transformers.logging.set_verbosity_error()


def sanitize_tag(s: str) -> str:
    s = s.strip()
    s = s.replace("/", "__")
    s = re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", s)
    return s


def intelligent_pooling(outputs, inputs, config):
    """
    Dynamically extracts the best sentence embedding based on model architecture.
    """
    last_hidden_state = outputs.last_hidden_state
    attention_mask = inputs["attention_mask"]
    model_type = (config.model_type or "").lower()

    # 1) Encoder-only: take CLS
    if any(k in model_type for k in ["bert", "roberta", "mpnet", "electra", "deberta"]):
        return last_hidden_state[:, 0, :]

    # 2) Decoder-only: take last non-pad
    elif any(k in model_type for k in ["gpt", "llama", "mistral", "qwen", "falcon"]):
        seq_lens = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_state.shape[0]
        return last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), seq_lens]

    # 3) T5-style: mean pooling
    elif any(k in model_type for k in ["t5", "mt5"]):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_emb = torch.sum(last_hidden_state * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        return sum_emb / sum_mask

    # Fallback: mean pooling
    else:
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_emb = torch.sum(last_hidden_state * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        return sum_emb / sum_mask


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--base_model", type=str, default="bert-base-uncased",
                        help="HF model name (e.g. bert-base-uncased, allenai/scibert_scivocab_uncased)")
    parser.add_argument("--tag", type=str, default=None,
                        help="Suffix for output files. If not set, derived from base_model.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--normalize", action="store_true",
                        help="L2-normalize embeddings before saving (recommended).")
    parser.add_argument("--splits", type=str, default="train,validation",
                        help="Comma-separated splits to process: train,validation")

    parser.add_argument("-f_data", default="data_baseline/data", type=str,
                        help="Folder containing source .pkl graph files")
    parser.add_argument("-f", default="data_baseline/data", type=str,
                        help="Output folder for embeddings")

    args = parser.parse_args()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    file_path = Path(os.path.abspath(__file__))
    parent_folder = file_path.parent

    data_path = parent_folder.parent / args.f_data
    out_path  = parent_folder.parent / args.f
    os.makedirs(str(out_path), exist_ok=True)

    tag = sanitize_tag(args.tag) if args.tag else sanitize_tag(args.base_model)

    print(f"Loading model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if "t5" in args.base_model.lower():
        model = T5EncoderModel.from_pretrained(args.base_model, return_dict=True)
    else:
        model = AutoModel.from_pretrained(args.base_model, return_dict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    print(f"Model loaded on: {device}")

    for split in splits:
        pkl_path = str(data_path / f"{split}_graphs.pkl")
        if not os.path.exists(pkl_path):
            print(f"Skipping {split}, file not found: {pkl_path}")
            continue

        print(f"\nProcessing {split} from {pkl_path}")
        with open(pkl_path, "rb") as f:
            graphs = pickle.load(f)

        descriptions = [g.description for g in graphs]
        graph_ids = [g.id for g in graphs]

        all_ids = []
        all_embs = []

        for i in tqdm(range(0, len(graphs), args.batch_size),
                      total=(len(graphs) + args.batch_size - 1) // args.batch_size):
            batch_text = descriptions[i:i + args.batch_size]
            batch_ids  = graph_ids[i:i + args.batch_size]

            inputs = tokenizer(
                batch_text,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_len,
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            emb = intelligent_pooling(outputs, inputs, model.config)  # (B, D)

            if args.normalize:
                emb = torch.nn.functional.normalize(emb, dim=-1)

            all_ids.extend(batch_ids)
            all_embs.extend(emb.cpu().numpy())

        # Save (unique filename per model)
        output_csv = out_path / f"{split}_embeddings__{tag}.csv"
        df = pd.DataFrame({
            "ID": all_ids,
            "embedding": [",".join(map(str, e.tolist())) for e in all_embs]
        })
        df.to_csv(str(output_csv), index=False)

        # Print embedding dim sanity
        emb_dim = len(all_embs[0]) if len(all_embs) else None
        print(f"Saved {split} embeddings to {output_csv} (dim={emb_dim})")

    print("\n✅ Embedding generation complete!")
