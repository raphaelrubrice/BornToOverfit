#!/usr/bin/env python3
"""Generate embeddings for molecular descriptions using a fine-tuned model."""

import pickle
import pandas as pd
import torch
from peft import PeftModel
from transformers.models.llama import LlamaForCausalLM, LlamaTokenizer
from tqdm import tqdm
from pathlib import Path
import os
from argparse import ArgumentParser
import huggingface_hub

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-f", default="data_baseline/data", type=str)
    args = parser.parse_args()
    folder = args.f

    file_path = Path(os.path.abspath(__file__))
    parent_folder = file_path.parent.parent

    data_path = parent_folder / "data_baseline" / "data"
    save_path = parent_folder / folder

    # Configuration
    MAX_TOKEN_LENGTH = 128
    BATCH_SIZE = 32  # tune this (16/32/64 depending on GPU)
    base_model = "Allanatrix/nexa-Llama-sci7b"
    finetuned = "zjunlp/llama-molinst-molecule-7b"

    # Login
    YOUR_TOKEN = input("your hugging face Token:")
    huggingface_hub.login(
        token=YOUR_TOKEN,
        new_session=False,
        add_to_git_credential=True
    )

    # Load model
    print("Loading MolInstruct model...")
    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        dtype=torch.float16,     # use torch_dtype (preferred)
        device_map={"": 0}
    )
    model = PeftModel.from_pretrained(
        model,
        finetuned,
        dtype=torch.float16,
        device_map={"": 0},
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Not needed for embeddings; can reduce overhead for some configs
    if hasattr(model, "config"):
        model.config.use_cache = False

    print(f"Model loaded on: {device}")

    for split in ["train", "validation"]:
        print(f"\nProcessing {split}...")

        pkl_path = str(data_path / f"{split}_graphs.pkl")
        print(f"Loading from {pkl_path}...")
        with open(pkl_path, "rb") as f:
            graphs = pickle.load(f)
        print(f"Loaded {len(graphs)} graphs")

        ids = []
        embeddings = []

        # Pre-extract text/ids (reduces Python overhead inside loop)
        descriptions = [g.description for g in graphs]
        graph_ids = [g.id for g in graphs]

        for i in tqdm(range(0, len(graphs), BATCH_SIZE), total=(len(graphs) + BATCH_SIZE - 1) // BATCH_SIZE):
            batch_text = descriptions[i:i + BATCH_SIZE]
            batch_ids = graph_ids[i:i + BATCH_SIZE]

            inputs = tokenizer(
                batch_text,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_TOKEN_LENGTH,
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.inference_mode():
                # AMP speeds up matmuls on GPU
                if device.type == "cuda":
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
                else:
                    outputs = model(**inputs, output_hidden_states=True, return_dict=True)

                last_hidden = outputs.hidden_states[-1]  # (B, T, H)

                # Fetch a single "general hidden state" per sequence:
                # Use last non-pad token: index = attention_mask.sum(1) - 1
                attn = inputs["attention_mask"]  # (B, T)
                last_idx = attn.sum(dim=1) - 1   # (B,)
                batch_range = torch.arange(last_hidden.size(0), device=last_hidden.device)

                seq_emb = last_hidden[batch_range, last_idx, :]  # (B, H)

            ids.extend(batch_ids)
            embeddings.append(seq_emb.float().cpu().numpy())  # list of (B, H)

        # Stack to (N, H)
        import numpy as np
        embeddings = np.concatenate(embeddings, axis=0)

        # Save to CSV (still slow/large, but keeping your format)
        result = pd.DataFrame({
            "ID": ids,
            "embedding": [",".join(map(str, emb.tolist())) for emb in embeddings]
        })
        os.makedirs(str(save_path), exist_ok=True)
        output_path = str(save_path / f"{split}_embeddings.csv")
        result.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")

    print("\nDone!")