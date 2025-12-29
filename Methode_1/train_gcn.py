import os
import random
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from data_utils import (
    MoleculeCaptionDataset,
    make_caption_collate_fn,
    x_map,
    e_map,
)

from model_captioning import Graph2TextModel


# =========================================================
# CONFIG
# =========================================================
TRAIN_GRAPHS = "data/train_graphs.pkl"
VAL_GRAPHS   = "data/validation_graphs.pkl"
TEST_GRAPHS  = "data/test_graphs.pkl"

MODEL_DIR = "checkpoints"
CKPT_PATH = os.path.join(MODEL_DIR, "graph2text_gpt2.pt")

GPT2_NAME = "gpt2"          # keep for tokenizer & decoder
MAX_LEN = 128

BATCH_SIZE = 16
EPOCHS = 15
LR = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
GRAD_CLIP = 1.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def greedy_validate(model, tokenizer, val_loader, device, max_batches: int = 30):
    """
    Quick qualitative validation: generate a few samples each epoch.
    Not a full BLEU/BERTScore eval (kept simple).
    """
    model.eval()
    samples = []
    for i, (g, input_ids, attn_mask, labels, ids) in enumerate(val_loader):
        if i >= max_batches:
            break
        g = g.to(device)
        gen_ids = model.generate(g, tokenizer, max_new_tokens=96, num_beams=3)
        texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        # Store first 2
        for t in texts[:2]:
            samples.append(t.strip())
        if len(samples) >= 6:
            break
    return samples


def main():
    seed_everything(SEED)
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(GPT2_NAME)
    # GPT2 has no pad token by default => set pad = eos
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Datasets
    train_ds = MoleculeCaptionDataset(TRAIN_GRAPHS)
    val_ds = MoleculeCaptionDataset(VAL_GRAPHS) if os.path.exists(VAL_GRAPHS) else None

    train_collate = make_caption_collate_fn(tokenizer, max_length=MAX_LEN)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=train_collate, num_workers=0)

    val_dl = None
    if val_ds is not None:
        val_collate = make_caption_collate_fn(tokenizer, max_length=MAX_LEN)
        val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=val_collate, num_workers=0)

    # Vocab sizes for embeddings
    atom_vocab_sizes = [
        len(x_map["atomic_num"]),
        len(x_map["chirality"]),
        len(x_map["degree"]),
        len(x_map["formal_charge"]),
        len(x_map["num_hs"]),
        len(x_map["num_radical_electrons"]),
        len(x_map["hybridization"]),
        len(x_map["is_aromatic"]),
        len(x_map["is_in_ring"]),
    ]
    bond_vocab_sizes = [
        len(e_map["bond_type"]),
        len(e_map["stereo"]),
        len(e_map["is_conjugated"]),
    ]

    # Model
    model = Graph2TextModel(
        atom_vocab_sizes=atom_vocab_sizes,
        bond_vocab_sizes=bond_vocab_sizes,
        d_model=768,
        gnn_layers=4,
        gpt2_name=GPT2_NAME,
        dropout=0.1,
    ).to(DEVICE)

    # Optim & sched
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    total_steps = EPOCHS * len(train_dl)
    warmup_steps = int(WARMUP_RATIO * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_loss = float("inf")

    for ep in range(1, EPOCHS + 1):
        model.train()
        pbar = tqdm(train_dl, desc=f"Epoch {ep}/{EPOCHS}")
        total_loss = 0.0
        total_items = 0

        for batch_graph, input_ids, attn_mask, labels, ids in pbar:
            batch_graph = batch_graph.to(DEVICE)
            input_ids = input_ids.to(DEVICE)
            attn_mask = attn_mask.to(DEVICE)
            labels = labels.to(DEVICE)

            out = model(
                batch_graph=batch_graph,
                input_ids=input_ids,
                attention_mask=attn_mask,
                labels=labels,
            )
            loss = out.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            bs = batch_graph.num_graphs
            total_loss += loss.item() * bs
            total_items += bs
            pbar.set_postfix(loss=loss.item())

        train_loss = total_loss / max(1, total_items)
        print(f"\nEpoch {ep}: train_loss={train_loss:.4f}")

        # Validation loss (teacher forcing)
        val_loss = None
        if val_dl is not None:
            model.eval()
            vloss_sum, vcount = 0.0, 0
            for batch_graph, input_ids, attn_mask, labels, ids in tqdm(val_dl, desc="Valid", leave=False):
                batch_graph = batch_graph.to(DEVICE)
                input_ids = input_ids.to(DEVICE)
                attn_mask = attn_mask.to(DEVICE)
                labels = labels.to(DEVICE)

                out = model(
                    batch_graph=batch_graph,
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    labels=labels,
                )
                bs = batch_graph.num_graphs
                vloss_sum += out.loss.item() * bs
                vcount += bs

            val_loss = vloss_sum / max(1, vcount)
            print(f"Epoch {ep}: val_loss={val_loss:.4f}")

            # quick qualitative generation
            samples = greedy_validate(model, tokenizer, val_dl, DEVICE, max_batches=10)
            print("\n[Sample generations]")
            for s in samples[:5]:
                print("-", s[:200])

        # Save best
        metric = val_loss if val_loss is not None else train_loss
        if metric < best_val_loss:
            best_val_loss = metric
            ckpt = {
                "model_state": model.state_dict(),
                "tokenizer_name": GPT2_NAME,
                "max_len": MAX_LEN,
            }
            torch.save(ckpt, CKPT_PATH)
            print(f"\n✅ Saved best checkpoint to {CKPT_PATH}")

    print("\nDone.")


if __name__ == "__main__":
    main()
