import os
import random

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

GPT2_NAME = "gpt2"
MAX_LEN = 128

BATCH_SIZE = 16
EPOCHS = 10              
LR = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
GRAD_CLIP = 1.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# Checkpoints (Drive-friendly)
MODEL_DIR = os.environ.get("CHECKPOINT_DIR", "checkpoints")
os.makedirs(MODEL_DIR, exist_ok=True)
LAST_CKPT_PATH = os.path.join(MODEL_DIR, "graph2text_gpt2_last.pt")
BEST_CKPT_PATH = os.path.join(MODEL_DIR, "graph2text_gpt2_best.pt")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_ckpt(path, model, optimizer, scheduler, epoch, best_val_loss, extra: dict | None = None):
    ckpt = {
        "epoch": epoch,
        "best_val_loss": float(best_val_loss),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_ckpt(path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    last_epoch = int(ckpt["epoch"])
    best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
    return last_epoch, best_val_loss


@torch.no_grad()
def quick_samples(model, tokenizer, val_loader, device, n_batches: int = 3):
    model.eval()
    outs = []
    for i, (g, input_ids, attn_mask, labels, ids) in enumerate(val_loader):
        if i >= n_batches:
            break
        g = g.to(device)
        gen_ids = model.generate(g, tokenizer, max_new_tokens=96, num_beams=3, length_penalty=1.0)
        texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        outs.extend([t.strip() for t in texts[:2]])
    return outs[:6]


def main():
    seed_everything(SEED)
    print(f"Device: {DEVICE}")
    print(f"Checkpoint dir: {MODEL_DIR}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(GPT2_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Datasets
    train_ds = MoleculeCaptionDataset(TRAIN_GRAPHS)
    train_collate = make_caption_collate_fn(tokenizer, max_length=MAX_LEN)
    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=train_collate,
        num_workers=0,
    )

    val_dl = None
    if os.path.exists(VAL_GRAPHS):
        val_ds = MoleculeCaptionDataset(VAL_GRAPHS)
        val_collate = make_caption_collate_fn(tokenizer, max_length=MAX_LEN)
        val_dl = DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=val_collate,
            num_workers=0,
        )

    # Vocab sizes for graph feature embeddings
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    total_steps = EPOCHS * len(train_dl)
    warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ===== RESUME =====
    start_epoch = 1
    best_val_loss = float("inf")

    if os.path.exists(LAST_CKPT_PATH):
        last_epoch, best_val_loss = load_ckpt(LAST_CKPT_PATH, model, optimizer, scheduler, DEVICE)
        start_epoch = last_epoch + 1
        print(f"🔁 Resuming from epoch {start_epoch} (best_val_loss={best_val_loss:.4f})")
    else:
        print("🆕 No previous checkpoint found. Starting from scratch.")

    # ===== TRAIN LOOP =====
    for ep in range(start_epoch, EPOCHS + 1):
        model.train()
        pbar = tqdm(train_dl, desc=f"Epoch {ep}/{EPOCHS}")
        tr_loss_sum, tr_count = 0.0, 0

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
            tr_loss_sum += loss.item() * bs
            tr_count += bs
            pbar.set_postfix(loss=float(loss.item()))

        train_loss = tr_loss_sum / max(1, tr_count)
        print(f"\nEpoch {ep}: train_loss={train_loss:.4f}")

        # ===== VALID (teacher forcing) =====
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

            # samples
            samples = quick_samples(model, tokenizer, val_dl, DEVICE, n_batches=2)
            print("\n[Samples]")
            for s in samples:
                print("-", s[:200])

        metric = val_loss if val_loss is not None else train_loss

        # ===== SAVE LAST (always) =====
        save_ckpt(
            LAST_CKPT_PATH,
            model,
            optimizer,
            scheduler,
            epoch=ep,
            best_val_loss=best_val_loss,
            extra={"tokenizer_name": GPT2_NAME, "max_len": MAX_LEN},
        )
        print(f"💾 Saved LAST checkpoint: {LAST_CKPT_PATH}")

        # ===== SAVE BEST (if improved) =====
        if metric < best_val_loss:
            best_val_loss = metric
            save_ckpt(
                BEST_CKPT_PATH,
                model,
                optimizer,
                scheduler,
                epoch=ep,
                best_val_loss=best_val_loss,
                extra={"tokenizer_name": GPT2_NAME, "max_len": MAX_LEN},
            )
            print(f"🏆 Saved BEST checkpoint: {BEST_CKPT_PATH} (best_val_loss={best_val_loss:.4f})")

    print("\nDone.")


if __name__ == "__main__":
    main()
