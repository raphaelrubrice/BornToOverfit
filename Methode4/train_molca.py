# train_molca.py
import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup

from data_utils import GraphCaptionDataset, collate_caption_fn
from model_molca import GraphEncoder, QFormerLite

# =====================
# PATHS
# =====================
TRAIN_GRAPHS = "data/train_graphs.pkl"
VAL_GRAPHS   = "data/validation_graphs.pkl"
TEST_GRAPHS  = "data/test_graphs.pkl"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================
# TRAINING HYPERPARAMS (T4-friendly defaults)
# =====================
EPOCHS = 15
BATCH_SIZE = 8                # try 8 on T4, if OOM -> 4
GRAD_ACCUM_STEPS = 2          # effective batch = BATCH_SIZE * GRAD_ACCUM_STEPS
LR_BRIDGE = 3e-4              # graph_enc + qformer + proj
LR_T5 = 8e-5                  # T5 params (lower)
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
MAX_LEN = 160                 # tokenizer max_len for labels
CLIP_NORM = 1.0
LABEL_SMOOTHING = 0.1         # set 0.0 to disable

EARLY_STOP_PATIENCE = 3       # stop if no improvement after N epochs

# =====================
# MODEL
# =====================
class MolCAStyleCaptioner(nn.Module):
    def __init__(self, t5_name="google/flan-t5-base", hidden=256, num_queries=16, freeze_t5_encoder=True):
        super().__init__()
        self.graph_enc = GraphEncoder(hidden=hidden, layers=4)
        self.qformer = QFormerLite(hidden=hidden, num_queries=num_queries, layers=2)

        self.t5 = AutoModelForSeq2SeqLM.from_pretrained(t5_name)
        d_model = self.t5.config.d_model
        self.proj = nn.Linear(hidden, d_model)

        if freeze_t5_encoder:
            for p in self.t5.encoder.parameters():
                p.requires_grad = False

    def forward(self, batch_graph, labels):
        node_emb, batch_vec = self.graph_enc(batch_graph)
        q = self.qformer(node_emb, batch_vec)                # [B,Q,H]
        enc = self.proj(q)                                   # [B,Q,d_model]
        attn_mask = torch.ones(enc.shape[:2], device=enc.device, dtype=torch.long)

        # label smoothing if supported by your transformers version
        try:
            out = self.t5(
                inputs_embeds=enc,
                attention_mask=attn_mask,
                labels=labels,
                label_smoothing_factor=LABEL_SMOOTHING if LABEL_SMOOTHING > 0 else 0.0,
            )
        except TypeError:
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
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_caption_fn(b, tokenizer, max_len=MAX_LEN),
        num_workers=2,
        pin_memory=(DEVICE == "cuda"),
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: collate_caption_fn(b, tokenizer, max_len=MAX_LEN),
        num_workers=2,
        pin_memory=(DEVICE == "cuda"),
    )

    model = MolCAStyleCaptioner(t5_name=t5_name, hidden=256, num_queries=16, freeze_t5_encoder=True).to(DEVICE)

    # Param groups: bridge higher LR, T5 lower LR
    bridge_params = list(model.graph_enc.parameters()) + list(model.qformer.parameters()) + list(model.proj.parameters())
    t5_params = [p for p in model.t5.parameters() if p.requires_grad]

    opt = torch.optim.AdamW(
        [
            {"params": bridge_params, "lr": LR_BRIDGE},
            {"params": t5_params, "lr": LR_T5},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    total_updates_per_epoch = math.ceil(len(train_dl) / GRAD_ACCUM_STEPS)
    total_updates = EPOCHS * total_updates_per_epoch
    warmup_steps = int(WARMUP_RATIO * total_updates)

    sched = get_linear_schedule_with_warmup(opt, warmup_steps, total_updates)

    use_amp = (DEVICE == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best = 1e9
    bad_epochs = 0

    for ep in range(EPOCHS):
        model.train()
        opt.zero_grad(set_to_none=True)

        pbar = tqdm(train_dl, desc=f"Epoch {ep+1}/{EPOCHS}", leave=True)
        running = 0.0
        seen = 0

        for step, (graphs, tok, labels) in enumerate(pbar, start=1):
            graphs = graphs.to(DEVICE)
            labels = labels.to(DEVICE)

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = model(graphs, labels)
                loss = loss / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            bs = graphs.num_graphs
            running += loss.item() * GRAD_ACCUM_STEPS * bs
            seen += bs

            if step % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()

            if step % 20 == 0:
                pbar.set_postfix({"train_loss": f"{running/max(1,seen):.4f}", "lr_bridge": opt.param_groups[0]["lr"]})

        vloss = eval_loss(model, val_dl, DEVICE)
        print(f"\nEpoch {ep+1}: val_loss={vloss:.4f}")

        if vloss < best:
            best = vloss
            bad_epochs = 0
            torch.save(model.state_dict(), "molca_t5_best.pt")
            print("  -> saved molca_t5_best.pt")
        else:
            bad_epochs += 1
            print(f"  -> no improvement ({bad_epochs}/{EARLY_STOP_PATIENCE})")
            if bad_epochs >= EARLY_STOP_PATIENCE:
                print("Early stopping.")
                break

if __name__ == "__main__":
    main()
