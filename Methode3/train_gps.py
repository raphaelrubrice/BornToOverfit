"""
train_gps.py - Train GPS graph encoder to align graph embeddings with text embeddings.

Usage:
python train_gps.py \
  --train_graphs data/train_graphs.pkl \
  --val_graphs data/validation_graphs.pkl \
  --train_emb_csv data/train_embeddings.bert-base-uncased.cls.csv \
  --val_emb_csv data/validation_embeddings.bert-base-uncased.cls.csv \
  --out_ckpt checkpoints/model_gps_bert-base-uncased_cls.pt
"""

import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_utils import load_id2emb, PreprocessedGraphDataset, collate_fn
from models_gps import MolGPS


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, total = 0.0, 0
    for graphs, text_emb in loader:
        graphs = graphs.to(device)
        text_emb = text_emb.to(device)

        mol_vec = model(graphs)
        txt_vec = F.normalize(text_emb, dim=-1)

        loss = F.mse_loss(mol_vec, txt_vec)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bs = graphs.num_graphs
        total_loss += loss.item() * bs
        total += bs

    return total_loss / max(1, total)


@torch.no_grad()
def eval_retrieval(data_path, emb_dict, model, device):
    model.eval()
    ds = PreprocessedGraphDataset(data_path, emb_dict)
    dl = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    all_mol, all_txt = [], []
    for graphs, text_emb in dl:
        graphs = graphs.to(device)
        text_emb = text_emb.to(device)
        all_mol.append(model(graphs))
        all_txt.append(F.normalize(text_emb, dim=-1))

    all_mol = torch.cat(all_mol, dim=0)
    all_txt = torch.cat(all_txt, dim=0)

    sims = all_txt @ all_mol.t()
    ranks = sims.argsort(dim=-1, descending=True)

    N = all_txt.size(0)
    correct = torch.arange(N, device=sims.device)
    pos = (ranks == correct.unsqueeze(1)).nonzero()[:, 1] + 1

    mrr = (1.0 / pos.float()).mean().item()
    results = {"MRR": mrr}
    for k in (1, 5, 10):
        results[f"R@{k}"] = (pos <= k).float().mean().item()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_graphs", type=str, required=True)
    ap.add_argument("--val_graphs", type=str, default=None)
    ap.add_argument("--train_emb_csv", type=str, required=True)
    ap.add_argument("--val_emb_csv", type=str, default=None)
    ap.add_argument("--out_ckpt", type=str, required=True)

    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--num_layers", type=int, default=4)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--attn_type", type=str, default="multihead", choices=["multihead", "performer"])

    ap.add_argument("--patience", type=int, default=5)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(args.out_ckpt) or ".", exist_ok=True)

    print("=" * 70)
    print("TRAIN GPS")
    print("=" * 70)
    print("Device:", device)
    print("Train emb:", args.train_emb_csv)
    print("Val emb:", args.val_emb_csv)

    train_emb = load_id2emb(args.train_emb_csv)
    val_emb = load_id2emb(args.val_emb_csv) if args.val_emb_csv and os.path.exists(args.val_emb_csv) else None

    emb_dim = len(next(iter(train_emb.values())))

    train_ds = PreprocessedGraphDataset(args.train_graphs, train_emb)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    model = MolGPS(
        hidden_dim=args.hidden_dim,
        out_dim=emb_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        attn_type=args.attn_type,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_mrr = -1.0
    patience_counter = 0

    for ep in range(args.epochs):
        loss = train_epoch(model, train_dl, optimizer, device)
        scheduler.step()

        val_scores = {}
        if val_emb is not None and args.val_graphs is not None and os.path.exists(args.val_graphs):
            val_scores = eval_retrieval(args.val_graphs, val_emb, model, device)

        print(f"Epoch {ep+1}/{args.epochs} | loss={loss:.4f} | {val_scores}")

        current_mrr = val_scores.get("MRR", None)
        if current_mrr is not None:
            if current_mrr > best_mrr:
                best_mrr = current_mrr
                patience_counter = 0
                torch.save(model.state_dict(), args.out_ckpt)
                print(f"  ✅ New best MRR: {best_mrr:.4f} -> saved {args.out_ckpt}")
            else:
                patience_counter += 1
                print(f"  → No improvement ({patience_counter}/{args.patience})")
                if patience_counter >= args.patience:
                    print(f"\nEarly stopping at epoch {ep+1}. Best MRR={best_mrr:.4f}")
                    break
        else:
            # No val => save last
            torch.save(model.state_dict(), args.out_ckpt)

    print("\nDone.")


if __name__ == "__main__":
    main()
