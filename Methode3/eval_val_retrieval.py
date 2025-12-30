import os
import argparse
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from data_utils import load_id2emb, load_descriptions_from_graphs, PreprocessedGraphDataset, collate_fn
from models_gps import MolGPS


@torch.no_grad()
def retrieve_on_validation(model, train_graphs, val_graphs, train_emb_csv, device):
    train_id2desc = load_descriptions_from_graphs(train_graphs)
    val_id2desc = load_descriptions_from_graphs(val_graphs)

    train_emb_dict = load_id2emb(train_emb_csv)
    train_ids = list(train_emb_dict.keys())
    train_embs = torch.stack([train_emb_dict[i] for i in train_ids]).to(device)
    train_embs = F.normalize(train_embs, dim=-1)

    val_ds = PreprocessedGraphDataset(val_graphs)
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    preds, refs = [], []
    val_ids_ordered = []

    for graphs in val_dl:
        graphs = graphs.to(device)
        mol_emb = model(graphs)
        sims = mol_emb @ train_embs.t()
        idx = sims.argmax(dim=-1).cpu()

        bs = graphs.num_graphs
        start = len(val_ids_ordered)
        batch_ids = val_ds.ids[start:start + bs]
        val_ids_ordered.extend(batch_ids)

        for j, vid in enumerate(batch_ids):
            retrieved_train_id = train_ids[idx[j].item()]
            pred = train_id2desc[retrieved_train_id]
            ref = val_id2desc[vid]
            preds.append(pred)
            refs.append(ref)

    return preds, refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_graphs", type=str, default="data/train_graphs.pkl")
    ap.add_argument("--val_graphs", type=str, default="data/validation_graphs.pkl")
    ap.add_argument("--train_emb_csv", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)

    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--num_layers", type=int, default=4)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--attn_type", type=str, default="multihead")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    emb_dim = len(next(iter(load_id2emb(args.train_emb_csv).values())))
    model = MolGPS(
        hidden_dim=args.hidden_dim,
        out_dim=emb_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        attn_type=args.attn_type,
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    preds, refs = retrieve_on_validation(
        model=model,
        train_graphs=args.train_graphs,
        val_graphs=args.val_graphs,
        train_emb_csv=args.train_emb_csv,
        device=device,
    )

    smoothie = SmoothingFunction().method4
    bleu = corpus_bleu(
        [[r.split()] for r in refs],
        [p.split() for p in preds],
        smoothing_function=smoothie
    )
    print(f"Validation BLEU (token split by space): {bleu:.4f}")


if __name__ == "__main__":
    main()
