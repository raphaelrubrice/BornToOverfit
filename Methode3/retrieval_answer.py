import os
import argparse
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_utils import (
    load_id2emb, load_descriptions_from_graphs, PreprocessedGraphDataset, collate_fn
)
from models_gps import MolGPS


@torch.no_grad()
def retrieve_descriptions(model, train_graphs_pkl, test_graphs_pkl, train_emb_csv, device, output_csv):
    train_id2desc = load_descriptions_from_graphs(train_graphs_pkl)

    train_emb_dict = load_id2emb(train_emb_csv)
    train_ids = list(train_emb_dict.keys())
    train_embs = torch.stack([train_emb_dict[i] for i in train_ids]).to(device)
    train_embs = F.normalize(train_embs, dim=-1)

    test_ds = PreprocessedGraphDataset(test_graphs_pkl)
    test_dl = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    test_mol_embs = []
    test_ids_ordered = []
    for graphs in test_dl:
        graphs = graphs.to(device)
        mol_emb = model(graphs)
        test_mol_embs.append(mol_emb)

        bs = graphs.num_graphs
        start = len(test_ids_ordered)
        test_ids_ordered.extend(test_ds.ids[start:start + bs])

    test_mol_embs = torch.cat(test_mol_embs, dim=0)
    sims = test_mol_embs @ train_embs.t()
    idx = sims.argmax(dim=-1).cpu()

    results = []
    for i, test_id in enumerate(test_ids_ordered):
        train_idx = idx[i].item()
        retrieved_train_id = train_ids[train_idx]
        retrieved_desc = train_id2desc[retrieved_train_id]
        results.append({"ID": test_id, "description": retrieved_desc})

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved: {output_csv} ({len(df)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_graphs", type=str, default="data/train_graphs.pkl")
    ap.add_argument("--test_graphs", type=str, default="data/test_graphs.pkl")
    ap.add_argument("--train_emb_csv", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--output_csv", type=str, default="test_retrieved_descriptions.csv")

    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--num_layers", type=int, default=4)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--attn_type", type=str, default="multihead")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

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

    retrieve_descriptions(
        model=model,
        train_graphs_pkl=args.train_graphs,
        test_graphs_pkl=args.test_graphs,
        train_emb_csv=args.train_emb_csv,
        device=device,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
