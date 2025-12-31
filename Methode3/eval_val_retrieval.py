%%writefile /content/BornToOverfit/Methode3/eval_val_retrieval.py
import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from data_utils import load_id2emb, load_descriptions_from_graphs, PreprocessedGraphDataset, collate_fn
from train_gcn_v3_gps import MolGNN, HIDDEN_DIM, NUM_LAYERS, NUM_HEADS, DROPOUT, DEVICE


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--train_emb", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    args = ap.parse_args()

    train_graphs = os.path.join(args.data_dir, "train_graphs.pkl")
    val_graphs = os.path.join(args.data_dir, "validation_graphs.pkl")

    train_id2desc = load_descriptions_from_graphs(train_graphs)
    val_id2desc = load_descriptions_from_graphs(val_graphs)

    train_emb = load_id2emb(args.train_emb)
    train_ids = list(train_emb.keys())
    train_mat = torch.stack([train_emb[i] for i in train_ids]).to(DEVICE)
    train_mat = F.normalize(train_mat, dim=-1)

    emb_dim = len(next(iter(train_emb.values())))
    model = MolGNN(hidden_dim=HIDDEN_DIM, out_dim=emb_dim, num_layers=NUM_LAYERS, num_heads=NUM_HEADS, dropout=DROPOUT).to(DEVICE)
    model.load_state_dict(torch.load(args.ckpt, map_location=DEVICE))
    model.eval()

    val_ds = PreprocessedGraphDataset(val_graphs)
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    preds, refs = [], []
    ordered_ids = []

    for graphs in val_dl:
        graphs = graphs.to(DEVICE)
        mol = model(graphs)
        sims = mol @ train_mat.t()
        nn_idx = sims.argmax(dim=-1).cpu().tolist()

        bs = graphs.num_graphs
        start = len(ordered_ids)
        batch_ids = val_ds.ids[start:start + bs]
        ordered_ids.extend(batch_ids)

        for j, vid in enumerate(batch_ids):
            rid = train_ids[nn_idx[j]]
            preds.append(train_id2desc[rid])
            refs.append(val_id2desc[vid])

    smoothie = SmoothingFunction().method4
    bleu4 = corpus_bleu(
        [[r.split()] for r in refs],
        [p.split() for p in preds],
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothie
    )
    print(f"{bleu4:.6f}")


if __name__ == "__main__":
    main()
