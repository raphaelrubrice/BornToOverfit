#!/usr/bin/env python3
"""
eval_val_retrieval.py

1) Encode validation molecules with trained GNN.
2) Retrieve nearest train text embedding (cosine via dot product on normalized vectors).
3) Get retrieved train description.
4) Compare retrieved description vs ground-truth validation description:
   - BLEU-4 (corpus + mean sentence BLEU4)
   - (Optional) BERTScore if bert-score installed

Example:
python eval_val_retrieval.py \
  --data_dir /content/BornToOverfit/Methode3/data \
  --train_emb_csv /content/BornToOverfit/Methode3/data/train_embeddings.xxx.csv \
  --ckpt /content/BornToOverfit/Methode3/results/checkpoints/model_xxx.pt
"""

import os
import re
import argparse
import pickle
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from data_utils import load_id2emb, PreprocessedGraphDataset, collate_fn


# -------------------------
# Utilities
# -------------------------
def simple_tokenize(s: str) -> List[str]:
    s = (s or "").lower().strip()
    # keep words/numbers, remove weird punctuation
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.split()


def load_id2desc_from_pkl(pkl_path: str) -> Dict[str, str]:
    with open(pkl_path, "rb") as f:
        graphs = pickle.load(f)
    return {str(g.id): (g.description or "") for g in graphs}


def try_bertscore(preds: List[str], refs: List[str], model_type: str, device: str) -> Optional[Dict[str, float]]:
    """
    Returns dict with P/R/F1 (mean), or None if bert_score not installed.
    """
    try:
        from bert_score import score as bertscore
    except Exception:
        return None

    P, R, F1 = bertscore(
        cands=preds,
        refs=refs,
        lang="en",
        model_type=model_type,
        device=device,
        rescale_with_baseline=True,
        verbose=False,
    )
    return {
        "BERTScore_P": float(P.mean().item()),
        "BERTScore_R": float(R.mean().item()),
        "BERTScore_F1": float(F1.mean().item()),
    }


@torch.no_grad()
def retrieve_val_descriptions(
    model,
    train_emb_csv: str,
    train_graphs: str,
    val_graphs: str,
    device: str,
    batch_size: int = 64,
) -> Tuple[List[str], List[str]]:
    """
    Returns (pred_texts, ref_texts) aligned in the same validation ID order.
    """
    # Train text embeddings
    train_emb = load_id2emb(train_emb_csv)
    train_ids = list(train_emb.keys())
    train_mat = torch.stack([train_emb[i] for i in train_ids]).to(device)
    train_mat = F.normalize(train_mat, dim=-1)

    # Descriptions
    train_id2desc = load_id2desc_from_pkl(train_graphs)
    val_id2desc = load_id2desc_from_pkl(val_graphs)

    # Encode val molecules
    val_ds = PreprocessedGraphDataset(val_graphs)  # graphs only
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model.eval()
    all_val_emb = []
    val_ids_ordered = []

    seen = 0
    for batch_graphs in val_dl:
        batch_graphs = batch_graphs.to(device)
        mol_vec = model(batch_graphs)
        mol_vec = F.normalize(mol_vec, dim=-1)
        all_val_emb.append(mol_vec)

        bs = batch_graphs.num_graphs
        val_ids_ordered.extend(val_ds.ids[seen:seen + bs])
        seen += bs

    val_mat = torch.cat(all_val_emb, dim=0)  # (Nval, D)

    # Retrieve nearest train text embedding
    sims = val_mat @ train_mat.t()
    best_idx = sims.argmax(dim=-1).cpu().tolist()

    preds, refs = [], []
    for i, vid in enumerate(val_ids_ordered):
        tid = train_ids[best_idx[i]]
        pred = train_id2desc.get(tid, "")
        ref = val_id2desc.get(str(vid), "")
        preds.append(pred)
        refs.append(ref)

    return preds, refs


def main():
    ap = argparse.ArgumentParser()

    # New: accept data_dir like your benchmark script passes
    ap.add_argument("--data_dir", type=str, default=None,
                    help="If provided, will resolve train_graphs/val_graphs from it when not explicitly given.")

    ap.add_argument("--train_graphs", type=str, default=None)
    ap.add_argument("--val_graphs", type=str, default=None)

    ap.add_argument("--train_emb_csv", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)

    ap.add_argument("--batch_size", type=int, default=64)

    # Optional BERTScore
    ap.add_argument("--do_bertscore", action="store_true")
    ap.add_argument("--bertscore_model", type=str, default="roberta-large")

    # For device
    ap.add_argument("--device", type=str, default=None)

    args = ap.parse_args()

    # Resolve paths
    if args.data_dir is not None:
        if args.train_graphs is None:
            args.train_graphs = os.path.join(args.data_dir, "train_graphs.pkl")
        if args.val_graphs is None:
            args.val_graphs = os.path.join(args.data_dir, "validation_graphs.pkl")

    if args.train_graphs is None or args.val_graphs is None:
        raise ValueError("You must provide either --data_dir or both --train_graphs and --val_graphs")

    if not os.path.exists(args.train_graphs):
        raise FileNotFoundError(args.train_graphs)
    if not os.path.exists(args.val_graphs):
        raise FileNotFoundError(args.val_graphs)
    if not os.path.exists(args.train_emb_csv):
        raise FileNotFoundError(args.train_emb_csv)
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(args.ckpt)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Import model
    MolGNN = None
    try:
        from train_gcn_v3_gps import MolGNN as MolGNN  # your GPS training file
    except Exception:
        try:
            from train_gcn import MolGNN as MolGNN
        except Exception as e:
            raise ImportError("Could not import MolGNN from train_gcn_v3_gps.py or train_gcn.py") from e

    # Load embeddings to get out_dim
    train_emb = load_id2emb(args.train_emb_csv)
    emb_dim = len(next(iter(train_emb.values())))

    model = MolGNN(out_dim=emb_dim).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    preds, refs = retrieve_val_descriptions(
        model=model,
        train_emb_csv=args.train_emb_csv,
        train_graphs=args.train_graphs,
        val_graphs=args.val_graphs,
        device=device,
        batch_size=args.batch_size,
    )

    # BLEU4
    smoothie = SmoothingFunction().method1
    list_of_refs = [[simple_tokenize(r)] for r in refs]  # corpus_bleu format: list of list of refs
    hyps = [simple_tokenize(p) for p in preds]

    bleu4_corpus = corpus_bleu(
        list_of_refs,
        hyps,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothie,
    )

    # Mean sentence BLEU4 (often less stable but useful)
    # We'll approximate via corpus_bleu on singletons -> too slow; keep only corpus.
    results = {
        "BLEU4_corpus": float(bleu4_corpus),
        "n_val": len(refs),
    }

    # Optional BERTScore
    if args.do_bertscore:
        bs = try_bertscore(preds, refs, model_type=args.bertscore_model, device=device)
        if bs is None:
            print("bert-score not installed. Install with: pip install bert-score")
        else:
            results.update(bs)

    print("\n==== VALIDATION TEXT METRICS ====")
    for k, v in results.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
