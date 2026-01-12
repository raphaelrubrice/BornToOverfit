#!/usr/bin/env python3
"""
retrieval_answer_new.py - Robust retrieval with dynamic imports + optional train+val corpus + chunked topK.

Example:
python data_baseline/retrieval_answer_new.py \
  --code train_gps_chembert_mse \
  --model model_gps_chembert_mse.pt \
  --data_dir data_baseline/data \
  --results_dir results \
  --train_emb data_baseline/data/train_embeddings.csv \
  --val_emb   data_baseline/data/validation_embeddings.csv \
  --use_trainval_corpus_for_test 1 \
  --topk 10
"""

import os
import sys
import argparse
import importlib
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_utils import (
    load_id2emb, load_descriptions_from_graphs, PreprocessedGraphDataset, collate_fn
)

# ----------------------------
# Utilities
# ----------------------------
def _to_device_f32(x: torch.Tensor, device: str) -> torch.Tensor:
    return x.to(device=device, dtype=torch.float32, non_blocking=True)

def _stack_embs(ids, emb_dict, device):
    # keep only ids that exist in emb_dict
    kept = [i for i in ids if i in emb_dict]
    embs = torch.stack([emb_dict[i] for i in kept], dim=0)
    embs = _to_device_f32(embs, device)
    embs = F.normalize(embs, dim=-1)
    return kept, embs

def build_corpus(graph_pkl_paths, emb_dicts, device):
    """
    Build a retrieval corpus from one or more (graphs.pkl, embeddings_dict).
    Returns:
      corpus_ids: list[str]
      corpus_embs: torch.Tensor [N, D] normalized
      corpus_id2desc: dict[str, str]
    """
    corpus_id2desc = {}
    merged_emb = {}

    for gpath, edict in zip(graph_pkl_paths, emb_dicts):
        if gpath is None or edict is None:
            continue
        gpath = str(gpath)
        if not os.path.exists(gpath):
            raise FileNotFoundError(f"Missing graphs file: {gpath}")

        # merge descriptions
        corpus_id2desc.update(load_descriptions_from_graphs(gpath))
        # merge embeddings
        merged_emb.update(edict)

    # keep only IDs present in BOTH desc + embeddings
    common_ids = [i for i in merged_emb.keys() if i in corpus_id2desc]
    if len(common_ids) == 0:
        raise RuntimeError("Corpus is empty after intersection (embeddings ∩ descriptions).")

    corpus_ids, corpus_embs = _stack_embs(common_ids, merged_emb, device)
    return corpus_ids, corpus_embs, corpus_id2desc


@torch.no_grad()
def retrieve_descriptions(
    model,
    corpus_ids,
    corpus_embs,        # [N, D] normalized
    corpus_id2desc,
    query_graphs_pkl,   # test/val graphs
    device,
    output_csv,
    batch_size=64,
    topk=1,
    chunk_size=4096,    # chunk queries to avoid huge similarity matrices
):
    """
    Retrieval:
      - Encode query graphs with model
      - Cosine sim vs corpus_embs
      - Choose best (or topk) corpus item per query
      - Output CSV: ID, description

    Note: topk>1 does minimal rerank = choose highest cosine (same as argmax) but computed safely in chunks.
          (Hook ready if you want more advanced rerank later.)
    """
    query_graphs_pkl = str(query_graphs_pkl)
    assert os.path.exists(query_graphs_pkl), f"Missing query graphs: {query_graphs_pkl}"

    # Load query dataset
    qds = PreprocessedGraphDataset(query_graphs_pkl)
    qdl = DataLoader(qds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"Corpus size: {len(corpus_ids)} | Query size: {len(qds)} | topk={topk}")
    D = corpus_embs.size(1)

    all_best_idx = []
    all_query_ids = []

    # We'll process in mini-batches from DataLoader and compute topk vs corpus
    # in a memory-safe way.
    seen = 0
    for graphs in qdl:
        graphs = graphs.to(device)
        q_emb = model(graphs)
        q_emb = _to_device_f32(q_emb, device)
        q_emb = F.normalize(q_emb, dim=-1)  # IMPORTANT

        bsz = q_emb.size(0)
        # collect IDs in the same order
        all_query_ids.extend(qds.ids[seen:seen+bsz])
        seen += bsz

        # compute similarities: [bsz, N] = q_emb @ corpus_embs.T
        # if corpus huge, still OK because bsz small; but we can also chunk corpus if needed.
        sims = q_emb @ corpus_embs.t()

        # topk indices
        k = min(topk, sims.size(1))
        top_vals, top_idx = torch.topk(sims, k=k, dim=1, largest=True, sorted=True)

        # minimal rerank: pick the best (top-1)
        best = top_idx[:, 0].detach().cpu()
        all_best_idx.append(best)

    all_best_idx = torch.cat(all_best_idx, dim=0).tolist()
    assert len(all_best_idx) == len(all_query_ids), "Mismatch query ids vs retrieved indices."

    # Build output
    rows = []
    for qid, idx in zip(all_query_ids, all_best_idx):
        rid = corpus_ids[idx]
        desc = corpus_id2desc.get(rid, "No description found")
        rows.append({"ID": qid, "description": desc})

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved {len(df)} rows to: {output_csv}")

    # Quick sanity prints
    for i in range(min(3, len(df))):
        print(f"\n[Query {df.loc[i,'ID']}] -> retrieved: {rows[i]['description'][:120]}...")

    return df


def main():
    parser = argparse.ArgumentParser(description="Retrieval generation (robust)")
    parser.add_argument("--code", type=str, required=True,
                        help="Python module name containing MolGNN (e.g., train_gps_chembert_mse)")
    parser.add_argument("--model", type=str, required=True,
                        help="Checkpoint filename (.pt) located in results_dir or data_dir")
    parser.add_argument("--data_dir", type=str, default="data_baseline/data",
                        help="Folder containing train/validation/test_graphs.pkl")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Folder containing checkpoints")
    parser.add_argument("--train_emb", type=str, required=True,
                        help="Path to train embeddings CSV/PT (ID, embedding)")
    parser.add_argument("--val_emb", type=str, default=None,
                        help="Path to val embeddings CSV/PT (needed only if you want train+val corpus)")
    parser.add_argument("--use_trainval_corpus_for_test", type=int, default=0,
                        help="If 1: build corpus from train+val for TEST retrieval (recommended for Kaggle)")
    parser.add_argument("--topk", type=int, default=1, help="Retrieve top-k then pick best (default 1)")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    base_path = Path(args.data_dir)
    results_path = Path(args.results_dir)

    # Ensure local imports work
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    # Dynamic import
    print(f"🔄 Import module: {args.code}")
    try:
        source_module = importlib.import_module(args.code)
    except Exception as e:
        print(f"❌ Failed to import '{args.code}': {e}")
        raise

    if not hasattr(source_module, "MolGNN"):
        raise AttributeError(f"Module '{args.code}' must expose a MolGNN class.")

    MolGNN = source_module.MolGNN
    DEVICE = getattr(source_module, "DEVICE", ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {DEVICE}")

    # Graph paths
    train_graphs_path = base_path / "train_graphs.pkl"
    val_graphs_path   = base_path / "validation_graphs.pkl"
    test_graphs_path  = base_path / "test_graphs.pkl"

    for p in [train_graphs_path, val_graphs_path, test_graphs_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing graphs file: {p}")

    # Locate model checkpoint
    model_file_path = results_path / args.model
    if not model_file_path.exists():
        # fallback to data_dir
        alt = base_path / args.model
        if alt.exists():
            model_file_path = alt
        else:
            raise FileNotFoundError(f"Checkpoint not found in results_dir or data_dir: {args.model}")

    # Load embeddings dictionaries
    print(f"Loading train embeddings: {args.train_emb}")
    train_emb = load_id2emb(args.train_emb)

    val_emb = None
    if args.use_trainval_corpus_for_test:
        if args.val_emb is None:
            raise ValueError("--val_emb is required when --use_trainval_corpus_for_test=1")
        print(f"Loading val embeddings: {args.val_emb}")
        val_emb = load_id2emb(args.val_emb)

    emb_dim = len(next(iter(train_emb.values())))
    print(f"Embedding dim: {emb_dim}")

    # Init model + load weights
    print(f"Init MolGNN(out_dim={emb_dim}) and load weights: {model_file_path}")
    model = MolGNN(out_dim=emb_dim).to(DEVICE)
    state = torch.load(model_file_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    # ----------------------------
    # 1) VALID retrieval (always corpus = train ONLY for fair eval)
    # ----------------------------
    corpus_ids_val, corpus_embs_val, corpus_id2desc_val = build_corpus(
        graph_pkl_paths=[train_graphs_path],
        emb_dicts=[train_emb],
        device=DEVICE
    )

    out_val_csv = results_path / "val_retrieved_descriptions.csv"
    retrieve_descriptions(
        model=model,
        corpus_ids=corpus_ids_val,
        corpus_embs=corpus_embs_val,
        corpus_id2desc=corpus_id2desc_val,
        query_graphs_pkl=val_graphs_path,
        device=DEVICE,
        output_csv=str(out_val_csv),
        batch_size=args.batch_size,
        topk=args.topk,
    )

    # ----------------------------
    # 2) TEST retrieval
    #     - if asked: corpus = train+val (better for Kaggle)
    #     - else: corpus = train only
    # ----------------------------
    if args.use_trainval_corpus_for_test:
        print("✅ Using TRAIN+VAL corpus for TEST retrieval.")
        corpus_ids_test, corpus_embs_test, corpus_id2desc_test = build_corpus(
            graph_pkl_paths=[train_graphs_path, val_graphs_path],
            emb_dicts=[train_emb, val_emb],
            device=DEVICE
        )
    else:
        print("Using TRAIN-only corpus for TEST retrieval.")
        corpus_ids_test, corpus_embs_test, corpus_id2desc_test = corpus_ids_val, corpus_embs_val, corpus_id2desc_val

    out_test_csv = results_path / "test_retrieved_descriptions.csv"
    retrieve_descriptions(
        model=model,
        corpus_ids=corpus_ids_test,
        corpus_embs=corpus_embs_test,
        corpus_id2desc=corpus_id2desc_test,
        query_graphs_pkl=test_graphs_path,
        device=DEVICE,
        output_csv=str(out_test_csv),
        batch_size=args.batch_size,
        topk=args.topk,
    )

    print("\n✅ Done.")
    print(f"VAL CSV : {out_val_csv}")
    print(f"TEST CSV: {out_test_csv}")


if __name__ == "__main__":
    main()
