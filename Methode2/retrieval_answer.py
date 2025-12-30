import os
import json
from pathlib import Path
from argparse import ArgumentParser
from typing import List, Dict

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sacrebleu
from bert_score import score as bertscore

try:
    from data_utils import (
        load_id2emb, load_descriptions_from_graphs, PreprocessedGraphDataset, collate_fn
    )
except Exception:
    from .data_utils import (
        load_id2emb, load_descriptions_from_graphs, PreprocessedGraphDataset, collate_fn
    )


def _safe_text(x) -> str:
    if x is None:
        return ""
    if not isinstance(x, str):
        return str(x)
    return x


def compute_bertscore_roberta_base(
    preds: List[str],
    refs: List[str],
    device: str,
    batch_size: int = 32,
    idf: bool = False,
) -> Dict[str, float]:
    P, R, F1 = bertscore(
        cands=preds,
        refs=refs,
        model_type="roberta-base",
        lang="en",
        device=device,
        batch_size=batch_size,
        idf=idf,
        verbose=False,
        rescale_with_baseline=False,
    )
    return {
        "bertscore_precision": float(P.mean().item()),
        "bertscore_recall": float(R.mean().item()),
        "bertscore_f1": float(F1.mean().item()),
    }


def evaluate_retrieval_text_metrics(
    results_df: pd.DataFrame,
    id2ref: Dict[str, str],
    device: str,
    save_path: str = None,
) -> Dict[str, float]:
    preds, refs = [], []
    missing = 0
    for _, row in results_df.iterrows():
        _id = str(row["ID"])
        pred = _safe_text(row["description"])
        ref = id2ref.get(_id, None)
        if ref is None:
            missing += 1
            ref = ""
        preds.append(pred)
        refs.append(_safe_text(ref))

    bleu4 = sacrebleu.corpus_bleu(preds, [refs], lowercase=True).score  # [0,100]
    bert_stats = compute_bertscore_roberta_base(preds, refs, device=device)
    bert_f1 = bert_stats["bertscore_f1"]  # [0,1]
    final_proxy = 0.5 * (bleu4 / 100.0) + 0.5 * bert_f1

    metrics = {
        "n_samples": int(len(preds)),
        "n_missing_refs": int(missing),
        "bleu4": float(bleu4),
        "bertscore_f1": float(bert_f1),
        "final_proxy": float(final_proxy),
        **bert_stats,
    }

    print("\n" + "=" * 80)
    print("Validation Metrics (retrieved caption vs ground-truth)")
    print(f"Samples: {metrics['n_samples']} | Missing refs: {metrics['n_missing_refs']}")
    print(f"BLEU-4:       {metrics['bleu4']:.4f} (0-100)")
    print(f"BERTScore F1: {metrics['bertscore_f1']:.4f} (0-1)")
    print(f"Final Proxy:  {metrics['final_proxy']:.4f}")
    print("=" * 80 + "\n")

    if save_path is not None:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"Saved metrics to: {save_path}")

    return metrics


def load_graph_model(graph_model: str, ckpt_path: str, device: str, out_dim_expected: int):
    graph_model = graph_model.lower().strip()

    if graph_model == "gps":
        # Uses Raphael/Camille GPS loader
        from train_gcn_v3_gps import load_molgnn_gps_from_checkpoint
        model = load_molgnn_gps_from_checkpoint(ckpt_path, device=device)
        return model

    elif graph_model == "gcn":
        from train_gcn import MolGNN
        model = MolGNN(out_dim=out_dim_expected).to(device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        return model

    else:
        raise ValueError(f"Unknown graph_model={graph_model}. Use 'gps' or 'gcn'.")


@torch.no_grad()
def retrieve_descriptions(
    model,
    train_graphs_pkl: str,
    query_graphs_pkl: str,
    train_emb_csv: str,
    output_csv: str,
    device: str,
    evaluate: bool = False,
):
    # --- Load train embeddings ---
    train_emb = load_id2emb(train_emb_csv)
    train_ids = list(train_emb.keys())
    train_embs = torch.stack([train_emb[i] for i in train_ids]).to(device)
    train_embs = F.normalize(train_embs, dim=-1)

    # --- Train id -> description ---
    train_id2desc = load_descriptions_from_graphs(train_graphs_pkl)

    # --- Encode query graphs ---
    query_ds = PreprocessedGraphDataset(query_graphs_pkl)
    query_dl = DataLoader(query_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    query_mol_embs = []
    query_ids = []
    for graphs in query_dl:
        graphs = graphs.to(device)
        emb = model(graphs)
        query_mol_embs.append(emb)

        bs = graphs.num_graphs
        start = len(query_ids)
        query_ids.extend(query_ds.ids[start:start + bs])

    query_mol_embs = torch.cat(query_mol_embs, dim=0)
    sims = query_mol_embs @ train_embs.t()
    idx = sims.argmax(dim=-1).cpu()

    # --- Build predictions ---
    rows = []
    for j, qid in enumerate(query_ids):
        tid = train_ids[int(idx[j].item())]
        rows.append({"ID": str(qid), "description": train_id2desc[tid]})

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Saved predictions to: {output_csv} ({len(df)} rows)")

    if evaluate:
        id2ref = load_descriptions_from_graphs(query_graphs_pkl)
        metrics_path = str(Path(output_csv).with_suffix(".metrics.json"))
        metrics = evaluate_retrieval_text_metrics(df, id2ref=id2ref, device=device, save_path=metrics_path)
        return df, metrics

    return df, None


def main():
    parser = ArgumentParser()

    # folders
    parser.add_argument("-f_data", default="data_baseline/data", type=str,
                        help="Folder containing graphs .pkl")
    parser.add_argument("-f", default="data_baseline/data", type=str,
                        help="Folder containing embeddings + checkpoints + outputs")
    parser.add_argument("--tag", default="run", type=str,
                        help="Tag to name outputs (e.g., scibert, pubmedbert, chemberta)")

    # embeddings
    parser.add_argument("--train_emb_csv", type=str, default=None,
                        help="Path to train embeddings CSV. If None, will use train_embeddings__{tag}.csv in -f folder.")
    parser.add_argument("--val_emb_csv", type=str, default=None,
                        help="Path to val embeddings CSV. If None, will use validation_embeddings__{tag}.csv in -f folder.")

    # graph model
    parser.add_argument("--graph_model", type=str, default="gps", choices=["gps", "gcn"])
    parser.add_argument("--graph_ckpt", type=str, default=None,
                        help="Path to graph model checkpoint (GPS checkpoint.pt or GCN state_dict). REQUIRED.")
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tag = args.tag

    file_path = Path(os.path.abspath(__file__))
    parent_folder = file_path.parent

    data_path = parent_folder.parent / args.f_data
    base_path = parent_folder.parent / args.f
    os.makedirs(str(base_path), exist_ok=True)

    TRAIN_GRAPHS = str(data_path / "train_graphs.pkl")
    VAL_GRAPHS   = str(data_path / "validation_graphs.pkl")
    TEST_GRAPHS  = str(data_path / "test_graphs.pkl")

    if args.train_emb_csv is None:
        train_emb_csv = str(base_path / f"train_embeddings__{tag}.csv")
    else:
        train_emb_csv = args.train_emb_csv

    if args.val_emb_csv is None:
        val_emb_csv = str(base_path / f"validation_embeddings__{tag}.csv")
    else:
        val_emb_csv = args.val_emb_csv

    if args.graph_ckpt is None:
        raise ValueError("You must provide --graph_ckpt (GPS or GCN checkpoint).")

    # --- Dimension check (critical) ---
    train_emb = load_id2emb(train_emb_csv)
    emb_dim = len(next(iter(train_emb.values())))
    print(f"Using text embeddings dim={emb_dim} from {train_emb_csv}")

    # If GPS: check model_config.json out_dim
    if args.graph_model == "gps":
        cfg_path = Path(args.graph_ckpt).parent / "model_config.json"
        if cfg_path.exists():
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
            out_dim = int(cfg.get("out_dim", emb_dim))
            if out_dim != emb_dim:
                raise RuntimeError(
                    f"Embedding dim mismatch: text emb dim={emb_dim} but GPS out_dim={out_dim}.\n"
                    f"➡️ Use only text encoders with same embedding size, OR retrain GPS for this text encoder."
                )

    # Load graph model
    model = load_graph_model(
        graph_model=args.graph_model,
        ckpt_path=args.graph_ckpt,
        device=device,
        out_dim_expected=emb_dim
    ).to(device).eval()

    # Outputs
    out_val_csv  = str(base_path / f"val_retrieved__{tag}.csv")
    out_test_csv = str(base_path / f"test_retrieved__{tag}.csv")

    # Validation retrieval + metrics
    retrieve_descriptions(
        model=model,
        train_graphs_pkl=TRAIN_GRAPHS,
        query_graphs_pkl=VAL_GRAPHS,
        train_emb_csv=train_emb_csv,
        output_csv=out_val_csv,
        device=device,
        evaluate=True
    )

    # Test retrieval (no metrics)
    retrieve_descriptions(
        model=model,
        train_graphs_pkl=TRAIN_GRAPHS,
        query_graphs_pkl=TEST_GRAPHS,
        train_emb_csv=train_emb_csv,
        output_csv=out_test_csv,
        device=device,
        evaluate=False
    )


if __name__ == "__main__":
    main()
