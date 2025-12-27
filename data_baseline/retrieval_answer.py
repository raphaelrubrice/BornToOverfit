import os

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_utils import (
    load_id2emb, load_descriptions_from_graphs, PreprocessedGraphDataset, collate_fn
)
import json
from typing import List, Dict, Tuple

import sacrebleu
from bert_score import score as bertscore

from train_gcn import MolGNN
from pathlib import Path
from argparse import ArgumentParser

def _safe_text(x) -> str:
    if x is None:
        return ""
    if not isinstance(x, str):
        return str(x)
    return x


def compute_bleu_f1(
    preds: List[str],
    refs: List[str],
    tokenize: str = "13a",
    lowercase: bool = True,
) -> Dict[str, float]:
    """
    BLEU-F1:
      - Precision proxy: BLEU(preds, refs)
      - Recall proxy:    BLEU(refs, preds)  (swap roles)
      - F1 = 2PR/(P+R)

    Notes:
      - This is a pragmatic BLEU-based F1 used in some retrieval/generation setups.
      - Scores returned in [0, 100] to match sacrebleu's BLEU scale.
    """
    preds = [_safe_text(p) for p in preds]
    refs  = [_safe_text(r) for r in refs]

    if len(preds) != len(refs):
        raise ValueError(f"preds and refs must have same length, got {len(preds)} vs {len(refs)}")

    if lowercase:
        preds = [p.lower() for p in preds]
        refs  = [r.lower() for r in refs]

    # sacrebleu expects refs as list-of-reference-streams: [refs]
    bleu_p = sacrebleu.corpus_bleu(preds, [refs], tokenize=tokenize).score
    bleu_r = sacrebleu.corpus_bleu(refs, [preds], tokenize=tokenize).score

    denom = (bleu_p + bleu_r)
    bleu_f1 = (2.0 * bleu_p * bleu_r / denom) if denom > 0 else 0.0

    return {
        "bleu_precision": float(bleu_p),
        "bleu_recall": float(bleu_r),
        "bleu_f1": float(bleu_f1),
    }


def compute_bertscore_roberta_base(
    preds: List[str],
    refs: List[str],
    device: str,
    batch_size: int = 32,
    idf: bool = False,
) -> Dict[str, float]:
    """
    BERTScore with RoBERTa-base. Returns mean precision/recall/F1 in [0, 1].
    """
    preds = [_safe_text(p) for p in preds]
    refs  = [_safe_text(r) for r in refs]

    if len(preds) != len(refs):
        raise ValueError(f"preds and refs must have same length, got {len(preds)} vs {len(refs)}")

    P, R, F1 = bertscore(
        cands=preds,
        refs=refs,
        model_type="roberta-base",
        lang="en",              # if your text is not English, you can remove this and rely on model_type only
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
    test_id2desc: Dict[str, str],
    device: str,
    save_path: str = None,
) -> Dict[str, float]:
    """
    Align predictions to references by ID, compute BLEU-F1 and BERTScore(roberta-base).
    """
    if "ID" not in results_df.columns or "description" not in results_df.columns:
        raise ValueError("results_df must contain columns: 'ID' and 'description'")

    preds = []
    refs = []

    missing = 0
    for _, row in results_df.iterrows():
        test_id = row["ID"]
        pred = row["description"]
        ref = test_id2desc.get(test_id, None)
        if ref is None:
            missing += 1
            ref = ""  # keep alignment; alternatively, skip—but then metrics become biased
        preds.append(pred)
        refs.append(ref)

    bleu_stats = compute_bleu_f1(preds, refs)
    bert_stats = compute_bertscore_roberta_base(preds, refs, device=device)

    metrics = {
        "n_samples": int(len(preds)),
        "n_missing_refs": int(missing),
        **bleu_stats,
        **bert_stats,
    }

    print("\n" + "=" * 80)
    print("Text Retrieval Metrics (retrieved description vs ground-truth test description)")
    print(f"Samples: {metrics['n_samples']} | Missing refs: {metrics['n_missing_refs']}")
    print(f"BLEU Precision: {metrics['bleu_precision']:.4f} | BLEU Recall: {metrics['bleu_recall']:.4f} | BLEU-F1: {metrics['bleu_f1']:.4f}")
    print(f"BERTScore P: {metrics['bertscore_precision']:.4f} | R: {metrics['bertscore_recall']:.4f} | F1: {metrics['bertscore_f1']:.4f}")
    print("=" * 80 + "\n")

    if save_path is not None:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"Saved metrics to: {save_path}")

    return metrics


@torch.no_grad()
def retrieve_descriptions(model, train_data, test_data, train_emb_dict, device, output_csv, evaluate=True):
    """
    Args:
        model: Trained GNN model
        train_data: Path to train preprocessed graphs
        test_data: Path to test preprocessed graphs
        train_emb_dict: Dictionary mapping train IDs to text embeddings
        device: Device to run on
        output_csv: Path to save retrieved descriptions
    """
    train_id2desc = load_descriptions_from_graphs(train_data)
    
    train_ids = list(train_emb_dict.keys())
    train_embs = torch.stack([train_emb_dict[id_] for id_ in train_ids]).to(device)
    train_embs = F.normalize(train_embs, dim=-1)
    
    print(f"Train set size: {len(train_ids)}")
    
    test_ds = PreprocessedGraphDataset(test_data)
    test_dl = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
    
    print(f"Test set size: {len(test_ds)}")
    
    test_mol_embs = []
    test_ids_ordered = []
    for graphs in test_dl:
        graphs = graphs.to(device)
        mol_emb = model(graphs)
        test_mol_embs.append(mol_emb)
        batch_size = graphs.num_graphs
        start_idx = len(test_ids_ordered)
        test_ids_ordered.extend(test_ds.ids[start_idx:start_idx + batch_size])
    
    test_mol_embs = torch.cat(test_mol_embs, dim=0)
    print(f"Encoded {test_mol_embs.size(0)} test molecules")
    
    similarities = test_mol_embs @ train_embs.t()
    
    most_similar_indices = similarities.argmax(dim=-1).cpu()
    
    results = []
    for i, test_id in enumerate(test_ids_ordered):
        train_idx = most_similar_indices[i].item()
        retrieved_train_id = train_ids[train_idx]
        retrieved_desc = train_id2desc[retrieved_train_id]
        
        results.append({
            'ID': test_id,
            'description': retrieved_desc
        })
        
        if i < 5:
            print(f"\nTest ID {test_id}: Retrieved from train ID {retrieved_train_id}")
            print(f"Description: {retrieved_desc[:150]}...")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"\n{'='*80}")
    print(f"Saved {len(results)} retrieved descriptions to: {output_csv}")
    
    if evaluate:
        test_id2desc = load_descriptions_from_graphs(test_data)
        
        metrics_path = str(Path(output_csv).with_suffix(".metrics.json"))
        evaluate_retrieval_text_metrics(
            results_df=results_df,
            test_id2desc=test_id2desc,
            device=device,
            save_path=metrics_path
        )

    return results_df


def main(folder, evaluate=True):
    file_path = Path(os.path.abspath(__file__))
    parent_folder = file_path.parent

    base_path = parent_folder.parent / folder

    print(f"Device: {DEVICE}")
    
    output_val_csv = str(base_path / "val_retrieved_descriptions.csv")
    output_test_csv = str(base_path / "test_retrieved_descriptions.csv")
    
    model_path = str(base_path / "model_checkpoint.pt")
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint '{model_path}' not found.")
        print("Please train a model first using train_gcn.py")
        return
    
    if not os.path.exists(TEST_GRAPHS):
        print(f"Error: Preprocessed graphs not found at {TEST_GRAPHS}")
        return
    
    train_emb = load_id2emb(TRAIN_EMB_CSV)
    
    emb_dim = len(next(iter(train_emb.values())))
    
    model = MolGNN(out_dim=emb_dim).to(DEVICE)
    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    retrieve_descriptions(
        model=model,
        train_data=TRAIN_GRAPHS,
        test_data=VAL_GRAPHS,
        train_emb_dict=train_emb,
        device=DEVICE,
        output_csv=output_val_csv,
        evaluate=True
    )

    retrieve_descriptions(
        model=model,
        train_data=TRAIN_GRAPHS,
        test_data=TEST_GRAPHS,
        train_emb_dict=train_emb,
        device=DEVICE,
        output_csv=output_test_csv,
        evaluate=False # cannot evaluate on Test data
    )
    


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-f_data", default="data_baseline/data", type=str)
    parser.add_argument("-f", default="data_baseline/data", type=str)

    args = parser.parse_args()
    data_folder = args.f_data
    folder = args.f
    
    # =========================================================
    # CONFIG
    # =========================================================
    file_path = Path(os.path.abspath(__file__))
    parent_folder = file_path.parent

    data_path = parent_folder.parent / data_folder
    base_path = parent_folder.parent / folder

    # Data paths
    TRAIN_GRAPHS = str(data_path / "train_graphs.pkl")
    VAL_GRAPHS   = str(data_path / "validation_graphs.pkl")
    TEST_GRAPHS  = str(data_path / "test_graphs.pkl")

    TRAIN_EMB_CSV = str(base_path / "train_embeddings.csv")
    VAL_EMB_CSV   = str(base_path / "validation_embeddings.csv")

    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 5
    LR = 1e-3
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    main(folder)

