import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import AutoTokenizer

from data_utils import MoleculeTestDataset, test_collate_fn, x_map, e_map
from model_captioning import Graph2TextModel


# =========================================================
# PATHS
# =========================================================
TEST_GRAPHS = "data/test_graphs.pkl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_DIR = os.environ.get("CHECKPOINT_DIR", "checkpoints")
BEST_CKPT_PATH = os.path.join(MODEL_DIR, "graph2text_gpt2_best.pt")

OUTPUT_CSV = os.environ.get("OUTPUT_CSV", "test_generated_descriptions.csv")


def main():
    print(f"Device: {DEVICE}")
    print(f"Checkpoint dir: {MODEL_DIR}")

    if not os.path.exists(BEST_CKPT_PATH):
        raise FileNotFoundError(f"BEST checkpoint not found: {BEST_CKPT_PATH}. Train first.")

    if not os.path.exists(TEST_GRAPHS):
        raise FileNotFoundError(f"Test graphs not found: {TEST_GRAPHS}")

    ckpt = torch.load(BEST_CKPT_PATH, map_location="cpu")
    tokenizer_name = ckpt.get("tokenizer_name", "gpt2")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    model = Graph2TextModel(
        atom_vocab_sizes=atom_vocab_sizes,
        bond_vocab_sizes=bond_vocab_sizes,
        d_model=768,
        gnn_layers=4,
        gpt2_name=tokenizer_name,
        dropout=0.1,
    ).to(DEVICE)

    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    test_ds = MoleculeTestDataset(TEST_GRAPHS)
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=test_collate_fn)

    results = []
    for batch_graph, ids in tqdm(test_dl, desc="Generating"):
        batch_graph = batch_graph.to(DEVICE)

        gen_ids = model.generate(
            batch_graph,
            tokenizer,
            max_new_tokens=128,
            num_beams=5,
            length_penalty=1.0,
        )
        texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        texts = [t.strip() for t in texts]

        for _id, txt in zip(ids, texts):
            results.append({"ID": _id, "description": txt})

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Saved: {OUTPUT_CSV} ({len(df)} rows)")
    print(df.head())


if __name__ == "__main__":
    main()
