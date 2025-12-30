import os
import subprocess
from pathlib import Path

MODELS = [
    "bert-base-uncased",
    "roberta-base",
    "allenai/scibert_scivocab_uncased",
    "microsoft/deberta-v3-base",
    "sentence-transformers/all-mpnet-base-v2",
]

DATA_FOLDER = "data_baseline/data"
OUT_FOLDER  = "data_baseline/data"
GRAPH_CKPT  = "data_baseline/data/model_checkpoint.pt"   # GPS checkpoint
GRAPH_MODEL = "gps"

def run(cmd):
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    for m in MODELS:
        tag = m.replace("/", "__")

        run(["python", "generate_description_embeddings.py",
             "--base_model", m,
             "--normalize",
             "-f_data", DATA_FOLDER,
             "-f", OUT_FOLDER])

        run(["python", "retrieval_answer.py",
             "--tag", tag,
             "--graph_model", GRAPH_MODEL,
             "--graph_ckpt", GRAPH_CKPT,
             "-f_data", DATA_FOLDER,
             "-f", OUT_FOLDER])

if __name__ == "__main__":
    main()
