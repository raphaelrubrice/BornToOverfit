# retrieve_topk.py
import os, json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from data_utils import GraphTextDataset, GraphOnlyDataset, collate_graph_only, atom_vocab_sizes, bond_vocab_sizes
from clip_model import CLIPGraphText

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLIP_CKPT = "clip_ckpt/clip.pt"
TRAIN_TEXT_EMB = "clip_ckpt/train_text_emb.pt"

TRAIN = "data/train_graphs.pkl"
VAL   = "data/validation_graphs.pkl"
TEST  = "data/test_graphs.pkl"

OUTDIR = "candidates"
TOPK = 20

def encode_graphs(model, ds, batch_size=128):
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_graph_only)
    Zg, ids = [], []
    with torch.no_grad():
        for bg, batch_ids in tqdm(dl, desc="Encode graphs"):
            bg = bg.to(DEVICE)
            z = model.g(bg).cpu()
            Zg.append(z)
            ids.extend(batch_ids)
    return torch.cat(Zg, dim=0), ids

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    ck = torch.load(CLIP_CKPT, map_location="cpu")
    text_model = ck["text_model"]
    graph_dim = ck["graph_dim"]

    model = CLIPGraphText(atom_vocab_sizes(), bond_vocab_sizes(), graph_dim=graph_dim, text_model=text_model).to(DEVICE)
    model.load_state_dict(ck["state"], strict=True)
    model.eval()

    train_pack = torch.load(TRAIN_TEXT_EMB)
    train_ids = train_pack["ids"]
    train_desc = train_pack["desc"]
    Zt = train_pack["Z"].to(DEVICE)                 # (N, d)
    Zt = torch.nn.functional.normalize(Zt, dim=-1)

    for name, pkl in [("train", TRAIN), ("val", VAL), ("test", TEST)]:
        ds = GraphOnlyDataset(pkl)
        Zg, q_ids = encode_graphs(model, ds)
        Zg = Zg.to(DEVICE)
        Zg = torch.nn.functional.normalize(Zg, dim=-1)

        sims = Zg @ Zt.t()  # (Q, N)
        topk = sims.topk(TOPK, dim=-1).indices.cpu()

        out = {}
        for i, qid in enumerate(q_ids):
            cand_idx = topk[i].tolist()
            out[qid] = [ {"train_id": train_ids[j], "desc": train_desc[j]} for j in cand_idx ]

        with open(os.path.join(OUTDIR, f"{name}_top{TOPK}.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)
        print("Wrote", os.path.join(OUTDIR, f"{name}_top{TOPK}.json"))

if __name__ == "__main__":
    main()
