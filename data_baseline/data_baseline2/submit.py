# submit.py (UPDATED: use cross-encoder reranker)
import os, json
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from torch_geometric.data import Batch as PygBatch

from data_utils import GraphOnlyDataset, collate_graph_only, atom_vocab_sizes, bond_vocab_sizes
from clip_model import CLIPGraphText
from editor_model import GraphCaptionEditor
from reranker_model import GraphTextReranker

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST = "data/test_graphs.pkl"
CANDS_TEST = "candidates/test_top20.json"

CLIP_CKPT = "clip_ckpt/clip.pt"
EDITOR_CKPT = "editor_ckpt/editor.pt"
RERANK_CKPT = "reranker_ckpt/reranker.pt"

OUTCSV = "submission.csv"

TOPM = 10  # generate TOPM variants then rerank

PROMPT_TEMPLATE = "Candidate: {cand}\nRewrite to match molecule:\n"
TEXT_MAXLEN = 192
GEN_MAX_NEW = 96

def main():
    with open(CANDS_TEST, "r", encoding="utf-8") as f:
        cands = json.load(f)

    # Load CLIP for retrieval pipeline (already done)
    ck = torch.load(CLIP_CKPT, map_location="cpu")
    clip = CLIPGraphText(atom_vocab_sizes(), bond_vocab_sizes(), graph_dim=ck["graph_dim"], text_model=ck["text_model"]).to(DEVICE)
    clip.load_state_dict(ck["state"], strict=True)
    clip.eval()

    # Load Editor
    ed_ck = torch.load(EDITOR_CKPT, map_location="cpu")
    tok_gpt = AutoTokenizer.from_pretrained(ed_ck["gpt2"])
    if tok_gpt.pad_token is None:
        tok_gpt.pad_token = tok_gpt.eos_token
    editor = GraphCaptionEditor(atom_vocab_sizes(), bond_vocab_sizes(), gpt2_name=ed_ck["gpt2"]).to(DEVICE)
    editor.load_state_dict(ed_ck["state"], strict=True)
    editor.eval()

    # Load Cross-encoder reranker
    rr_ck = torch.load(RERANK_CKPT, map_location="cpu")
    rr_tok = AutoTokenizer.from_pretrained(rr_ck["bert_name"])
    reranker = GraphTextReranker(
        atom_vocab=atom_vocab_sizes(),
        bond_vocab=bond_vocab_sizes(),
        bert_name=rr_ck["bert_name"],
        graph_dim=256,
        graph_layers=4,
        max_graph_tokens=rr_ck["max_graph_tokens"],
        dropout=0.1,
    ).to(DEVICE)
    reranker.load_state_dict(rr_ck["state"], strict=True)
    reranker.eval()

    ds = GraphOnlyDataset(TEST)
    dl = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate_graph_only)

    rows = []
    for bg, ids in tqdm(dl, desc="Submit"):
        bg = bg.to(DEVICE)

        best_texts = []
        for b, gid in enumerate(ids):
            cand_list = cands[gid][:TOPM]
            prompts = [PROMPT_TEMPLATE.format(cand=c["desc"]) for c in cand_list]

            enc = tok_gpt(prompts, padding=True, truncation=True, max_length=TEXT_MAXLEN, return_tensors="pt")
            in_ids = enc["input_ids"].to(DEVICE)
            attn = enc["attention_mask"].to(DEVICE)

            # build batch of same graph repeated TOPM times
            subgraph = bg.get_example(b)
            sub_batch = PygBatch.from_data_list([subgraph for _ in range(len(prompts))]).to(DEVICE)

            gen_ids = editor.generate(
                sub_batch, in_ids, attn, tok_gpt,
                max_new_tokens=GEN_MAX_NEW, num_beams=4, length_penalty=1.0, no_repeat_ngram_size=3
            )
            gen_texts = tok_gpt.batch_decode(gen_ids, skip_special_tokens=True)
            gen_texts = [t.split("Rewrite to match molecule:")[-1].strip() for t in gen_texts]

            # rerank with cross-encoder: score(graph, text)
            rr_enc = rr_tok(gen_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
            with torch.no_grad():
                logits = reranker(sub_batch, rr_enc["input_ids"].to(DEVICE), rr_enc["attention_mask"].to(DEVICE))
                best = gen_texts[int(torch.argmax(logits).item())]

            best_texts.append(best)

        for gid, txt in zip(ids, best_texts):
            rows.append({"ID": gid, "description": txt})

    df = pd.DataFrame(rows)
    df.to_csv(OUTCSV, index=False)
    print("Wrote", OUTCSV, "rows=", len(df))

if __name__ == "__main__":
    main()
