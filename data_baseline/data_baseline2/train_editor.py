# train_editor.py
import os, json
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from data_utils import GraphTextDataset, atom_vocab_sizes, bond_vocab_sizes
from torch_geometric.data import Batch
from editor_model import GraphCaptionEditor

TRAIN = "data/train_graphs.pkl"
VAL   = "data/validation_graphs.pkl"
CANDS_TRAIN = "candidates/train_top20.json"
CANDS_VAL   = "candidates/val_top20.json"

OUTDIR = "editor_ckpt"
CKPT = os.path.join(OUTDIR, "editor.pt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPT2_NAME = "gpt2"
BATCH = 8
EPOCHS = 3
LR = 5e-5
MAXLEN = 192

PROMPT_TEMPLATE = "Candidate: {cand}\nRewrite to match molecule:\n"

class EditorDataset(Dataset):
    def __init__(self, pkl_path: str, cands_json: str):
        self.base = GraphTextDataset(pkl_path)
        with open(cands_json, "r", encoding="utf-8") as f:
            self.cands = json.load(f)

    def __len__(self): return len(self.base)

    def __getitem__(self, idx: int):
        g, target, gid = self.base[idx]
        cand = self.cands[gid][0]["desc"]  # top-1
        prompt = PROMPT_TEMPLATE.format(cand=cand)
        return g, prompt, target, gid

def collate_editor(batch, tokenizer):
    graphs, prompts, targets, ids = zip(*batch)
    bg = Batch.from_data_list(list(graphs))

    # input = prompt + target (teacher forcing)
    full = [p + t for p, t in zip(prompts, targets)]
    enc = tokenizer(full, padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt")
    input_ids = enc["input_ids"]
    attn = enc["attention_mask"]

    # labels: mask prompt tokens to -100
    labels = input_ids.clone()
    for i, p in enumerate(prompts):
        p_ids = tokenizer(p, truncation=True, max_length=MAXLEN, return_tensors="pt")["input_ids"][0]
        plen = min(p_ids.numel(), labels.size(1))
        labels[i, :plen] = -100
    labels[labels == tokenizer.pad_token_id] = -100
    return bg, input_ids, attn, labels, list(ids)

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(GPT2_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train_ds = EditorDataset(TRAIN, CANDS_TRAIN)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                          collate_fn=lambda b: collate_editor(b, tok))

    model = GraphCaptionEditor(atom_vocab_sizes(), bond_vocab_sizes(), gpt2_name=GPT2_NAME).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    total_steps = EPOCHS * len(train_dl)
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=int(0.06*total_steps), num_training_steps=total_steps)

    for ep in range(1, EPOCHS+1):
        model.train()
        pbar = tqdm(train_dl, desc=f"Editor Epoch {ep}/{EPOCHS}")
        for bg, input_ids, attn, labels, ids in pbar:
            bg = bg.to(DEVICE)
            input_ids = input_ids.to(DEVICE)
            attn = attn.to(DEVICE)
            labels = labels.to(DEVICE)

            out = model(bg, input_ids, attn, labels)
            loss = out.loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            pbar.set_postfix(loss=float(loss.item()))

    torch.save({"state": model.state_dict(), "gpt2": GPT2_NAME}, CKPT)
    print("Saved:", CKPT)

if __name__ == "__main__":
    main()
