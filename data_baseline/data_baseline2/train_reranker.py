# train_reranker.py
import os, json, random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch_geometric.data import Batch

from data_utils import GraphTextDataset, atom_vocab_sizes, bond_vocab_sizes
from reranker_model import GraphTextReranker

TRAIN_PKL = "data/train_graphs.pkl"
VAL_PKL   = "data/validation_graphs.pkl"

CANDS_TRAIN = "candidates/train_top20.json"
CANDS_VAL   = "candidates/val_top20.json"

OUTDIR = "reranker_ckpt"
CKPT = os.path.join(OUTDIR, "reranker.pt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BERT_NAME = "bert-base-uncased"
MAX_TEXT_LEN = 128
MAX_GRAPH_TOKENS = 64

BATCH = 16
EPOCHS = 2
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06

HARD_NEG_PER_POS = 5   # nb de hard negatives / example
SEED = 42


class PairDataset(Dataset):
    """
    Builds pairs (graph_idx, text, label) on the fly using candidates JSON.
    """
    def __init__(self, pkl_path: str, cands_json: str, hard_neg_per_pos: int = 5):
        self.base = GraphTextDataset(pkl_path)
        with open(cands_json, "r", encoding="utf-8") as f:
            self.cands = json.load(f)
        self.hard_neg_per_pos = hard_neg_per_pos

        # Pre-store true desc for convenience
        self.true_desc = {gid: desc for (_, desc, gid) in self.base}

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        g, pos_text, gid = self.base[idx]
        # positives
        items = [(g, pos_text, 1.0)]

        # hard negatives from retrieved list
        cand_list = [c["desc"] for c in self.cands[gid]]
        # remove exact positive if present
        cand_list = [c for c in cand_list if c.strip() != pos_text.strip()]
        if len(cand_list) == 0:
            return items

        # sample hard negs
        n = min(self.hard_neg_per_pos, len(cand_list))
        negs = random.sample(cand_list, n)
        for neg in negs:
            items.append((g, neg, 0.0))
        return items


def collate_pairs(batch, tokenizer):
    """
    batch is a list where each element is a list of items (pos + negs) for one graph.
    We flatten.
    """
    flat = []
    for items in batch:
        flat.extend(items)

    graphs, texts, labels = zip(*flat)
    bg = Batch.from_data_list(list(graphs))

    enc = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=MAX_TEXT_LEN,
        return_tensors="pt",
    )
    y = torch.tensor(labels, dtype=torch.float32)
    return bg, enc["input_ids"], enc["attention_mask"], y


@torch.no_grad()
def eval_auc_like(model, tokenizer, loader):
    # quick metric: average sigmoid(pos) vs sigmoid(neg) separation
    model.eval()
    pos_scores, neg_scores = [], []
    for bg, ids, attn, y in tqdm(loader, desc="Eval", leave=False):
        bg = bg.to(DEVICE)
        ids = ids.to(DEVICE)
        attn = attn.to(DEVICE)
        y = y.to(DEVICE)

        logit = model(bg, ids, attn)
        p = torch.sigmoid(logit)
        pos_scores.append(p[y > 0.5])
        neg_scores.append(p[y < 0.5])

    pos = torch.cat(pos_scores) if len(pos_scores) else torch.tensor([])
    neg = torch.cat(neg_scores) if len(neg_scores) else torch.tensor([])
    if pos.numel() == 0 or neg.numel() == 0:
        return {"pos_mean": None, "neg_mean": None}
    return {"pos_mean": float(pos.mean().item()), "neg_mean": float(neg.mean().item())}


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(BERT_NAME)

    train_ds = PairDataset(TRAIN_PKL, CANDS_TRAIN, hard_neg_per_pos=HARD_NEG_PER_POS)
    # DataLoader batches graphs; each sample expands into (1 + HARD_NEG_PER_POS)
    train_dl = DataLoader(
        train_ds, batch_size=BATCH, shuffle=True,
        collate_fn=lambda b: collate_pairs(b, tokenizer),
        num_workers=0
    )

    # small val set optional
    val_dl = None
    if os.path.exists(VAL_PKL) and os.path.exists(CANDS_VAL):
        val_ds = PairDataset(VAL_PKL, CANDS_VAL, hard_neg_per_pos=HARD_NEG_PER_POS)
        val_dl = DataLoader(
            val_ds, batch_size=BATCH, shuffle=False,
            collate_fn=lambda b: collate_pairs(b, tokenizer),
            num_workers=0
        )

    model = GraphTextReranker(
        atom_vocab=atom_vocab_sizes(),
        bond_vocab=bond_vocab_sizes(),
        bert_name=BERT_NAME,
        graph_dim=256,
        graph_layers=4,
        max_graph_tokens=MAX_GRAPH_TOKENS,
        dropout=0.1,
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    total_steps = EPOCHS * len(train_dl)
    warmup = int(WARMUP_RATIO * total_steps)
    sched = get_linear_schedule_with_warmup(opt, warmup, total_steps)

    best_sep = -1e9

    for ep in range(1, EPOCHS + 1):
        model.train()
        pbar = tqdm(train_dl, desc=f"Reranker Epoch {ep}/{EPOCHS}")
        for bg, input_ids, attn, y in pbar:
            bg = bg.to(DEVICE)
            input_ids = input_ids.to(DEVICE)
            attn = attn.to(DEVICE)
            y = y.to(DEVICE)

            logit = model(bg, input_ids, attn)
            loss = F.binary_cross_entropy_with_logits(logit, y)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            pbar.set_postfix(loss=float(loss.item()))

        if val_dl is not None:
            stats = eval_auc_like(model, tokenizer, val_dl)
            if stats["pos_mean"] is not None:
                sep = stats["pos_mean"] - stats["neg_mean"]
                print(f"\nVal pos_mean={stats['pos_mean']:.3f} neg_mean={stats['neg_mean']:.3f} sep={sep:.3f}")
                if sep > best_sep:
                    best_sep = sep
                    torch.save(
                        {"state": model.state_dict(), "bert_name": BERT_NAME, "max_graph_tokens": MAX_GRAPH_TOKENS},
                        CKPT
                    )
                    print(f"✅ Saved best reranker to {CKPT}")
        else:
            # save last
            torch.save({"state": model.state_dict(), "bert_name": BERT_NAME, "max_graph_tokens": MAX_GRAPH_TOKENS}, CKPT)
            print(f"✅ Saved reranker to {CKPT}")

    print("Done.")


if __name__ == "__main__":
    main()
