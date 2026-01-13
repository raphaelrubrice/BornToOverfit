# generate_submission.py
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from data_utils import GraphCaptionDataset, collate_caption_fn
from train_molca import MolCAStyleCaptioner, TEST_GRAPHS, DEVICE

@torch.no_grad()
def main():
    t5_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(t5_name)

    ds = GraphCaptionDataset(TEST_GRAPHS, split="test")
    dl = DataLoader(ds, batch_size=16, shuffle=False,
                    collate_fn=lambda b: collate_caption_fn(b, tokenizer, max_len=128))

    model = MolCAStyleCaptioner(t5_name=t5_name, hidden=256, num_queries=16).to(DEVICE)
    model.load_state_dict(torch.load("molca_t5_best.pt", map_location=DEVICE))
    model.eval()

    rows = []
    for graphs, _, _ in dl:
        graphs = graphs.to(DEVICE)
        node_emb, batch_vec = model.graph_enc(graphs)
        q = model.qformer(node_emb, batch_vec)
        enc = model.proj(q)
        attn_mask = torch.ones(enc.shape[:2], device=enc.device, dtype=torch.long)

        gen = model.t5.generate(
            inputs_embeds=enc,
            attention_mask=attn_mask,
            max_new_tokens=80,
            num_beams=5,
            length_penalty=1.0,
            no_repeat_ngram_size=3
        )
        texts = tokenizer.batch_decode(gen, skip_special_tokens=True)

        # récupérer les IDs dans le bon ordre
        start = len(rows)
        for i, txt in enumerate(texts):
            rows.append({"ID": ds.ids[start + i], "description": txt})

    pd.DataFrame(rows).to_csv("submission.csv", index=False)
    print("Saved submission.csv")

if __name__ == "__main__":
    main()
