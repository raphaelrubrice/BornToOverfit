import os, json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch_geometric.data import Batch
from torch_geometric.nn import GINEConv, GCNConv, global_add_pool

try:
    from.data_utils import (
        load_id2emb,
        PreprocessedGraphDataset, collate_fn, x_map, e_map
    )
except:
    from data_utils import (
        load_id2emb,
        PreprocessedGraphDataset, collate_fn, x_map, e_map
    )

from pathlib import Path
from argparse import ArgumentParser


# =========================================================
# MODEL: GNN to encode graphs (simple GCN, no edge features)
# =========================================================
class Baseline_MolGNN(nn.Module):
    def __init__(self, hidden=128, out_dim=256, layers=3):
        super().__init__()

        # Use a single learnable embedding for all nodes (no node features)
        self.node_init = nn.Parameter(torch.randn(hidden))

        self.convs = nn.ModuleList()
        for _ in range(layers):
            self.convs.append(GCNConv(hidden, hidden))

        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, batch: Batch):
        # Initialize all nodes with the same learnable embedding
        num_nodes = batch.x.size(0)
        h = self.node_init.unsqueeze(0).expand(num_nodes, -1)
        
        for conv in self.convs:
            h = conv(h, batch.edge_index)
            h = F.relu(h)
        g = global_add_pool(h, batch.batch)
        g = self.proj(g)
        g = F.normalize(g, dim=-1)
        return g

class MolGNN(nn.Module):
    def __init__(self, hidden=128, out_dim=256, layers=3, x_map=None, e_map=None):
        super().__init__()
        assert x_map is not None, "Pass x_map from data_utils so embedding sizes match"
        assert e_map is not None, "Pass e_map from data_utils so edge embedding sizes match"

        # 9 categorical features in x
        self.feat_names = [
            "atomic_num",
            "chirality",
            "degree",
            "formal_charge",
            "num_hs",
            "num_radical_electrons",
            "hybridization",
            "is_aromatic",
            "is_in_ring",
        ]

        # Node feature embeddings -> hidden, summed
        self.node_embs = nn.ModuleDict({
            name: nn.Embedding(len(x_map[name]), hidden)
            for name in self.feat_names
        })

        # Edge feature embeddings (3 categorical indices per edge)
        self.edge_feat_names = ["bond_type", "stereo", "is_conjugated"]
        self.edge_embs = nn.ModuleDict({
            "bond_type": nn.Embedding(len(e_map["bond_type"]), hidden),
            "stereo": nn.Embedding(len(e_map["stereo"]), hidden),
            "is_conjugated": nn.Embedding(len(e_map["is_conjugated"]), hidden),
        })

        # Edge embedding projection (GINE expects edge_attr to match node dim)
        self.edge_proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

        # Edge-aware conv stack
        self.convs = nn.ModuleList()
        for _ in range(layers):
            nn_edge = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            self.convs.append(GINEConv(nn_edge))

        self.proj = nn.Linear(hidden, out_dim)

    def _embed_nodes(self, x: torch.Tensor) -> torch.Tensor:
        h = 0
        for i, name in enumerate(self.feat_names):
            h = h + self.node_embs[name](x[:, i])
        return h

    def _embed_edges(self, edge_attr: torch.Tensor) -> torch.Tensor:
        # edge_attr: [num_edges, 3] categorical indices
        if edge_attr is None:
            raise ValueError("edge_attr is required for edge-aware MolGNN")

        e = (
            self.edge_embs["bond_type"](edge_attr[:, 0])
            + self.edge_embs["stereo"](edge_attr[:, 1])
            + self.edge_embs["is_conjugated"](edge_attr[:, 2])
        )
        return self.edge_proj(e)

    def forward(self, batch: Batch):
        # Node embeddings
        h = self._embed_nodes(batch.x)

        # Edge embeddings
        e = self._embed_edges(batch.edge_attr)

        # Message passing with edge attributes
        for conv in self.convs:
            h = conv(h, batch.edge_index, e)
            h = F.relu(h)

        # Graph pooling + projection to embedding space
        g = global_add_pool(h, batch.batch)
        g = self.proj(g)
        g = F.normalize(g, dim=-1)
        return g

# =========================================================
# Training and Evaluation
# =========================================================
def clip_infonce_loss(mol_vec, txt_vec, temperature=0.07):
    # mol_vec, txt_vec already normalized
    logits = (txt_vec @ mol_vec.t()) / temperature  # [B,B]
    labels = torch.arange(logits.size(0), device=logits.device)

    loss_t2m = F.cross_entropy(logits, labels)
    loss_m2t = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_t2m + loss_m2t)

def train_epoch(mol_enc, loader, optimizer, device):
    mol_enc.train()

    total_loss, total = 0.0, 0
    for graphs, text_emb in loader:
        graphs = graphs.to(device)
        text_emb = text_emb.to(device)

        mol_vec = mol_enc(graphs)
        txt_vec = F.normalize(text_emb, dim=-1)

        loss = clip_infonce_loss(mol_vec, txt_vec, temperature=0.07) #F.mse_loss(mol_vec, txt_vec)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = graphs.num_graphs
        total_loss += loss.item() * bs
        total += bs

    return total_loss / total


@torch.no_grad()
def eval_retrieval(data_path, emb_dict, mol_enc, device, dl=None):
    if dl is None:
        ds = PreprocessedGraphDataset(data_path, emb_dict)
        dl = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    all_mol, all_txt = [], []
    for graphs, text_emb in dl:
        graphs = graphs.to(device)
        text_emb = text_emb.to(device)
        all_mol.append(mol_enc(graphs))
        all_txt.append(F.normalize(text_emb, dim=-1))
    all_mol = torch.cat(all_mol, dim=0)
    all_txt = torch.cat(all_txt, dim=0)

    sims = all_txt @ all_mol.t()
    ranks = sims.argsort(dim=-1, descending=True)

    N = all_txt.size(0)
    device = sims.device
    correct = torch.arange(N, device=device)

    pos = (ranks == correct.unsqueeze(1)).nonzero()[:, 1] + 1

    mrr = (1.0 / pos.float()).mean().item()

    results = {"MRR": mrr}

    for k in (1, 5, 10):
        hitk = (pos <= k).float().mean().item()
        results[f"R@{k}"] = hitk
        results[f"Hit@{k}"] = hitk

    return results

def load_gnn_from_checkpoint(
    model_path: str,
    device: str,
    x_map,
    e_map,
):
    """
    Load MolGNN using a saved config + state_dict.
    """
    model_dir = Path(model_path).parent
    config_path = model_dir / "model_config.json"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing GNN config file: {config_path}"
        )

    with open(config_path, "r") as f:
        cfg = json.load(f)

    model_class = cfg.get("model_class", "MolGNN")

    if model_class != "MolGNN":
        raise ValueError(f"Unsupported GNN class: {model_class}")

    gnn = MolGNN(
        hidden=cfg["hidden"],
        out_dim=cfg["out_dim"],
        layers=cfg["layers"],
        x_map=x_map,
        e_map=e_map,
    ).to(device)

    state = torch.load(model_path, map_location=device)
    gnn.load_state_dict(state)
    gnn.eval()

    return gnn

# =========================================================
# Main Training Loop
# =========================================================
def main(folder):
    print(f"Device: {DEVICE}")

    train_emb = load_id2emb(TRAIN_EMB_CSV)
    val_emb = load_id2emb(VAL_EMB_CSV) if os.path.exists(VAL_EMB_CSV) else None

    emb_dim = len(next(iter(train_emb.values())))

    if not os.path.exists(TRAIN_GRAPHS):
        print(f"Error: Preprocessed graphs not found at {TRAIN_GRAPHS}")
        print("Please run: python prepare_graph_data.py")
        return
    
    train_ds = PreprocessedGraphDataset(TRAIN_GRAPHS, train_emb)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    val_ds = PreprocessedGraphDataset(VAL_GRAPHS, val_emb)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    mol_enc = MolGNN(out_dim=emb_dim, hidden=HIDDEN, layers=LAYERS, x_map=x_map, e_map=e_map).to(DEVICE)

    optimizer = torch.optim.Adam(mol_enc.parameters(), lr=LR)

    for ep in range(EPOCHS):
        train_loss = train_epoch(mol_enc, train_dl, optimizer, DEVICE)

        val_scores = eval_retrieval(VAL_GRAPHS, val_emb, mol_enc, DEVICE, dl=val_dl)
        str_val_scores = {key:f'{val:.4f}' for key, val in val_scores.items()}
        print(f"Epoch {ep+1}/{EPOCHS} - loss={train_loss:.4f} - val={str_val_scores}")

    save_path = parent_folder.parent / folder
    os.makedirs(str(save_path), exist_ok=True)

    # ---- Save model config ----
    config = {
        "model_class": "MolGNN",
        "hidden": HIDDEN,
        "out_dim": emb_dim,
        "layers": LAYERS,
        "uses_edge_features": True,
    }

    config_path = str(save_path / "model_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # ---- Save weights ----
    model_path = str(save_path / "model_checkpoint.pt")
    torch.save(mol_enc.state_dict(), model_path)

    print(f"\nModel saved to {model_path}")
    print(f"Config saved to {config_path}")


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
    EPOCHS = 50
    LR = 1e-3
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    HIDDEN = 256
    LAYERS = 3

    main(folder)