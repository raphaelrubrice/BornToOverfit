"""
Data loading and processing utilities for molecule-text retrieval.
Includes dataset classes and data loading functions.
"""
from typing import Dict
import pickle

import pandas as pd
import torch
from collections import Counter
from typing import Dict, Any, Optional, List, Tuple
from torch.utils.data import Dataset
from torch_geometric.data import Batch


# =========================================================
# Feature maps for atom and bond attributes
# =========================================================

x_map: Dict[str, List[Any]] = {
    'atomic_num': list(range(0, 119)),
    'chirality': [
        'CHI_UNSPECIFIED','CHI_TETRAHEDRAL_CW','CHI_TETRAHEDRAL_CCW','CHI_OTHER',
        'CHI_TETRAHEDRAL','CHI_ALLENE','CHI_SQUAREPLANAR','CHI_TRIGONALBIPYRAMIDAL',
        'CHI_OCTAHEDRAL',
    ],
    'degree': list(range(0, 11)),
    'formal_charge': list(range(-5, 7)),
    'num_hs': list(range(0, 9)),
    'num_radical_electrons': list(range(0, 5)),
    'hybridization': [
        'UNSPECIFIED','S','SP','SP2','SP3','SP3D','SP3D2','OTHER',
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

e_map: Dict[str, List[Any]] = {
    'bond_type': [
        'UNSPECIFIED','SINGLE','DOUBLE','TRIPLE','QUADRUPLE','QUINTUPLE','HEXTUPLE',
        'ONEANDAHALF','TWOANDAHALF','THREEANDAHALF','FOURANDAHALF','FIVEANDAHALF',
        'AROMATIC','IONIC','HYDROGEN','THREECENTER','DATIVEONE','DATIVE','DATIVEL',
        'DATIVER','OTHER','ZERO',
    ],
    'stereo': [
        'STEREONONE','STEREOANY','STEREOZ','STEREOE','STEREOCIS','STEREOTRANS',
    ],
    'is_conjugated': [False, True],
}


# =========================================================
# Load precomputed text embeddings
# =========================================================
def load_id2emb(csv_path: str) -> Dict[str, torch.Tensor]:
    """
    Load precomputed text embeddings from CSV file.
    
    Args:
        csv_path: Path to CSV file with columns: ID, embedding
                  where embedding is comma-separated floats
        
    Returns:
        Dictionary mapping ID (str) to embedding tensor
    """
    df = pd.read_csv(csv_path)
    id2emb = {}
    for _, row in df.iterrows():
        id_ = str(row["ID"])
        emb_str = row["embedding"]
        emb_vals = [float(x) for x in str(emb_str).split(',')]
        id2emb[id_] = torch.tensor(emb_vals, dtype=torch.float32)
    return id2emb


# =========================================================
# Load descriptions from preprocessed graphs
# =========================================================
def load_descriptions_from_graphs(graph_path: str) -> Dict[str, str]:
    """
    Load ID to description mapping from preprocessed graph file.
    
    Args:
        graph_path: Path to .pkl file containing list of pre-saved graphs
        
    Returns:
        Dictionary mapping ID (str) to description (str)
    """
    with open(graph_path, 'rb') as f:
        graphs = pickle.load(f)
    
    id2desc = {}
    for graph in graphs:
        id2desc[graph.id] = graph.description
    
    return id2desc

# Text repr

_PERIODIC_TABLE = [
    None,
    "H","He",
    "Li","Be","B","C","N","O","F","Ne",
    "Na","Mg","Al","Si","P","S","Cl","Ar",
    "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Ga","Ge","As","Se","Br","Kr",
    "Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
    "In","Sn","Sb","Te","I","Xe",
    "Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
    "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
    "Tl","Pb","Bi","Po","At","Rn",
    "Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr",
    "Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn",
    "Nh","Fl","Mc","Lv","Ts","Og"
]

def atomic_number_to_symbol(z: int) -> str:
    try:
        z = int(z)
    except Exception:
        return "Z?"
    if 1 <= z <= 118:
        return _PERIODIC_TABLE[z]
    return "UNK_Z"

_HALOGENS = {"F", "Cl", "Br", "I"}
_ORGANOGENS = {"C", "H", "N", "O", "P", "S", "Se", "B", "Si"}

def _safe_map_lookup(map_list: List[Any], idx: int, default: Any) -> Any:
    try:
        if idx is None:
            return default
        idx_int = int(idx)
        if 0 <= idx_int < len(map_list):
            return map_list[idx_int]
        return default
    except Exception:
        return default

def _format_counter(counter: Counter, keys: Optional[List[str]] = None, max_items: int = 12) -> str:
    if not counter:
        return "none"
    items = counter.items()
    if keys is not None:
        # Keep specified order, then remaining by count
        ordered = [(k, counter.get(k, 0)) for k in keys if counter.get(k, 0) > 0]
        remaining = [(k, v) for k, v in items if k not in set(keys)]
        remaining.sort(key=lambda kv: kv[1], reverse=True)
        items2 = ordered + remaining
    else:
        items2 = sorted(items, key=lambda kv: kv[1], reverse=True)

    items2 = items2[:max_items]
    return " ".join([f"{k}={v}" for k, v in items2 if v > 0]) or "none"

def make_mol_repr(
    data,
    *,
    include_id: bool = False,
    include_histograms: bool = True,
    include_stereo: bool = False,
    max_top_elements: int = 10,
) -> str:
    """
    Build a chemically grounded, text-serializable "molecule card" from a PyG Data object.

    Expects:
      - data.x: [num_nodes, 9] categorical indices
      - data.edge_index: [2, num_directed_edges]
      - data.edge_attr: [num_directed_edges, 3] categorical indices

    Returns:
      A compact multi-line string suitable for conditioning a text generator.

    Notes:
      - Edges are stored as directed pairs; bond counts are computed on undirected edges (i<j).
      - Feature index -> value decoding uses x_map/e_map from data_utils.py.
    """
    x = data.x
    edge_index = getattr(data, "edge_index", None)
    edge_attr = getattr(data, "edge_attr", None)

    n = int(x.size(0)) if isinstance(x, torch.Tensor) else 0
    lines: List[str] = []
    if include_id and hasattr(data, "id"):
        lines.append(f"id: {getattr(data, 'id')}")

    # -------- Node features --------
    if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.size(1) < 9:
        # Fallback if malformed
        lines.append(f"atoms_total: {n}")
        return "[MOL_FEATURES]\n" + "\n".join(lines) + "\n[/MOL_FEATURES]"

    atomic_num_idx = x[:, 0].tolist()
    chirality_idx = x[:, 1].tolist()
    degree_idx = x[:, 2].tolist()
    formal_charge_idx = x[:, 3].tolist()
    num_hs_idx = x[:, 4].tolist()
    radical_idx = x[:, 5].tolist()
    hybrid_idx = x[:, 6].tolist()
    aromatic_idx = x[:, 7].tolist()
    ring_idx = x[:, 8].tolist()

    # Decode atomic numbers
    atomic_nums = [_safe_map_lookup(x_map["atomic_num"], i, 0) for i in atomic_num_idx]
    elem_syms = []
    for z in atomic_nums:
        try:
            z_int = int(z)
        except Exception:
            z_int = 0
        elem_syms.append(atomic_number_to_symbol(z_int))

    elem_counts = Counter(elem_syms)
    # Aggregate halogens/metals/other
    halogen_count = sum(v for k, v in elem_counts.items() if k in _HALOGENS)
    organogen_count = sum(v for k, v in elem_counts.items() if k in _ORGANOGENS)
    metals_or_other = n - (halogen_count + organogen_count)

    # Net formal charge (decode indices)
    formal_charges = [_safe_map_lookup(x_map["formal_charge"], i, 0) for i in formal_charge_idx]
    formal_charge_net = int(sum(int(fc) for fc in formal_charges))

    # Aromatic/ring counts
    aromatic_bools = [_safe_map_lookup(x_map["is_aromatic"], i, False) for i in aromatic_idx]
    ring_bools = [_safe_map_lookup(x_map["is_in_ring"], i, False) for i in ring_idx]
    aromatic_atoms = int(sum(bool(a) for a in aromatic_bools))
    ring_atoms = int(sum(bool(r) for r in ring_bools))

    # Chirality / radicals
    chiral_tags = [_safe_map_lookup(x_map["chirality"], i, "CHI_UNSPECIFIED") for i in chirality_idx]
    chirality_centers = int(sum(t != "CHI_UNSPECIFIED" for t in chiral_tags))

    radical_vals = [_safe_map_lookup(x_map["num_radical_electrons"], i, 0) for i in radical_idx]
    radical_atoms = int(sum(int(rv) > 0 for rv in radical_vals))

    # Hybridization histogram (decoded)
    hybrid_vals = [_safe_map_lookup(x_map["hybridization"], i, "UNSPECIFIED") for i in hybrid_idx]
    hybrid_counts = Counter(hybrid_vals)

    # Degree histogram (decoded)
    degree_vals = [_safe_map_lookup(x_map["degree"], i, 0) for i in degree_idx]
    degree_counts = Counter(int(d) for d in degree_vals)

    # H count histogram (decoded)
    num_hs_vals = [_safe_map_lookup(x_map["num_hs"], i, 0) for i in num_hs_idx]
    num_hs_counts = Counter(int(h) for h in num_hs_vals)

    # Summarize elements with top-k + other buckets
    # Keep key elements first; then others by frequency.
    key_order = [
                # Core organic chemistry
                "C", "H", "N", "O", "P", "S",

                # Halogens
                "F", "Cl", "Br", "I", "At",

                # Common metalloids
                "B", "Si", "As", "Se", "Te",

                # Alkali / alkaline earth metals (frequent in salts)
                "Li", "Na", "K", "Rb", "Cs",
                "Mg", "Ca", "Sr", "Ba",

                # Common transition metals (biochem / drugs / complexes)
                "Fe", "Cu", "Zn", "Mn", "Co", "Ni", "Cr", "Mo",

                # Heavy metals often mentioned explicitly
                "Ag", "Au", "Hg", "Pb", "Cd", "Pt", "Pd",
            ]
    elem_str = _format_counter(elem_counts, keys=key_order, max_items=max_top_elements)

    # -------- Edge features (undirected) --------
    bond_counts = Counter()
    stereo_counts = Counter()
    conjugated_count = 0
    undirected_edges = 0

    if isinstance(edge_index, torch.Tensor) and isinstance(edge_attr, torch.Tensor) and edge_index.numel() > 0:
        # Build mask for one direction only: i < j
        src = edge_index[0]
        dst = edge_index[1]
        mask = src < dst
        if mask.any():
            ei = edge_index[:, mask]
            ea = edge_attr[mask]
            undirected_edges = int(ei.size(1))

            # Decode bond type, stereo, conjugation
            bond_types = [
                _safe_map_lookup(e_map["bond_type"], int(i), "UNSPECIFIED")
                for i in ea[:, 0].tolist()
            ]
            stereos = [
                _safe_map_lookup(e_map["stereo"], int(i), "STEREONONE")
                for i in ea[:, 1].tolist()
            ]
            conjugated_flags = [
                bool(_safe_map_lookup(e_map["is_conjugated"], int(i), False))
                for i in ea[:, 2].tolist()
            ]

            bond_counts.update(bond_types)
            stereo_counts.update(stereos)
            conjugated_count = int(sum(conjugated_flags))

    # -------- Heuristic tags (optional but useful for RL constraints) --------
    elem_C = elem_counts.get("C", 0)
    elem_N = elem_counts.get("N", 0)
    elem_O = elem_counts.get("O", 0)
    elem_P = elem_counts.get("P", 0)
    elem_S = elem_counts.get("S", 0)
    hetero = elem_N + elem_O + elem_P + elem_S + elem_counts.get("F", 0) + elem_counts.get("Cl", 0) + elem_counts.get("Br", 0) + elem_counts.get("I", 0)

    hetero_ratio = float(hetero) / float(max(n, 1))
    aromatic_bonds = bond_counts.get("AROMATIC", 0)
    total_bonds = sum(bond_counts.values()) if bond_counts else 0
    aromatic_bond_ratio = float(aromatic_bonds) / float(max(total_bonds, 1))

    tags: List[str] = []
    if formal_charge_net <= -1:
        tags.append(f"net_negative_charge({formal_charge_net})")
    elif formal_charge_net >= 1:
        tags.append(f"net_positive_charge(+{formal_charge_net})")
    if elem_P > 0 and elem_O >= 3:
        tags.append("contains_P_and_O")
    if elem_S > 0 and elem_O >= 2:
        tags.append("contains_S_and_O")
    if aromatic_atoms >= 6 or aromatic_bond_ratio >= 0.2:
        tags.append("aromatic_character")
    if ring_atoms >= 5:
        tags.append("ring_system")
    if hetero_ratio >= 0.35:
        tags.append("heteroatom_rich")
    if radical_atoms > 0:
        tags.append("radical_possible")

    # -------- Compose card --------
    lines.append(f"atoms_total: {n}")
    lines.append(f"elements: {elem_str}")
    lines.append(f"halogens_total: {halogen_count}")
    lines.append(f"metals_or_other_total: {metals_or_other}")
    lines.append(f"formal_charge_net: {formal_charge_net}")
    lines.append(f"aromatic_atoms: {aromatic_atoms}")
    lines.append(f"ring_atoms: {ring_atoms}")
    lines.append(f"undirected_bonds: {undirected_edges}")

    if bond_counts:
        # Prefer common bond orders first; then remainder
        bond_order = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "ONEANDAHALF", "IONIC", "HYDROGEN", "DATIVE", "OTHER", "UNSPECIFIED", "ZERO"]
        lines.append(f"bond_types: {_format_counter(bond_counts, keys=bond_order, max_items=12)}")
        lines.append(f"conjugated_bonds: {conjugated_count}")

    if include_histograms:
        # Keep these compact; they serve as factual anchors
        lines.append(f"hybridization: {_format_counter(hybrid_counts, keys=['SP','SP2','SP3','SP3D','SP3D2','S','OTHER','UNSPECIFIED'], max_items=8)}")
        # Bucket degrees for readability
        deg_bucket = Counter()
        for d, c in degree_counts.items():
            if d <= 4:
                deg_bucket[str(d)] += c
            else:
                deg_bucket["5+"] += c
        lines.append(f"degree_hist: {_format_counter(deg_bucket, keys=['0','1','2','3','4','5+'], max_items=6)}")
        lines.append(f"num_hs_hist: {_format_counter(num_hs_counts, keys=[str(i) for i in range(0, 9)], max_items=9)}")
        lines.append(f"chirality_centers: {chirality_centers}")
        lines.append(f"radical_atoms: {radical_atoms}")

    if include_stereo and stereo_counts:
        lines.append(f"stereo: {_format_counter(stereo_counts, keys=['STEREONONE','STEREOANY','STEREOZ','STEREOE','STEREOCIS','STEREOTRANS'], max_items=6)}")

    if tags:
        lines.append(f"tags: {'; '.join(tags)}")

    return "[MOL_FEATURES]\n" + "\n".join(lines) + "\n[/MOL_FEATURES]"

def load_mol_cards_from_graphs(graph_pkl: str) -> Dict[str, str]:
    """
    Loads id -> mol_card from a graphs.pkl. If mol_card is missing (older pickles),
    compute it on-the-fly using make_mol_repr and return it (without saving).
    """
    import pickle

    with open(graph_pkl, "rb") as f:
        graphs = pickle.load(f)

    id2card: Dict[str, str] = {}
    for g in graphs:
        gid = getattr(g, "id", None)
        if gid is None:
            continue

        if hasattr(g, "mol_card") and isinstance(getattr(g, "mol_card"), str) and len(getattr(g, "mol_card")) > 0:
            id2card[str(gid)] = getattr(g, "mol_card")
        else:
            # Fallback for backward compatibility
            id2card[str(gid)] = make_mol_repr(
                g,
                include_id=False,
                include_histograms=True,
                include_stereo=False,
            )
    return id2card

# =========================================================
# Dataset that loads preprocessed graphs and text embeddings
# =========================================================
class PreprocessedGraphDataset(Dataset):
    """
    Dataset that loads pre-saved molecule graphs with optional text embeddings.
    
    Args:
        graph_path: Path to .pkl file containing list of pre-saved graphs
        emb_dict: Dictionary mapping ID to text embedding tensors (optional)
    """
    def __init__(self, graph_path: str, emb_dict: Dict[str, torch.Tensor] = None):
        print(f"Loading graphs from: {graph_path}")
        with open(graph_path, 'rb') as f:
            self.graphs = pickle.load(f)
        self.emb_dict = emb_dict
        self.ids = [g.id for g in self.graphs]
        print(f"Loaded {len(self.graphs)} graphs")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        if self.emb_dict is not None:
            id_ = graph.id
            text_emb = self.emb_dict[id_]
            return graph, text_emb
        else:
            return graph


def collate_fn(batch):
    """
    Collate function for DataLoader to batch graphs with optional text embeddings.
    
    Args:
        batch: List of graph Data objects or (graph, text_embedding) tuples
        
    Returns:
        Batched graph or (batched_graph, stacked_text_embeddings)
    """
    if isinstance(batch[0], tuple):
        graphs, text_embs = zip(*batch)
        batch_graph = Batch.from_data_list(list(graphs))
        text_embs = torch.stack(text_embs, dim=0)
        return batch_graph, text_embs
    else:
        return Batch.from_data_list(batch)

if __name__ == "__main__":
    import os
    import pickle
    from pathlib import Path

    file_path = Path(os.path.abspath(__file__))
    parent_folder = file_path.parent
    base_path = parent_folder / "data"

    print("=" * 100)
    print("BUILD + SAVE: mol_card for all graphs")
    print("=" * 100)

    candidates = [
        base_path / "train_graphs.pkl",
        base_path / "validation_graphs.pkl",
        base_path / "test_graphs.pkl",
    ]

    def _add_mol_cards_inplace(graphs):
        updated = 0
        for g in graphs:
            # Always (re)compute to ensure consistent settings across runs
            g.mol_card = make_mol_repr(
                g,
                include_id=False,
                include_histograms=True,
                include_stereo=False,
            )
            updated += 1
        return updated

    processed_any = False
    for pkl_path in candidates:
        if not pkl_path.exists():
            continue

        processed_any = True
        print(f"\n[1/3] Loading: {pkl_path}")
        with open(pkl_path, "rb") as f:
            graphs = pickle.load(f)

        if not graphs:
            raise RuntimeError(f"Loaded empty list from {pkl_path}")

        print(f"Loaded graphs: {len(graphs)}")
        n_updated = _add_mol_cards_inplace(graphs)
        print(f"Added/updated mol_card for: {n_updated}")

        print(f"[2/3] Saving back to same location: {pkl_path}")
        with open(pkl_path, "wb") as f:
            pickle.dump(graphs, f)

        print(f"[3/3] Reloading to verify mol_card persisted: {pkl_path}")
        with open(pkl_path, "rb") as f:
            graphs_reloaded = pickle.load(f)

        if not graphs_reloaded:
            raise RuntimeError(f"Reloaded empty list from {pkl_path}")

        # Verification: check a handful deterministically + full scan for attribute presence
        sample_idxs = [0, min(1, len(graphs_reloaded) - 1), len(graphs_reloaded) - 1]
        for idx in sample_idxs:
            g = graphs_reloaded[idx]
            if not hasattr(g, "mol_card"):
                raise AssertionError(f"Graph at idx={idx} in {pkl_path} is missing 'mol_card'")
            if not isinstance(getattr(g, "mol_card"), str) or len(getattr(g, "mol_card")) == 0:
                raise AssertionError(f"Graph at idx={idx} in {pkl_path} has invalid 'mol_card'")

        # Full scan (fast) to ensure every graph has mol_card
        missing = sum(1 for g in graphs_reloaded if not hasattr(g, "mol_card"))
        if missing != 0:
            raise AssertionError(f"{missing} graphs in {pkl_path} missing 'mol_card' after reload")

        # Print a short sample
        g0 = graphs_reloaded[0]
        print("\n--- Sample mol_card (first graph) ---")
        print(g0.mol_card[:1200] + ("..." if len(g0.mol_card) > 1200 else ""))

        if hasattr(g0, "description"):
            desc = getattr(g0, "description", "")
            print("\n--- Ground-truth description (truncated) ---")
            print(desc[:500] + ("..." if len(desc) > 500 else ""))

        print(f"\nOK: {pkl_path.name} updated and verified.")

    if not processed_any:
        print(f"No .pkl graph files found under: {base_path}")
        print("Nothing to update.")

    print("\nDONE")
    print("=" * 100)