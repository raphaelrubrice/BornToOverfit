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
from rdkit import Chem, rdBase
from rdkit.Chem import rdchem
from tqdm.auto import tqdm

# This prevents the red "Can't kekulize" spam in your console
rdBase.DisableLog('rdApp.*') 

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


def pyg_to_smiles(data, x_map, e_map) -> str:
    """
    Robust reconstruction of SMILES from PyG Data.
    Silences RDKit errors and attempts multiple fallback strategies.
    """
    try:
        mol = Chem.RWMol()
        node_to_idx = {}

        # --- Step A: Rebuild Graph (Atoms & Bonds) ---
        x = data.x
        for i in range(x.size(0)):
            # atomic_num is usually index 0
            atomic_num = _safe_map_lookup(x_map["atomic_num"], x[i, 0].item(), 6)
            a = Chem.Atom(int(atomic_num))
            
            # Formal Charge (Index 3) - vital for validity
            f_charge = _safe_map_lookup(x_map["formal_charge"], x[i, 3].item(), 0)
            a.SetFormalCharge(int(f_charge))
            
            # Chirality (Index 1) - helpful but optional
            chirality = _safe_map_lookup(x_map["chirality"], x[i, 1].item(), "CHI_UNSPECIFIED")
            if chirality == "CHI_TETRAHEDRAL_CW":
                a.SetChiralTag(Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW)
            elif chirality == "CHI_TETRAHEDRAL_CCW":
                a.SetChiralTag(Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW)

            idx = mol.AddAtom(a)
            node_to_idx[i] = idx

        edge_index = data.edge_index
        edge_attr = data.edge_attr
        
        for i in range(edge_index.size(1)):
            src = edge_index[0, i].item()
            dst = edge_index[1, i].item()
            
            if src >= dst: continue 
            
            bond_type_str = _safe_map_lookup(e_map["bond_type"], edge_attr[i, 0].item(), "SINGLE")
            
            if bond_type_str == "SINGLE": btype = Chem.rdchem.BondType.SINGLE
            elif bond_type_str == "DOUBLE": btype = Chem.rdchem.BondType.DOUBLE
            elif bond_type_str == "TRIPLE": btype = Chem.rdchem.BondType.TRIPLE
            elif bond_type_str == "AROMATIC": btype = Chem.rdchem.BondType.AROMATIC
            else: btype = Chem.rdchem.BondType.SINGLE
            
            mol.AddBond(node_to_idx[src], node_to_idx[dst], btype)

        mol = mol.GetMol()

        # --- Step B: Safe Conversion ---
        
        # 1. Fast update (no strict valence checks)
        try:
            mol.UpdatePropertyCache(strict=False)
        except:
            pass

        # 2. Try Standard SMILES (Kekulized)
        try:
            # We skip explicit SanitizeMol because it throws the C++ errors.
            # MolToSmiles triggers implicit sanitization.
            return Chem.MolToSmiles(mol, isomericSmiles=True, kekuleSmiles=True)
        except:
            pass

        # 3. Fallback: Raw SMILES (Unkekulized)
        # This handles the "Can't kekulize" cases by writing aromatic bonds explicitly (e.g. c:c)
        try:
            return Chem.MolToSmiles(mol, isomericSmiles=True, kekuleSmiles=False)
        except:
            return "INVALID_SMILES"

    except Exception:
        return "INVALID_SMILES"

# def _safe_map_lookup(mapping, idx, default):
#     # Reverse lookup helper: assumes mapping is {value: index}
#     # You likely need to invert your dictionaries once: {index: value}
#     for k, v in mapping.items():
#         if v == idx:
#             return k
#     return default

# Format: { AtomicNumber: (StandardMass, AbundanceOfMajorIsotope, SpinOfMajorIsotope) }
# Data sources: IUPAC Standard Atomic Weights & NNDC NuDat (for spins/abundance).
# "Abundance" is roughly the % natural abundance of the most common isotope. (set to 0 for synthetic elements)
# "Spin" is I for the most common isotope. "0" usually implies an even-even nucleus (like C12, O16).

# Format: { AtomicNumber: (StandardMass, NaturalAbundanceOfMajorIsotope, NuclearSpinOfMajorIsotope) }
ATOM_PROPS = {
    1:   (1.0080,   99.98,  "1/2"),   # H  (H-1)
    2:   (4.0026,   100.00, "0"),     # He (He-4)
    3:   (6.9400,   92.50,  "3/2"),   # Li (Li-7)
    4:   (9.0122,   100.00, "3/2"),   # Be (Be-9)
    5:   (10.8100,  80.10,  "3/2"),   # B  (B-11)
    6:   (12.0110,  98.93,  "0"),     # C  (C-12)
    7:   (14.0070,  99.64,  "1"),     # N  (N-14)
    8:   (15.9990,  99.76,  "0"),     # O  (O-16)
    9:   (18.9980,  100.00, "1/2"),   # F  (F-19)
    10:  (20.1800,  90.48,  "0"),     # Ne (Ne-20)
    11:  (22.9900,  100.00, "3/2"),   # Na (Na-23)
    12:  (24.3050,  78.99,  "0"),     # Mg (Mg-24)
    13:  (26.9820,  100.00, "5/2"),   # Al (Al-27)
    14:  (28.0850,  92.22,  "0"),     # Si (Si-28)
    15:  (30.9740,  100.00, "1/2"),   # P  (P-31)
    16:  (32.0600,  94.99,  "0"),     # S  (S-32)
    17:  (35.4500,  75.76,  "3/2"),   # Cl (Cl-35)
    18:  (39.9500,  99.60,  "0"),     # Ar (Ar-40)
    19:  (39.0980,  93.26,  "3/2"),   # K  (K-39)
    20:  (40.0780,  96.94,  "0"),     # Ca (Ca-40)
    21:  (44.9560,  100.00, "7/2"),   # Sc (Sc-45)
    22:  (47.8670,  73.72,  "0"),     # Ti (Ti-48)
    23:  (50.9420,  99.75,  "7/2"),   # V  (V-51)
    24:  (51.9960,  83.79,  "0"),     # Cr (Cr-52)
    25:  (54.9380,  100.00, "5/2"),   # Mn (Mn-55)
    26:  (55.8450,  91.75,  "0"),     # Fe (Fe-56)
    27:  (58.9330,  100.00, "7/2"),   # Co (Co-59)
    28:  (58.6930,  68.08,  "0"),     # Ni (Ni-58)
    29:  (63.5460,  69.15,  "3/2"),   # Cu (Cu-63)
    30:  (65.3800,  48.63,  "0"),     # Zn (Zn-64)
    31:  (69.7230,  60.11,  "3/2"),   # Ga (Ga-69)
    32:  (72.6300,  35.94,  "0"),     # Ge (Ge-74)
    33:  (74.9220,  100.00, "3/2"),   # As (As-75)
    34:  (78.9710,  49.61,  "0"),     # Se (Se-80)
    35:  (79.9040,  50.69,  "3/2"),   # Br (Br-79)
    36:  (83.7980,  57.00,  "0"),     # Kr (Kr-84)
    37:  (85.4680,  72.17,  "3/2"),   # Rb (Rb-85)
    38:  (87.6200,  82.58,  "0"),     # Sr (Sr-88)
    39:  (88.9060,  100.00, "1/2"),   # Y  (Y-89)
    40:  (91.2240,  51.45,  "0"),     # Zr (Zr-90)
    41:  (92.9060,  100.00, "9/2"),   # Nb (Nb-93)
    42:  (95.9500,  24.13,  "0"),     # Mo (Mo-98)
    43:  (98.0000,  0.00,   "9/2"),   # Tc (Tc-99, synth)
    44:  (101.0700, 31.55,  "0"),     # Ru (Ru-102)
    45:  (102.9100, 100.00, "1/2"),   # Rh (Rh-103)
    46:  (106.4200, 27.33,  "0"),     # Pd (Pd-106)
    47:  (107.8700, 51.84,  "1/2"),   # Ag (Ag-107)
    48:  (112.4100, 28.73,  "0"),     # Cd (Cd-114)
    49:  (114.8200, 95.71,  "9/2"),   # In (In-115)
    50:  (118.7100, 32.58,  "0"),     # Sn (Sn-120)
    51:  (121.7600, 57.21,  "5/2"),   # Sb (Sb-121)
    52:  (127.6000, 33.80,  "0"),     # Te (Te-130)
    53:  (126.9000, 100.00, "5/2"),   # I  (I-127)
    54:  (131.2900, 26.89,  "0"),     # Xe (Xe-132)
    55:  (132.9100, 100.00, "7/2"),   # Cs (Cs-133)
    56:  (137.3300, 71.70,  "0"),     # Ba (Ba-138)
    57:  (138.9100, 99.91,  "7/2"),   # La (La-139)
    58:  (140.1200, 88.45,  "0"),     # Ce (Ce-140)
    59:  (140.9100, 100.00, "5/2"),   # Pr (Pr-141)
    60:  (144.2400, 27.20,  "0"),     # Nd (Nd-142)
    61:  (145.0000, 0.00,   "7/2"),   # Pm (Pm-145, synth)
    62:  (150.3600, 26.74,  "0"),     # Sm (Sm-152)
    63:  (151.9600, 47.81,  "5/2"),   # Eu (Eu-153)
    64:  (157.2500, 24.84,  "0"),     # Gd (Gd-158)
    65:  (158.9300, 100.00, "3/2"),   # Tb (Tb-159)
    66:  (162.5000, 28.18,  "0"),     # Dy (Dy-164)
    67:  (164.9300, 100.00, "7/2"),   # Ho (Ho-165)
    68:  (167.2600, 33.50,  "0"),     # Er (Er-166)
    69:  (168.9300, 100.00, "1/2"),   # Tm (Tm-169)
    70:  (173.0500, 31.83,  "0"),     # Yb (Yb-174)
    71:  (174.9700, 97.41,  "7/2"),   # Lu (Lu-175)
    72:  (178.4900, 35.08,  "0"),     # Hf (Hf-180)
    73:  (180.9500, 99.99,  "7/2"),   # Ta (Ta-181)
    74:  (183.8400, 30.64,  "0"),     # W  (W-184)
    75:  (186.2100, 62.60,  "5/2"),   # Re (Re-187)
    76:  (190.2300, 41.02,  "0"),     # Os (Os-192)
    77:  (192.2200, 62.70,  "3/2"),   # Ir (Ir-193)
    78:  (195.0800, 33.83,  "1/2"),   # Pt (Pt-195)
    79:  (196.9700, 100.00, "3/2"),   # Au (Au-197)
    80:  (200.5900, 29.86,  "0"),     # Hg (Hg-202)
    81:  (204.3800, 70.48,  "1/2"),   # Tl (Tl-205)
    82:  (207.2000, 52.40,  "0"),     # Pb (Pb-208)
    83:  (208.9800, 100.00, "9/2"),   # Bi (Bi-209)
    84:  (209.0000, 0.00,   "0"),     # Po (Po-209, synth)
    85:  (210.0000, 0.00,   "9/2"),   # At (At-210, synth)
    86:  (222.0000, 0.00,   "0"),     # Rn (Rn-222, synth)
    87:  (223.0000, 0.00,   "3/2"),   # Fr (Fr-223, synth)
    88:  (226.0000, 0.00,   "0"),     # Ra (Ra-226, synth)
    89:  (227.0000, 0.00,   "3/2"),   # Ac (Ac-227, synth)
    90:  (232.0400, 100.00, "0"),     # Th (Th-232)
    91:  (231.0400, 100.00, "3/2"),   # Pa (Pa-231)
    92:  (238.0300, 99.27,  "0"),     # U  (U-238)
    93:  (237.0000, 0.00,   "5/2"),   # Np (Np-237, synth)
    94:  (244.0000, 0.00,   "0"),     # Pu (Pu-244, synth)
    95:  (243.0000, 0.00,   "5/2"),   # Am (Am-243, synth)
    96:  (247.0000, 0.00,   "7/2"),   # Cm (Cm-247, synth)
    97:  (247.0000, 0.00,   "3/2"),   # Bk (Bk-247, synth)
    98:  (251.0000, 0.00,   "9/2"),   # Cf (Cf-251, synth)
    99:  (252.0000, 0.00,   "5"),     # Es (Es-252, synth)
    100: (257.0000, 0.00,   "7/2"),   # Fm (Fm-257, synth)
    101: (258.0000, 0.00,   "0"),     # Md (Md-258, synth)
    102: (259.0000, 0.00,   "0"),     # No (No-259, synth)
    103: (266.0000, 0.00,   "9/2"),   # Lr (Lr-266, synth)
    104: (267.0000, 0.00,   "0"),     # Rf (Rf-267, synth)
    105: (268.0000, 0.00,   "0"),     # Db (Db-268, synth)
    106: (269.0000, 0.00,   "0"),     # Sg (Sg-269, synth)
    107: (270.0000, 0.00,   "0"),     # Bh (Bh-270, synth)
    108: (277.0000, 0.00,   "0"),     # Hs (Hs-277, synth)
    109: (278.0000, 0.00,   "0"),     # Mt (Mt-278, synth)
    110: (281.0000, 0.00,   "0"),     # Ds (Ds-281, synth)
    111: (282.0000, 0.00,   "0"),     # Rg (Rg-282, synth)
    112: (285.0000, 0.00,   "0"),     # Cn (Cn-285, synth)
    113: (286.0000, 0.00,   "0"),     # Nh (Nh-286, synth)
    114: (289.0000, 0.00,   "0"),     # Fl (Fl-289, synth)
    115: (290.0000, 0.00,   "0"),     # Mc (Mc-290, synth)
    116: (293.0000, 0.00,   "0"),     # Lv (Lv-293, synth)
    117: (294.0000, 0.00,   "0"),     # Ts (Ts-294, synth)
    118: (294.0000, 0.00,   "0"),     # Og (Og-294, synth)
}

def get_atom_props(z: int):
    """
    Returns (mass, abundance_percent, nuclear_spin) for a given atomic number Z.
    Abundance/Spin refer to the most abundant isotope.
    """
    return ATOM_PROPS.get(z, (0.0, 0.0, "UNSPECIFIED"))

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
    
    # Number of Hydrogens
    num_hs_vals = [_safe_map_lookup(x_map["num_hs"], i, 0) for i in num_hs_idx]

    # [INSERTION 1 START: Initialize accumulators]
    total_mass = 0.0
    spin_counts = Counter()
    high_abundance_atoms = 0
    # [INSERTION 1 END]

    elem_syms = []
    for z, h_count in zip(atomic_nums, num_hs_vals):
        try:
            z_int = int(z)
            h_int = int(h_count)
        except Exception:
            z_int = 0
            h_int = 0
            
        elem_syms.append(atomic_number_to_symbol(z_int))

        # 1. Add Mass of the Heavy Atom
        props = ATOM_PROPS.get(z_int, (0.0, 0.0, "UNSPECIFIED"))
        total_mass += props[0]
        
        # 2. Add Mass of attached Hydrogens (Standard H mass = 1.008)
        total_mass += (h_int * 1.008)

        # Check abundance
        if props[1] >= 99.0:
            high_abundance_atoms += 1
            
        # Count spins (heavy atom only)
        s_val = props[2]
        if s_val != "UNSPECIFIED" and s_val != "0":
            spin_counts[s_val] += 1
        # [INSERTION 2 END]

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

    # Generate SMILES
    smiles_str = pyg_to_smiles(data, x_map, e_map)

    # -------- Compose card --------
    lines.append(f"SMILES: {smiles_str}")
    lines.append(f"atoms_total: {n}")
    lines.append(f"elements: {elem_str}")

    # [INSERTION 3 START: Append new features to the output lines]
    lines.append(f"molecular_weight: {total_mass:.6f}")
    lines.append(f"high_natural_abundance_atoms: {high_abundance_atoms}")
    if spin_counts:
        # Sort spins for deterministic output
        spins_str = "; ".join([f"{k}:{v}" for k, v in sorted(spin_counts.items())])
        lines.append(f"nuclear_spins: {spins_str}")
    else:
        lines.append("nuclear_spins: NONE")
    # [INSERTION 3 END]

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
        
        # removed to save some tokens
        # # Bucket degrees for readability
        # deg_bucket = Counter()
        # for d, c in degree_counts.items():
        #     if d <= 4:
        #         deg_bucket[str(d)] += c
        #     else:
        #         deg_bucket["5+"] += c
        # lines.append(f"degree_hist: {_format_counter(deg_bucket, keys=['0','1','2','3','4','5+'], max_items=6)}")
        # lines.append(f"num_hs_hist: {_format_counter(num_hs_counts, keys=[str(i) for i in range(0, 9)], max_items=9)}")
        
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
        for g in tqdm(graphs, desc="Adding Molecule Cards.."):
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
