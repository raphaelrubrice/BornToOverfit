#!/usr/bin/env python3
"""
knn_generate.py

kNN + LLM generation with GRPO fine-tuning (TRL).
Optimized for speed by caching token lengths.
"""

import os
import re
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from argparse import ArgumentParser
from typing import Dict, List, Tuple, Optional, Any
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    GenerationConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq, 
    DataCollatorWithPadding
)

from peft import LoraConfig, get_peft_model

from trl import (
    GRPOTrainer,
    GRPOConfig,
    PPOTrainer,
    PPOConfig,
    AutoModelForCausalLMWithValueHead,
    AutoModelForSeq2SeqLMWithValueHead,
)

import sacrebleu
from bert_score import score as bertscore

import sys
# To ensure the custom package is found
path_to_repo = "/".join(os.path.abspath(__file__).split("/")[:-2])
if path_to_repo not in sys.path:
    sys.path.append(path_to_repo)

from data_baseline.data_utils import (
    load_id2emb,
    load_descriptions_from_graphs,
    PreprocessedGraphDataset,
    collate_fn,
    make_mol_repr,
    load_mol_cards_from_graphs,
    x_map,
    e_map
)
from data_baseline.train_gcn import MolGNN, load_molgnn_from_checkpoint
from data_baseline.train_gcn_v3_gps import MolGNN_GPS, load_molgnn_gps_from_checkpoint

SUPPORTED_GNNS = {"MolGNN":load_molgnn_from_checkpoint, 
                "MolGNN_GPS":load_molgnn_gps_from_checkpoint}

_EVAL_AVAILABLE = True
try:
    from data_baseline.retrieval_answer import evaluate_retrieval_text_metrics 
except Exception:
    _EVAL_AVAILABLE = False

# --------------------------------------------------------------------------------------
# Optimization: Token Length Cache
# --------------------------------------------------------------------------------------
class TokenLengthCache:
    """
    Pre-computes and caches token lengths for strings to avoid 
    repeated CPU-bound tokenizer calls during prompt construction.
    """
    def __init__(self, tokenizer, data_dict: Dict[str, str], description: str = "Pre-computing lengths"):
        self.cache = {}
        self.tokenizer = tokenizer
        
        # Batch tokenization is much faster than sequential
        keys = list(data_dict.keys())
        texts = [str(data_dict[k]) for k in keys]
        
        # We use a simple approximation or batch encode. 
        # To be perfectly safe and fast, we iterate but simply.
        print(f"[{description}] Caching token counts for {len(texts)} items...")
        
        # Fast batch encoding (no padding needed for count, just raw IDs)
        # We do it in chunks to avoid memory spikes
        batch_size = 1000
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_enc = tokenizer(batch_texts, add_special_tokens=False, verbose=False)
            
            for j, ids in enumerate(batch_enc["input_ids"]):
                original_key = keys[i + j]
                self.cache[original_key] = len(ids)

    def get_len(self, key: str) -> int:
        return self.cache.get(key, 0)
    
    def estimate_text_len(self, text: str) -> int:
        """Fallback for non-cached text"""
        return len(self.tokenizer.encode(text, add_special_tokens=False))

# --------------------------------------------------------------------------------------
# Helper: Dynamic Context Length Detection
# --------------------------------------------------------------------------------------
def get_safe_context_length(model_name_or_path: str, cap: int = 4096) -> int:
    try:
        config = AutoConfig.from_pretrained(model_name_or_path)
        possible_limits = [
            getattr(config, "max_position_embeddings", None),
            getattr(config, "n_positions", None),
            getattr(config, "seq_length", None),
            getattr(config, "max_seq_len", None), 
        ]
        valid_limits = [x for x in possible_limits if x is not None and isinstance(x, int)]
        limit = max(valid_limits) if valid_limits else 1024 
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
            if hasattr(tokenizer, "model_max_length") and tokenizer.model_max_length < 1e9:
                limit = max(limit, tokenizer.model_max_length)
        except Exception:
            pass

        print(f" [Context Detection] Model native limit: {limit}")
        if limit > cap:
            print(f" [Context Detection] Capping context to {cap} for memory safety.")
            return cap
        return limit
    except Exception as e:
        print(f" [Context Detection] Warning: Could not auto-detect. Error: {e}")
        return 1024

def prompt_as_chat(prompt_text: str) -> List[Dict[str, str]]:
    return [{"role": "user", "content": prompt_text}]

# --------------------------------------------------------------------------------------
# GNN encoding + KNN retrieval
# --------------------------------------------------------------------------------------
GNN_LOADING_FUNCS = [load_molgnn_from_checkpoint,load_molgnn_gps_from_checkpoint]

def load_gnn_from_checkpoint(*args, **kwargs):
    for func in GNN_LOADING_FUNCS:
        try:
            return func(*args, **kwargs)
        except:
            pass
    raise ValueError(f"All attempted loading methods failed.")

@torch.no_grad()
def encode_graphs(model: nn.Module, graph_pkl: str, device: str, batch_size: int = 64) -> Tuple[torch.Tensor, List[str]]:
    ds = PreprocessedGraphDataset(graph_pkl)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    all_embs: List[torch.Tensor] = []
    all_ids: List[str] = []

    for graphs in dl:
        graphs = graphs.to(device)
        mol_emb = model(graphs)
        mol_emb = F.normalize(mol_emb, dim=-1)
        all_embs.append(mol_emb)

        bs = graphs.num_graphs
        start_idx = len(all_ids)
        all_ids.extend(ds.ids[start_idx:start_idx + bs])

    embs = torch.cat(all_embs, dim=0)
    return embs, all_ids

# def knn_topk(query_embs_norm, train_embs_norm, k, mask_self=None):
#     sims = query_embs_norm @ train_embs_norm.t()
#     if mask_self is not None:
#         sims = sims.masked_fill(mask_self, float("-inf"))
#     k_eff = min(k, sims.size(1))
#     top_sims, top_idx = torch.topk(sims, k=k_eff, dim=-1, largest=True, sorted=True)
#     return top_idx, top_sims

def knn_topk(
    query_embs_norm: torch.Tensor,
    train_embs_norm: torch.Tensor,
    k: int,
    mask_self: Optional[torch.Tensor] = None,
    batch_size: int = 4096  # Process 4096 queries at a time
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batched KNN to prevent OOM / Memory Fragmentation on large datasets.
    """
    num_queries = query_embs_norm.size(0)
    all_top_idx = []
    all_top_sims = []

    # Process queries in chunks
    for start in range(0, num_queries, batch_size):
        end = min(start + batch_size, num_queries)
        
        # 1. Compute similarities for just this batch [Batch_Size, N_train]
        q_batch = query_embs_norm[start:end] 
        sims = q_batch @ train_embs_norm.t() 
        
        # 2. Apply mask if needed (Slice the mask for this batch)
        if mask_self is not None:
            mask_batch = mask_self[start:end]
            sims = sims.masked_fill(mask_batch, float("-inf"))

        # 3. Top-K for this batch (results are small tensors)
        k_eff = min(k, sims.size(1))
        batch_sims, batch_idx = torch.topk(sims, k=k_eff, dim=-1, largest=True, sorted=True)
        
        # 4. Move small results to CPU immediately to free GPU memory
        all_top_idx.append(batch_idx.cpu())
        all_top_sims.append(batch_sims.cpu())
        
        # Explicitly delete intermediates to help the allocator
        del sims, batch_idx, batch_sims, q_batch
    
    # Concatenate all results
    top_idx = torch.cat(all_top_idx, dim=0)
    top_sims = torch.cat(all_top_sims, dim=0)
    
    return top_idx, top_sims

def compute_similarity_thresholds(sims_tensor: torch.Tensor) -> List[float]:
    flat_sims = sims_tensor.view(-1).float()
    quantiles = torch.tensor([0.2, 0.4, 0.6, 0.8], device=flat_sims.device)
    thresholds = torch.quantile(flat_sims, quantiles)
    return thresholds.cpu().tolist()

def get_closeness_tag(sim: float, thresholds: List[float]) -> str:
    q20, q40, q60, q80 = thresholds
    if sim >= q80: return "[VERY_CLOSE]"
    elif sim >= q60: return "[CLOSE]"
    elif sim >= q40: return "[AVERAGE]"
    elif sim >= q20: return "[DISTANT]"
    else: return "[VERY_DISTANT]"

def save_thresholds(thresholds: List[float], path: str):
    with open(path, 'w') as f:
        json.dump(thresholds, f)
    print(f"Saved similarity thresholds to {path}: {thresholds}")

def load_thresholds(path: str) -> List[float]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Thresholds file not found at {path}. Run training first or ensure file exists.")
    with open(path, 'r') as f:
        t = json.load(f)
    print(f"Loaded similarity thresholds from {path}: {t}")
    return t

# --------------------------------------------------------------------------------------
# Updated Prompt Builder (Split Strategy)
# --------------------------------------------------------------------------------------
def fit_neighbors_fast(
    tokenizer,
    query_card: str,
    struct_neighbors: List[Dict],
    func_neighbors: List[Dict],
    desc_cache,
    card_cache,
    max_length: int = 1024
) -> str:
    """
    Constructs a Dynamic Prompt that adapts to data quality (Hybrid Strategy).
    Handles edge cases (No Structure / No Function) by changing the System Instruction.
    """
    
    # --- 1. Analyze Data Availability ---
    has_structure = len(struct_neighbors) > 0
    has_function = len(func_neighbors) > 0
    
    # --- 2. Dynamic System Header Construction ---
    
    # Base Role
    intro = "You are an expert chemist. Generate a factual description of the [QUERY MOLECULE] strictly based on its feature card."
    
    # Condition A: Structural Templates
    if has_structure:
        struct_instr = (
            "You are provided with **[STRUCTURAL TEMPLATES]** (similar backbones). "
            "Use them to determine the sentence structure, IUPAC nomenclature style, and scaffold description. "
            "However, you MUST update specific atoms/groups to match the [QUERY MOLECULE] Card."
        )
    else:
        struct_instr = (
            "**WARNING: NO STRUCTURAL MATCHES FOUND.** "
            "The [QUERY MOLECULE] has a unique scaffold. "
            "Do NOT hallucinate a template. Construct the description **from scratch** based strictly on the SMILES and Card data."
        )

    # Condition B: Functional References
    if has_function:
        func_instr = (
            "You are provided with **[FUNCTIONAL REFERENCES]** (biologically similar). "
            "Use them ONLY to identify potential biomedical roles (e.g., 'anti-bacterial', 'metabolite') "
            "if they align with the query's structure."
        )
    else:
        func_instr = (
            "**NOTE: NO FUNCTIONAL CONTEXT AVAILABLE.** "
            "Describe the chemical structure and properties only. "
            "Do NOT infer specific biological applications (like 'anti-cancer') unless explicitly evident from the structure."
        )

    # Global Constraints
    constraints = (
        "**Strict Constraints:**\n"
        "1. The [QUERY MOLECULE] Card is the ABSOLUTE GROUND TRUTH.\n"
        "2. If a neighbor contradicts the Query (e.g., different halogen, different ring count), IGNORE the neighbor.\n"
        "3. Never mention 'Neighbor 1' or 'Template' in the final output."
    )
    
    # Combine Header
    system_header = f"{intro}\n\n**Context Guidelines:**\n* {struct_instr}\n* {func_instr}\n\n{constraints}"

    # --- 3. Construct Body ---
    
    # Query Block (Primacy Effect - put it early)
    query_block = (
        "\n\n[QUERY MOLECULE]\n"
        f"Card: {query_card.strip() if query_card else 'UNKNOWN'}\n"
    )
    
    footer = "**Generate the description strictly adhering to the Guidelines:**"

    # Estimate Overhead
    # Headers + Tags + Footer
    static_text = system_header + query_block + "\n[STRUCTURAL TEMPLATES]\n[FUNCTIONAL REFERENCE]\n[/CONTEXT]\n" + footer
    base_len = len(tokenizer.encode(static_text, add_special_tokens=False))
    remaining_budget = max_length - base_len - 20 # Safety Buffer

    # --- 4. Fill Context (Structure First) ---
    struct_text_list = []
    if has_structure:
        struct_text_list.append("\n[STRUCTURAL TEMPLATES]")
        for i, n in enumerate(struct_neighbors, 1):
            score_display = f"{n['score']:.2f}"
            c = n['card']
            d = n['desc']
            nid = n['id']
            
            # Cost: Card + Desc + Overhead
            c_len = card_cache.get_len(nid)
            d_len = desc_cache.get_len(nid)
            overhead = 25 
            
            if (c_len + d_len + overhead) <= remaining_budget:
                entry = (
                    f"\n> Template {i} (Tanimoto: {score_display}):\n"
                    f"Card: {c.strip()}\n"
                    f"Description: {d.strip()}"
                )
                struct_text_list.append(entry)
                remaining_budget -= (c_len + d_len + overhead)
            else:
                break
    
    # --- 5. Fill Context (Function Second) ---
    func_text_list = []
    # Only add function if we have budget AND (we have structure OR we have enough space for at least 1)
    if has_function and remaining_budget > 60:
        func_text_list.append("\n\n[FUNCTIONAL REFERENCE]")
        for i, n in enumerate(func_neighbors, 1):
            d = n['desc']
            nid = n['id']
            
            # Cost: Desc ONLY (Card Hidden)
            d_len = desc_cache.get_len(nid)
            overhead = 20
            
            if (d_len + overhead) <= remaining_budget:
                entry = (
                    f"\n> Ref {i} (Biomedical Only):\n"
                    f"Description: {d.strip()}"
                )
                func_text_list.append(entry)
                remaining_budget -= (d_len + overhead)
            else:
                break

    # --- 6. Final Assembly ---
    context_section = ""
    if struct_text_list or func_text_list:
        context_section = "\n[CONTEXT]" + "".join(struct_text_list) + "".join(func_text_list) + "\n[/CONTEXT]\n"
    
    return f"{system_header}{query_block}{context_section}{footer}"


# --------------------------------------------------------------------------------------
# Rewards & Utils
# --------------------------------------------------------------------------------------
_WS_RE = re.compile(r"\s+")
def _normalize_text(s: str) -> str:
    if s is None: return ""
    s = str(s).strip().lower()
    s = _WS_RE.sub(" ", s)
    return s

def bleu_f1_sentence(pred: str, ref: str, tokenize: str = "13a") -> float:
    pred_l = (pred or "").lower()
    ref_l = (ref or "").lower()
    bleu_p = sacrebleu.sentence_bleu(pred_l, [ref_l], tokenize=tokenize).score
    bleu_r = sacrebleu.sentence_bleu(ref_l, [pred_l], tokenize=tokenize).score
    denom = bleu_p + bleu_r
    bleu_f1 = (2.0 * bleu_p * bleu_r / denom) if denom > 0 else 0.0
    return float(bleu_f1 / 100.0)

class CachedBERTScorer:
    def __init__(self, device: str, batch_size: int = 32):
        from bert_score import BERTScorer
        self.scorer = BERTScorer(model_type="roberta-base", lang="en", device=device, batch_size=batch_size, idf=False, rescale_with_baseline=False)

    @torch.no_grad()
    def score_f1(self, preds: List[str], refs: List[str]) -> List[float]:
        P, R, F1 = self.scorer.score(preds, refs)
        return F1.detach().cpu().tolist()

import re
from collections import Counter

# --------------------------------------------------------------------------------------
# Advanced Reward Engineering: Dynamic + Constraint Based
# --------------------------------------------------------------------------------------

class ChemistryReward:
    def __init__(self, device: str, bert_batch_size: int = 32):
        self.bert_scorer = CachedBERTScorer(device=device, batch_size=bert_batch_size)
        
        # --- NEGATIVE CONSTRAINTS (The "Safety Net") ---
        # These catch what BLEU/BERT miss: factual contradictions with the Input Card.
        self.halogens_re = re.compile(r"fluor|chlor|brom|iod", re.IGNORECASE)
        self.phospho_re = re.compile(r"phosph", re.IGNORECASE)
        self.sulfur_re = re.compile(r"sulf|thio", re.IGNORECASE)
        self.aromatic_re = re.compile(r"benz|phenyl|cycl|pyrid|furan|pyrrol|imida|indol", re.IGNORECASE)

    def check_card_constraints(self, pred: str, prompt: str) -> float:
        """
        Parses the MolCard from the prompt.
        If the Card says "0 Halogens", and the model generates "Chlorine", 
        we apply a MASSIVE penalty. BLEU/BERT cannot detect this.
        """
        penalty = 0.0
        pred_lower = pred.lower()

        # Extract 'elements' and 'counts' from the prompt's MolCard
        try:
            query_part = prompt.split("[QUERY MOLECULE]")[-1]
        except IndexError:
            return 0.0 

        # Regex helpers to find values in the MolCard string
        def get_val(key):
            m = re.search(rf"{key}:\s*(\d+)", query_part)
            return int(m.group(1)) if m else None

        # 1. No Halogens Allowed?
        halogens = get_val("halogens_total")
        if halogens == 0 and self.halogens_re.search(pred_lower):
            penalty -= 1.5 # Heavy penalty

        # 2. No Phosphorus? (Check elements line)
        if "P=" not in query_part and self.phospho_re.search(pred_lower):
            penalty -= 1.5

        # 3. No Sulfur?
        if "S=" not in query_part and self.sulfur_re.search(pred_lower):
            penalty -= 1.5

        # 4. No Aromatics?
        aromatic = get_val("aromatic_atoms")
        if aromatic == 0 and self.aromatic_re.search(pred_lower):
            penalty -= 1.0

        return penalty

    def compute_length_penalty(self, pred: str, ref: str) -> float:
        # Simple Gaussian window to prevent "short-caption" gaming
        l_pred = len(pred.split())
        l_ref = len(ref.split())
        if l_ref == 0: return 0.0
        
        ratio = l_pred / l_ref
        dev = abs(ratio - 1.0)
        
        # If length is within +/- 20% of reference, give bonus
        if dev <= 0.2: return 0.5
        # Else decay linearly
        return max(-1.0, 0.5 - (dev * 2.0))

    def get_rewards(self, prompts, completions, references):
        # 1. Handle Completions (GRPO usually sends strings, PPO sends dicts)
        preds = []
        for c in completions:
            if isinstance(c, str):
                preds.append(c)
            elif isinstance(c, list): 
                # Handle [{"content": "..."}] format
                preds.append(c[0]["content"])
            else:
                preds.append(str(c))
                
        refs = list(references)
        
        # --- FIX START: Unpack Prompts from Chat Format ---
        prompt_texts = []
        for p in prompts:
            if isinstance(p, list):
                # Input is [{"role": "user", "content": "..."}]
                # We need the plain string 'content' to parse the MolCard
                parts = [m.get("content", "") for m in p]
                prompt_texts.append("\n".join(parts))
            else:
                # Input is already a string
                prompt_texts.append(str(p))
        # --- FIX END ---
        
        # 2. Semantic Foundation (BLEU + BERT)
        bert_scores = self.bert_scorer.score_f1(preds, refs)
        
        rewards = []
        # Zip over the extracted 'prompt_texts', not the raw 'prompts'
        for prompt_text, pred, ref, bert in zip(prompt_texts, preds, refs, bert_scores):
            bleu = bleu_f1_sentence(pred, ref)
            
            # Weighted Composite
            total = (
                0.5 * bert + 
                0.4 * bleu + 
                1.0 * self.check_card_constraints(pred, prompt_text) + 
                0.2 * self.compute_length_penalty(pred, ref)
            )
            rewards.append(max(-3.0, min(3.0, float(total))))
            
        return rewards

def make_reward_fn(device: str, bert_batch_size: int = 32):
    engine = ChemistryReward(device=device, bert_batch_size=bert_batch_size)

    # Note: Signature accepts 'prompts' now
    def reward_fn(prompts, completions, reference, **kwargs):
        return engine.get_rewards(prompts, completions, reference)

    return reward_fn

# --------------------------------------------------------------------------------------
# Helpers for Tanimoto
# --------------------------------------------------------------------------------------
def get_smiles_from_card(card_str: str) -> Optional[str]:
    """Extracts SMILES from the feature card string."""
    if not card_str: return None
    match = re.search(r"SMILES:\s*([^\n]+)", card_str)
    return match.group(1).strip() if match else None

def compute_fingerprints(id2card: Dict[str, str]) -> Dict[str, Any]:
    """Pre-computes Morgan Fingerprints for the dataset."""
    id2fp = {}
    for nid, card in tqdm(id2card.items(), desc="Computing Fingerprints"):
        smiles = get_smiles_from_card(card)
        if smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                # Radius 2 (ECFP4), 2048 bits
                id2fp[nid] = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            else:
                id2fp[nid] = None
        else:
            id2fp[nid] = None
    return id2fp

# --------------------------------------------------------------------------------------
# Updated KNN Builder (Hybrid Retrieval)
# --------------------------------------------------------------------------------------
@torch.no_grad()
def build_knn_prompt_dataset(
    gnn_model: nn.Module,
    train_graphs: str,
    train_emb_csv: str,
    device: str,
    k: int,
    encode_batch_size: int,
    tokenizer_or_path: str,
    thresholds: List[float],
    max_prompt_length: int = 1024,
    max_samples: Optional[int] = None,
    leave_one_out: bool = True,
    desc_key="Train",
) -> Dataset:
    
    # --- 1. Setup & Loading ---
    if isinstance(tokenizer_or_path, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_or_path, use_fast=True, trust_remote_code=True)
    else:
        tokenizer = tokenizer_or_path
        
    # Load Semantic Embeddings (GNN)
    train_id2emb = load_id2emb(train_emb_csv)
    train_ids = list(train_id2emb.keys())
    train_embs = torch.stack([train_id2emb[i] for i in train_ids]).to(device)
    train_embs = F.normalize(train_embs, dim=-1)

    # Load Text Data
    train_id2desc = load_descriptions_from_graphs(train_graphs)
    train_id2card = load_mol_cards_from_graphs(train_graphs)

    # --- 2. Phase 1: Semantic Retrieval (GNN) ---
    print("Phase 1: Computing GNN Embeddings...")
    train_gnn_embs, train_ids_ordered = encode_graphs(gnn_model, train_graphs, device=device, batch_size=encode_batch_size)

    id_to_idx = {tid: idx for idx, tid in enumerate(train_ids)} # Map ID -> DB Index

    # Create Mask for Leave-One-Out
    mask_self = None
    if leave_one_out:
        n_q = len(train_ids_ordered)
        n_t = len(train_ids)
        mask_self = torch.zeros((n_q, n_t), dtype=torch.bool, device=train_gnn_embs.device)
        for i, qid in enumerate(train_ids_ordered):
            j = id_to_idx.get(qid, None)
            if j is not None:
                mask_self[i, j] = True

    # Get GNN Neighbors (we get more than needed to filter later)
    # We want at least 1-2 functional neighbors
    top_idx_gnn, top_sims_gnn = knn_topk(train_gnn_embs, train_embs, k=k, mask_self=mask_self)
    top_idx_gnn = top_idx_gnn.cpu().tolist()
    
    # --- 3. Phase 2: Structural Retrieval (Tanimoto) ---
    print("Phase 2: Computing Tanimoto Neighborhoods...")
    
    # Pre-compute fingerprints for Database (train_ids) and Queries (train_ids_ordered)
    # Note: If train_graphs == train_emb_csv content, these sets overlap.
    db_fps_dict = compute_fingerprints(train_id2card)
    
    # Convert DB FPs to a list aligned with train_ids
    db_fps_list = [db_fps_dict.get(tid) for tid in train_ids]
    
    # Prepare Result Lists
    prompts: List[List[Dict[str, str]]] = []
    references: List[str] = []

    # Caches
    desc_cache = TokenLengthCache(tokenizer, train_id2desc, f"{desc_key} Descriptions")
    card_cache = TokenLengthCache(tokenizer, train_id2card, f"{desc_key} Cards")
    
    # Defined Ratio: 4 Structural, 1 Functional (for K=5)
    k_struct = max(1, int(k * 0.8))
    k_func = k - k_struct

    n = len(train_ids_ordered)
    if max_samples is not None:
        n = min(n, int(max_samples))

    for i in tqdm(range(n), desc="Building Hybrid Prompts"):
        qid = train_ids_ordered[i]
        q_fp = db_fps_dict.get(qid) # Query FP
        
        # --- A. Retrieve Structural Neighbors (Tanimoto) ---
        struct_neighbors = []
        if q_fp is not None:
            # Efficient Bulk Calc
            t_sims = DataStructs.BulkTanimotoSimilarity(q_fp, db_fps_list)
            
            # Mask self if needed
            if leave_one_out:
                self_idx = id_to_idx.get(qid, -1)
                if self_idx >= 0:
                    t_sims[self_idx] = -1.0
            
            # Get Top K indices (using numpy argpartition for speed or argsort)
            # We need indices of the top k_struct
            t_sims_np = np.array(t_sims)
            # argsort is ascending, take last k_struct
            top_struct_indices = np.argsort(t_sims_np)[-k_struct:][::-1]
            
            for idx in top_struct_indices:
                score = t_sims_np[idx]
                if score > 0.5:
                    nid = train_ids[idx]
                    struct_neighbors.append({
                        "id": nid,
                        "desc": train_id2desc.get(nid, ""),
                        "card": train_id2card.get(nid, ""),
                        "score": score,
                        "type": "STRUCTURAL"
                    })
        
        # --- B. Retrieve Functional Neighbors (GNN) ---
        func_neighbors = []
        gnn_indices = top_idx_gnn[i] # Pre-computed list of indices
        
        # Avoid Duplicates: Don't pick a neighbor as Functional if it's already in Structural
        struct_nids = set(n['id'] for n in struct_neighbors)
        
        for idx in gnn_indices:
            if len(func_neighbors) >= k_func:
                break
            
            nid = train_ids[idx]
            if nid not in struct_nids:
                func_neighbors.append({
                    "id": nid,
                    "desc": train_id2desc.get(nid, ""),
                    "card": train_id2card.get(nid, ""), # We might hide this later
                    "score": 0.0, # Placeholder, GNN score not strictly comparable
                    "type": "FUNCTIONAL"
                })

        # --- C. Construct Hybrid Prompt ---
        # Merge lists (Structure First)
        # We pass them to the prompter
        
        query_card = train_id2card.get(qid, "")
        
        smart_prompt_text = fit_neighbors_fast(
            tokenizer=tokenizer,
            query_card=query_card,
            struct_neighbors=struct_neighbors,
            func_neighbors=func_neighbors,
            desc_cache=desc_cache,
            card_cache=card_cache,
            max_length=max_prompt_length
        )

        prompts.append(prompt_as_chat(smart_prompt_text))
        references.append(train_id2desc.get(qid, ""))

    return Dataset.from_dict({"prompt": prompts, "reference": references})

# --------------------------------------------------------------------------------------
# SFT
# --------------------------------------------------------------------------------------
def _render_prompt_for_sft(tokenizer, prompt_messages: List[Dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(prompt_messages, add_generation_prompt=True, tokenize=False)
    parts = []
    for m in prompt_messages:
        role = (m.get("role", "user") or "user").upper()
        parts.append(f"{role}: {m.get('content','')}")
    parts.append("ASSISTANT:")
    return "\n".join(parts)

class CausalSeq2SeqTrainer(Seq2SeqTrainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Custom prediction step for Causal LMs (Decoder-Only) in Seq2SeqTrainer.
        Strips 'labels' to prevent shape mismatch errors in .generate(), 
        but preserves them for metrics.
        """
        if prediction_loss_only:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

        # 1. Separate Labels from Inputs
        # Causal models crash if 'labels' are passed to .generate() alongside input_ids
        has_labels = "labels" in inputs
        inputs_no_labels = {k: v for k, v in inputs.items() if k != "labels"}

        with torch.no_grad():
            # 2. Generate
            # We rely on model.generation_config (which we configured in sft_finetune_on_knn)
            # to handle max_new_tokens, etc. No need to pull from self.args.
            generated_tokens = model.generate(**inputs_no_labels)
            
            # Handle case where generate returns a dict (unlikely here, but safe)
            if isinstance(generated_tokens, dict):
                generated_tokens = generated_tokens["sequences"]

            # 3. Restore Labels for Metrics
            if has_labels:
                labels = inputs["labels"]
            else:
                labels = None

            # 4. Pad if necessary
            # The Trainer loop expects all batches to have the same tensor width to stack them.
            if labels is not None:
                # We use the 'generation_max_length' we explicitly passed to TrainingArguments
                # This corresponds to 'safe_max_total_len' from our main function.
                max_len = self.args.generation_max_length
                
                # Fallback just in case
                if max_len is None:
                    max_len = model.generation_config.max_length

                if max_len is not None and generated_tokens.shape[-1] < max_len:
                     generated_tokens = self._pad_tensors_to_max_len(generated_tokens, max_len)
                
        # Return tuple: (loss, logits, labels)
        # Loss is None because we are evaluating generation, not perplexity here
        return (None, generated_tokens, labels)
    
def sft_finetune_on_knn(
    base_model_name_or_path: str,
    output_dir: str,
    train_dataset: Dataset,
    device: str,
    eval_dataset: Optional[Dataset] = None,
    use_lora: bool = True,
    lora_r: int = 128,
    lora_alpha: int = None,
    lora_dropout: float = 0.05,
    max_prompt_length: int = 1024,
    max_completion_length: int = 192,
    num_train_steps: int = 800,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-5,
    bf16: bool = False,
    fp16: bool = True,
    seed: int = 42,
    patience: int = None,
    use_early_stopping: bool = False,
    neftune_noise_alpha: float = None,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    config = AutoConfig.from_pretrained(base_model_name_or_path)

    # We need a max_length that is an INT (for the Trainer) 
    # but large enough to fit the Prompt + Generation (to avoid crashes).
    safe_max_total_len = max_prompt_length + max_completion_length
    print(f"Setting generation max_length to: {safe_max_total_len} (Prompt: {max_prompt_length} + Gen: {max_completion_length})")

    # Determine standard floating point type based on arguments and hardware
    if bf16:
        dtype = torch.bfloat16
    elif fp16:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
    else:
        dtype = torch.float32

    # Set default LoRA alpha if not provided
    if lora_alpha is None:
        lora_alpha = 2 * lora_r

    # Set default patience for Early Stopping
    if patience is None:
        patience = min(5, int(0.05 * num_train_steps))

    # Initialize Model
    if config.is_encoder_decoder:
        model_cls = AutoModelForSeq2SeqLM
    else:
        model_cls = AutoModelForCausalLM

    model = model_cls.from_pretrained(
        base_model_name_or_path,
        dtype=dtype,
        device_map="auto"
    )
    model.gradient_checkpointing_enable()

    if model.generation_config is None:
        model.generation_config = GenerationConfig.from_pretrained(base_model_name_or_path)
    
    # 1. Limit the NEW tokens generated
    model.generation_config.max_new_tokens = max_completion_length

    # 2. Set the TOTAL limit (Prompt + New) to our safe integer
    model.generation_config.max_length = safe_max_total_len

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name_or_path,
        use_fast=True,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if config.is_encoder_decoder:
        tokenizer.padding_side = "right"
    else:
        tokenizer.padding_side = "left"

    # Configure LoRA
    if use_lora:
        if config.is_encoder_decoder:
            task_type = "SEQ_2_SEQ_LM"
        else:
            task_type = "CAUSAL_LM"

        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=task_type,
            target_modules="all-linear"
        )
        model = get_peft_model(model, lora_cfg)
        
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
            
        model.print_trainable_parameters()

    # --- Tokenization Logic ---
    def _tokenize(ex, is_eval=False):
        prompt_msgs = ex["prompt"]
        ref = ex["reference"] or ""
        prompt_text = _render_prompt_for_sft(tokenizer, prompt_msgs)

        if config.is_encoder_decoder or is_eval:
            model_inputs = tokenizer(
                prompt_text,
                max_length=max_prompt_length,
                truncation=True
            )
            labels = tokenizer(
                text_target=ref,
                max_length=max_completion_length,
                truncation=True
            )["input_ids"]
            model_inputs["labels"] = labels
            return model_inputs

        # CausalLM Training
        full_text = prompt_text + " " + ref + tokenizer.eos_token
        full = tokenizer(
            full_text,
            max_length=max_prompt_length + max_completion_length,
            truncation=True,
            add_special_tokens=True
        )
        
        input_ids = full["input_ids"]
        labels = input_ids.copy()

        prompt_ids = tokenizer(
            prompt_text,
            max_length=max_prompt_length,
            truncation=True,
            add_special_tokens=False
        )["input_ids"]
        
        prompt_len = len(prompt_ids)
        if tokenizer.bos_token_id and input_ids[0] == tokenizer.bos_token_id:
            prompt_len += 1

        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100
            
        full["labels"] = labels
        return full

    tokenized_train = train_dataset.map(
        lambda x: _tokenize(x, is_eval=False),
        remove_columns=train_dataset.column_names
    )

    tokenized_eval = None
    if eval_dataset is not None:
        print("Tokenizing validation set for Generation Metrics...")
        tokenized_eval = eval_dataset.map(
            lambda x: _tokenize(x, is_eval=True),
            remove_columns=eval_dataset.column_names
        )

    # --- Compute Metrics Function ---
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        if isinstance(labels, np.ndarray):
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [p.strip().lower() for p in decoded_preds]
        decoded_labels = [l.strip().lower() for l in decoded_labels]

        bleu_res = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels], lowercase=True)
        bleu4 = bleu_res.score

        try:
            P, R, F1 = bertscore(
                decoded_preds,
                decoded_labels,
                model_type="roberta-base",
                lang="en",
                device=device,
                verbose=False,
                batch_size=len(decoded_preds)
            )
            bert_f1 = F1.mean().item()
        except Exception as e:
            print(f"Warning: BERTScore computation failed: {e}")
            bert_f1 = 0.0

        final_proxy = 0.5 * (bleu4 / 100.0) + 0.5 * bert_f1

        return {
            "bleu4": bleu4,
            "bertscore_f1": bert_f1,
            "final_proxy": final_proxy
        }

    # Data Collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100
    )

    # --- Evaluation & Early Stopping Setup ---
    callbacks = []
    eval_strategy = "no"
    load_best = False
    metric_for_best = None
    
    if tokenized_eval is not None:
        eval_strategy = "steps"
        eval_steps = min(num_train_steps // 2, max(20, num_train_steps // 20))
        
        if use_early_stopping:
            print(f"Enabling Early Stopping (Patience={patience}) based on final_proxy...")
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))
            load_best = True
            metric_for_best = "eval_loss" # Trainer looks for "eval_final_proxy"
    else:
        eval_steps = 0

    if eval_strategy == "steps":
        save_strategy = "steps"
    else:
        save_strategy = "no"

    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        max_steps=num_train_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        neftune_noise_alpha=neftune_noise_alpha,
        
        # If using older transformers (<4.41), change this back to 'evaluation_strategy'
        # Strategies must match for load_best_model_at_end to work
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        save_strategy=save_strategy,
        save_steps=eval_steps,
        save_total_limit=2,
        
        # Early Stopping Config
        load_best_model_at_end=load_best,
        metric_for_best_model=metric_for_best,
        greater_is_better=False,
        
        predict_with_generate=False,
        generation_max_length=None,
        
        bf16=bf16,
        fp16=(fp16 and not bf16),
        report_to=[],
        seed=seed,
        gradient_checkpointing=True,
        remove_unused_columns=False, # Essential when using custom prompts columns
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # 9. Dynamic Trainer Selection
    if config.is_encoder_decoder:
        print("Using Standard Seq2SeqTrainer (Encoder-Decoder detected)")
        trainer_cls = Seq2SeqTrainer
    else:
        print("Using Custom CausalSeq2SeqTrainer (Decoder-Only detected)")
        trainer_cls = CausalSeq2SeqTrainer

    trainer = trainer_cls(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=callbacks,
        compute_metrics=compute_metrics if tokenized_eval else None
    )

    trainer.train()

    if use_lora:
        model = model.merge_and_unload()
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return output_dir

# --------------------------------------------------------------------------------------
# RL fine-tuning
# --------------------------------------------------------------------------------------
def rl_finetune_on_knn(
    base_model_name_or_path: str,
    output_dir: str,
    train_dataset: Dataset,
    device: str,
    rl_algo: str = "ppo",
    use_lora: bool = True,
    lora_r: int = 128,
    lora_alpha: int = None,
    lora_dropout: float = 0.05,
    max_prompt_length: int = 1024,
    max_completion_length: int = 192,
    num_train_steps: int = 500,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    num_generations: int = 4,
    learning_rate: float = 5e-6,
    bf16: bool = False,
    fp16: bool = True,
    seed: int = 42,
    bert_reward_batch_size: int = 16,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    rl_algo = (rl_algo or "grpo").strip().lower()
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    config = AutoConfig.from_pretrained(base_model_name_or_path)

    if bf16: 
        dtype = torch.bfloat16
    elif fp16: 
        dtype = torch.float16 if not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.bfloat16
    else: 
        dtype = torch.float32
    
    # Using Lora alpha = 2 * lora_r
    if lora_alpha is None:
        lora_alpha = 2 * lora_r

    if rl_algo == "ppo":
        model_cls = AutoModelForSeq2SeqLMWithValueHead if config.is_encoder_decoder else AutoModelForCausalLMWithValueHead
        base_cls = AutoModelForSeq2SeqLM if config.is_encoder_decoder else AutoModelForCausalLM
        base = base_cls.from_pretrained(base_model_name_or_path, dtype=dtype, device_map="auto")
        ref_base = base_cls.from_pretrained(base_model_name_or_path, dtype=dtype, device_map="auto")
        if use_lora:
            lora_cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias="none", task_type="SEQ_2_SEQ_LM" if config.is_encoder_decoder else "CAUSAL_LM", target_modules="all-linear")
            base = get_peft_model(base, lora_cfg)
        model = model_cls(base)
        ref_model = model_cls(ref_base)
    else:
        model_cls = AutoModelForSeq2SeqLM if config.is_encoder_decoder else AutoModelForCausalLM
        model = model_cls.from_pretrained(base_model_name_or_path, dtype=dtype, device_map="auto")
        if use_lora:
            lora_cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias="none", task_type="SEQ_2_SEQ_LM" if config.is_encoder_decoder else "CAUSAL_LM", target_modules="all-linear")
            model = get_peft_model(model, lora_cfg)

    reward_fn = make_reward_fn(device=device, bert_batch_size=bert_reward_batch_size)

    if rl_algo == "grpo":
        training_args = GRPOConfig(output_dir=output_dir, learning_rate=learning_rate, max_steps=num_train_steps, 
                        per_device_train_batch_size=per_device_train_batch_size, gradient_accumulation_steps=gradient_accumulation_steps, 
                        num_generations=num_generations, max_prompt_length=max_prompt_length, max_completion_length=max_completion_length, 
                        bf16=bf16, fp16=fp16 if not bf16 else False, logging_steps=10, save_steps=100, seed=seed, report_to=[])
        trainer = GRPOTrainer(model=model, args=training_args, train_dataset=train_dataset, reward_funcs=reward_fn)
        trainer.train()
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        return output_dir
    
    # PPO Path
    def _prompt_to_text(p):
        if isinstance(p, str): return p
        if isinstance(p, list):
            parts = []
            for m in p:
                role = (m.get("role", "user") or "user").upper()
                parts.append(f"{role}: {m.get('content','')}")
            parts.append("ASSISTANT:")
            return "\n".join(parts)
        return str(p)

    train_dataset = train_dataset.map(lambda ex: {"prompt": _prompt_to_text(ex["prompt"])})
    ppo_cfg = PPOConfig(batch_size=per_device_train_batch_size, mini_batch_size=max(1, per_device_train_batch_size // max(1, gradient_accumulation_steps)), gradient_accumulation_steps=gradient_accumulation_steps, learning_rate=learning_rate, seed=seed)

    def _collate(batch):
        prompts = [b["prompt"] for b in batch]
        refs = [b["reference"] for b in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_prompt_length)
        enc["reference_text"] = refs 
        return enc

    trainer = PPOTrainer(config=ppo_cfg, model=model, ref_model=ref_model, tokenizer=tokenizer, dataset=train_dataset, data_collator=_collate)
    gen_kwargs = dict(max_new_tokens=max_completion_length, do_sample=True, top_p=0.9, temperature=0.8, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)

    step = 0
    for batch in trainer.dataloader:
        if step >= num_train_steps: break
        query_tensors = batch["input_ids"].to(trainer.accelerator.device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None: attention_mask = attention_mask.to(trainer.accelerator.device)
        response_tensors = trainer.generate(query_tensors, attention_mask=attention_mask, **gen_kwargs)
        responses_dec = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        completions = [[{"role": "assistant", "content": r}] for r in responses_dec]
        reward_vals = reward_fn(completions=completions, reference=batch["reference_text"])
        rewards = [torch.tensor(r, device=trainer.accelerator.device) for r in reward_vals]
        trainer.step(query_tensors, response_tensors, rewards)
        step += 1

    trainer.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir

# --------------------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------------------
# Enable faster matrix multiplications on Ampere+ GPUs (RTX 30xx, A100, etc.)
torch.backends.cuda.matmul.allow_tf32 = True

@torch.no_grad()
def generate_desc(
    gnn_model,
    llm_dir_or_name: str,
    train_graphs: str,
    query_graphs: str,
    train_emb_csv: str,
    device: str,
    k: int,
    out_csv: str,
    thresholds: List[float],
    encode_batch_size: int = 64,
    gen_batch_size: int = 8,
    max_new_tokens: int = 192,
    max_length: int = 4096,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    evaluate: bool = False,
    ) -> pd.DataFrame:
    
    # ----------------------------
    # 1. Setup Model (Optimization: TF32 & BF16)
    # ----------------------------
    print(f"Loading LLM and Tokenizer from {llm_dir_or_name}...")
    
    # Auto-detect context length (reusing helper from your script)
    try:
        model_max_length = get_safe_context_length(llm_dir_or_name, cap=max_length)
    except NameError:
        model_max_length = 2048

    tokenizer = AutoTokenizer.from_pretrained(llm_dir_or_name, use_fast=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    config = AutoConfig.from_pretrained(llm_dir_or_name)
    is_seq2seq = bool(getattr(config, "is_encoder_decoder", False))
    
    # T5 (Seq2Seq) uses Right Padding for encoder; GPT (Causal) uses Left Padding
    tokenizer.padding_side = "right" if is_seq2seq else "left"

    # Dtype Selection
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
        print("Using bfloat16 for generation.")
    elif torch.cuda.is_available():
        dtype = torch.float16
        print("Using float16 for generation.")
    else:
        dtype = torch.float32

    model_cls = AutoModelForSeq2SeqLM if is_seq2seq else AutoModelForCausalLM
    llm = model_cls.from_pretrained(llm_dir_or_name, dtype=dtype, device_map="auto")
    llm.eval()
    
    # Compile model for ~20% speedup on modern PyTorch (Linux only, usually)
    if hasattr(torch, "compile") and os.name != "nt":
        try:
            print("Compiling model with torch.compile...")
            llm = torch.compile(llm)
        except Exception:
            print("Could not compile model (skipping optimization).")

    # ----------------------------
    # 2. Retrieval Phase (Standard)
    # ----------------------------
    # (Assuming helper functions exist in your scope)
    train_id2desc = load_descriptions_from_graphs(train_graphs)
    train_id2card = load_mol_cards_from_graphs(train_graphs)
    query_id2card = load_mol_cards_from_graphs(query_graphs)

    # Initialize Caches for Speed (from previous optimization)
    desc_cache = TokenLengthCache(tokenizer, train_id2desc, "Train Descriptions")
    card_cache = TokenLengthCache(tokenizer, train_id2card, "Train Cards")

    train_id2emb = load_id2emb(train_emb_csv)
    train_ids = list(train_id2emb.keys())
    train_embs = torch.stack([train_id2emb[i] for i in train_ids]).to(device)
    train_embs = F.normalize(train_embs, dim=-1)

    print("Encoding query..")
    query_embs, query_ids = encode_graphs(gnn_model, query_graphs, device=device, batch_size=encode_batch_size)

    print(f"Retrieving top {k} neighbors..")
    top_idx, top_sims = knn_topk(query_embs, train_embs, k=k)
    # thresholds = compute_similarity_thresholds(top_sims) Now using passed thresholds
    top_idx = top_idx.cpu().tolist()
    top_sims = top_sims.cpu().tolist()

    # ----------------------------
    # 3. Prompt Construction
    # ----------------------------
    # We store tuples: (length, prompt_text, qid, nn_ids, sims)
    data_buffer = []

    print("Building prompts...")
    for i in tqdm(range(len(query_ids)), desc="Constructing"):
        qid = query_ids[i]
        nn_ids = [train_ids[j] for j in top_idx[i]]
        nn_descs = [train_id2desc.get(nid, "") for nid in nn_ids]
        nn_cards = [train_id2card.get(nid, "") for nid in nn_ids]
        query_card = query_id2card.get(qid, "")
        
        current_sims = top_sims[i]
        neighbor_tags = [get_closeness_tag(s, thresholds) for s in current_sims]

        # Use the optimized prompt builder
        smart_prompt = fit_neighbors_fast(
            tokenizer=tokenizer,
            query_card=query_card,
            neighbor_ids=nn_ids,
            neighbor_descs=nn_descs,
            neighbor_cards=nn_cards,
            neighbor_tags=neighbor_tags,
            desc_cache=desc_cache,
            card_cache=card_cache,
            max_length=model_max_length
        )
        
        # Estimate length for sorting (just string length is a good enough proxy for speed)
        data_buffer.append({
            "len": len(smart_prompt), 
            "prompt": smart_prompt,
            "id": qid,
            "nn_ids": nn_ids,
            "sims": current_sims
        })

    # ----------------------------
    # 4. OPTIMIZATION: Sort by Length
    # ----------------------------
    # Sorting groups short prompts with short, long with long. 
    # This drastically reduces padding overhead in batches.
    print("Sorting data by prompt length to optimize batch padding...")
    data_buffer.sort(key=lambda x: x["len"])
    
    sorted_prompts = [x["prompt"] for x in data_buffer]
    sorted_ids = [x["id"] for x in data_buffer]
    sorted_nn_ids = [x["nn_ids"] for x in data_buffer]
    sorted_sims = [x["sims"] for x in data_buffer]

    # ----------------------------
    # 5. Batch Generation
    # ----------------------------
    print(f"Generating with Batch Size {gen_batch_size} (Greedy/Sampling, No Beams)...")
    
    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,       
        num_beams=1,  # <--- CRITICAL: Disable Beam Search for speed
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=llm.config.decoder_start_token_id if hasattr(llm.config, "decoder_start_token_id") else tokenizer.pad_token_id
    )

    use_chat_template = bool(hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None))
    generations = []

    # Iterate through the SORTED list
    for start in tqdm(range(0, len(sorted_prompts), gen_batch_size), desc="Generating"):
        batch_prompts = sorted_prompts[start : start + gen_batch_size]
        
        # Ensure left-truncation if we somehow exceed limits (unlikely with sorting + pre-check)
        tokenizer.truncation_side = 'left' 

        if use_chat_template:
            batch_msgs = [prompt_as_chat(p) for p in batch_prompts]
            inputs = tokenizer.apply_chat_template(
                batch_msgs, 
                add_generation_prompt=True, 
                tokenize=True, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=model_max_length
            )
            input_ids = inputs.to(llm.device)
        else:
            inputs = tokenizer(
                batch_prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=model_max_length
            )
            input_ids = inputs["input_ids"].to(llm.device)

        attention_mask = (input_ids != tokenizer.pad_token_id).long()

        with torch.no_grad():
            outputs = llm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=gen_cfg,
                use_cache=True 
            )

        if is_seq2seq:
            texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        else:
            # For CausalLM, slice off input
            gen_only = outputs[:, input_ids.shape[1] :]
            texts = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
        
        generations.extend([t.strip() for t in texts])

    # ----------------------------
    # 6. Re-align & Save
    # ----------------------------
    # The generations are currently in "Length-Sorted" order. 
    # We can save them as is, just keeping the IDs aligned.
    
    rows = []
    for i in range(len(sorted_ids)):
        row = { 
            "ID": sorted_ids[i], 
            "Prompt": sorted_prompts[i], 
            "generated_description": generations[i] 
        }
        for j in range(len(sorted_nn_ids[i])):
            row[f"NN{j+1}"] = sorted_nn_ids[i][j]
            row[f"SIM{j+1}"] = sorted_sims[i][j]
        rows.append(row)

    df = pd.DataFrame(rows)
    # Optional: Sort back by ID if you prefer original order
    # df = df.sort_values(by="ID") 
    
    os.makedirs(str(Path(out_csv).parent), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved full dataframe generations to: {out_csv}")

    eval_df = df[["ID", "generated_description"]].rename(columns={"generated_description": "description"})
    submit_out_csv = out_csv.replace(".csv", "_submit.csv")
    eval_df.to_csv(submit_out_csv, index=False)
    print(f"Saved submission generations to: {submit_out_csv}")

    if evaluate and _EVAL_AVAILABLE:
        # Load descriptions for evaluation
        print("Running evaluation metrics...")
        query_id2desc = load_descriptions_from_graphs(query_graphs)
        metrics_path = str(Path(out_csv).with_suffix(".metrics.json"))

        evaluate_retrieval_text_metrics(
            results_df=eval_df, 
            test_id2desc=query_id2desc, 
            device=device, 
            save_path=metrics_path
        )

    return df

# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def main():
    parser = ArgumentParser()
    parser.add_argument("-f_data", default="data_baseline/data", type=str)
    parser.add_argument("-f", default="data_baseline/data", type=str)
    parser.add_argument("--k", default=3, type=int)
    parser.add_argument("--encode_batch_size", default=128, type=int)
    parser.add_argument("--max_train_samples", default=None, type=int)
    parser.add_argument("--base_llm", default="QizhiPei/biot5-plus-base", type=str)
    parser.add_argument("--max_prompt_length", default=None, type=int)
    parser.add_argument("--out_llm_dir", default="knn_llm", type=str)

    parser.add_argument("--do_finetune", action="store_true")
    parser.add_argument("--do_generate", action="store_true")
    parser.add_argument("--do_sft", action="store_true")

    parser.add_argument("--sft_steps", default=800, type=int)
    parser.add_argument("--sft_lr", default=2e-5, type=float)
    parser.add_argument("--sft_early_stopping", action="store_true", help="Enable early stopping based on validation metrics")
    parser.add_argument("--sft_patience", default=None, help="Early Stopping patience window for SFT")
    parser.add_argument("--sft_neftune", action="store_true", help="Enable NEFTune noise alpha=5")
    parser.add_argument("--sft_lora", default=128, type=int, help="LoRA rank for SFT")

    parser.add_argument("--split", default="test", choices=["validation", "val", "test"])
    parser.add_argument("--gen_batch_size", default=4, type=int)
    parser.add_argument("--max_new_tokens", default=256, type=int)

    parser.add_argument("--rl_algo", default=None, choices=["grpo", "ppo"])
    parser.add_argument("--rl_steps", default=300, type=int)
    parser.add_argument("--lr", default=0.5e-6, type=float)
    parser.add_argument("--per_device_bs", default=2, type=int)
    parser.add_argument("--grad_accum", default=4, type=int)
    parser.add_argument("--num_generations", default=4, type=int)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--no_lora", action="store_true")
    parser.add_argument("--bert_reward_bs", default=16, type=int)
    args = parser.parse_args()

    file_path = Path(os.path.abspath(__file__))
    parent_folder = file_path.parent
    data_path = parent_folder.parent / args.f_data
    base_path = parent_folder.parent / args.f
    TRAIN_GRAPHS = str(data_path / "train_graphs.pkl")
    VAL_GRAPHS = str(data_path / "validation_graphs.pkl")
    TEST_GRAPHS = str(data_path / "test_graphs.pkl")
    MODEL_PATH = str(base_path / "model_checkpoint.pt")
    TRAIN_EMB_CSV = str(base_path / "train_embeddings.csv")
    THRESHOLDS_PATH = str(base_path / f"sim_thresholds_k{args.k}.json")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if args.max_prompt_length is not None: 
        context_len = args.max_prompt_length
    else: 
        context_len = get_safe_context_length(args.base_llm, cap=4096)

    train_id2emb = load_id2emb(TRAIN_EMB_CSV)
    print(f"Loading GNN checkpoint: {MODEL_PATH}")
    gnn = load_gnn_from_checkpoint(model_path=MODEL_PATH, device=device, x_map=x_map, e_map=e_map)
    gnn.eval()

    llm_dir = args.out_llm_dir
    use_lora = args.use_lora and (not args.no_lora)
    global_thresholds = None
    
    # ---------------------------------------------------------
    # 1. THRESHOLD STRATEGY
    # ---------------------------------------------------------
    # We need thresholds to be consistent between Train and Test.
    global_thresholds = None

    # If we are training, we MUST compute them from training data
    if args.do_finetune:
        print("Computing global similarity thresholds from Training Data...")
        
        # We need the training GNN embeddings to compute thresholds
        # (Re-using the logic from build_dataset briefly to get sims)
        train_id2emb = load_id2emb(TRAIN_EMB_CSV)
        train_ids = list(train_id2emb.keys())
        train_tgt_embs = torch.stack([train_id2emb[i] for i in train_ids]).to(device)
        train_tgt_embs = F.normalize(train_tgt_embs, dim=-1)
        
        # Encode Train Graphs
        print("Encoding training graphs for threshold calibration...")
        train_gnn_embs, _ = encode_graphs(gnn, TRAIN_GRAPHS, device=device, batch_size=args.encode_batch_size)
        
        # Leave-one-out KNN
        # (We only need the sims, not the indices, to determine distribution)
        # Note: This is fast on GPU
        sims = train_gnn_embs @ train_tgt_embs.t()
        # Mask self
        n = len(train_ids)
        mask = torch.eye(n, device=device).bool()
        sims.masked_fill_(mask, -float('inf'))
        
        # Get top k
        top_sims, _ = torch.topk(sims, k=args.k, dim=-1)
        
        # Compute & Save
        global_thresholds = compute_similarity_thresholds(top_sims)
        save_thresholds(global_thresholds, THRESHOLDS_PATH)

    # If we are NOT training but generating, we load them
    elif args.do_generate:
        try:
            global_thresholds = load_thresholds(THRESHOLDS_PATH)
        except FileNotFoundError:
            raise FileNotFoundError("Threshold file not found.")

    if args.do_finetune:
        print("Building KNN dataset (TRAIN)...")
        knn_ds = build_knn_prompt_dataset(
            gnn_model=gnn, train_graphs=TRAIN_GRAPHS, train_emb_csv=TRAIN_EMB_CSV, 
            device=device, k=args.k, encode_batch_size=args.encode_batch_size, 
            tokenizer_or_path=args.base_llm, thresholds=global_thresholds, 
            max_prompt_length=context_len, max_samples=args.max_train_samples, 
            leave_one_out=True, desc_key="Train",
        )
        
        # <--- NEW: Build Validation Dataset for Early Stopping --->
        knn_val_ds = None
        if args.do_sft and args.sft_early_stopping: 
            print("Building KNN dataset (VAL) for Early Stopping...")
            # Note: We use VAL_GRAPHS as the "query", but still retrieve from TRAIN_EMB_CSV
            knn_val_ds = build_knn_prompt_dataset(
                gnn_model=gnn, 
                train_graphs=VAL_GRAPHS, # <--- Use Val graphs here
                train_emb_csv=TRAIN_EMB_CSV, 
                device=device, k=args.k, encode_batch_size=args.encode_batch_size, 
                tokenizer_or_path=args.base_llm, thresholds=global_thresholds, 
                max_prompt_length=context_len, 
                # We don't need leave_one_out for validation since val graphs aren't in the train DB
                leave_one_out=False, desc_key="Val",
            )
        
        print("Optimizing memory for SFT...")
        
        # 1. Move GNN to CPU
        gnn.cpu()
        
        # 2. Clear Python variables holding GPU tensors
        # (train_id2emb holds text embeddings on GPU, we don't need them for SFT)
        if 'train_id2emb' in locals():
            del train_id2emb 
            
        # 3. Force PyTorch to release "Reserved" memory back to the OS
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        print(f"GPU Memory Free: {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB")
        # =========================================================
        # [END] MEMORY OPTIMIZATION BLOCK

        llm_dir = args.base_llm
        if args.do_sft:
            sft_dir = str(base_path / (args.out_llm_dir + "_sft"))
            print(f"Running SFT -> {sft_dir}")
            
            # Decide NEFTune alpha
            neft_alpha = 5.0 if args.sft_neftune else None

            llm_dir = sft_finetune_on_knn(
                base_model_name_or_path=args.base_llm, 
                output_dir=sft_dir, 
                train_dataset=knn_ds, 
                eval_dataset=knn_val_ds,
                device=device, 
                use_lora=use_lora, 
                lora_r=args.sft_lora,
                num_train_steps=args.sft_steps, 
                learning_rate=args.sft_lr, 
                per_device_train_batch_size=args.per_device_bs, 
                gradient_accumulation_steps=args.grad_accum, 
                max_prompt_length=context_len, 
                max_completion_length=args.max_new_tokens, 
                bf16=False, 
                fp16=torch.cuda.is_available(),
                use_early_stopping=args.sft_early_stopping,
                neftune_noise_alpha=neft_alpha
            )

        if args.rl_algo is not None:
            print("Running RL fine-tuning...")
            llm_dir = rl_finetune_on_knn(base_model_name_or_path=llm_dir, output_dir=str((base_path / args.out_llm_dir)), 
                                          train_dataset=knn_ds, device=device, rl_algo=args.rl_algo, use_lora=use_lora, 
                                          num_train_steps=args.rl_steps, learning_rate=args.lr, 
                                          per_device_train_batch_size=args.per_device_bs, 
                                          gradient_accumulation_steps=args.grad_accum, max_prompt_length=context_len, 
                                          num_generations=args.num_generations, bert_reward_batch_size=args.bert_reward_bs, 
                                          bf16=False, fp16=torch.cuda.is_available())

    if args.do_generate:
        if args.split in ("validation", "val"):
            query_graphs = VAL_GRAPHS
            out_csv = str(base_path / f"val_knn_gen_k{args.k}.csv")
            eval_flag = True
        else:
            query_graphs = TEST_GRAPHS
            out_csv = str(base_path / f"test_knn_gen_k{args.k}.csv")
            eval_flag = False
        llm_path = llm_dir
        if not os.path.isabs(llm_path):
            candidate = base_path / llm_path
            if candidate.exists(): llm_path = str(candidate)
        
        # [START] Bring GNN back to GPU
        gnn.to(device) 
        # [END]

        print(f"Generating with model: {llm_path}")
        generate_desc(gnn_model=gnn, llm_dir_or_name=llm_path, train_graphs=TRAIN_GRAPHS, 
                      query_graphs=query_graphs, train_emb_csv=TRAIN_EMB_CSV, device=device, 
                      k=args.k, out_csv=out_csv, thresholds=global_thresholds,
                      encode_batch_size=args.encode_batch_size, gen_batch_size=args.gen_batch_size, 
                      max_new_tokens=args.max_new_tokens, evaluate=eval_flag, max_length=context_len)

if __name__ == "__main__":
    main()