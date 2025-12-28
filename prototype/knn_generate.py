#!/usr/bin/env python3
"""
knn_generate.py

kNN + LLM generation with GRPO fine-tuning (TRL).

Key design choices:
- Retrieval/prompting uses your existing GNN embedding space (MolGNN) + train text embeddings.
- Base generative model: a light modern HF instruct model (<= 1B params) by default:
    "Qwen/Qwen2-0.5B-Instruct"
- Fine-tuning: TRL's GRPOTrainer (Group Relative Policy Optimization).
- Reward: heavy penalty if not exact match, plus informative shaping using BLEU-F1 + BERTScore(roberta-base).

This file contains:
  - finetune_base_model(...) : GRPO post-training on a kNN-prompt-to-description task
  - generate_desc(...)       : batch generation on validation/test
  - a CLI entrypoint to run finetuning and/or generation.

Dependencies:
  pip install -U transformers datasets accelerate trl peft sacrebleu bert-score

Notes:
- GRPO is online RL; reward functions must be efficient. BERTScore is expensive; this prototype uses
  batched computation and suggests small batch sizes.
- This is a prototype. For serious training runs: use LoRA, bf16/fp16, gradient checkpointing,
  and consider running reward metrics on a subset or replacing BERTScore with a learned reward model.

"""

import os
import re
import json
from dataclasses import dataclass
from pathlib import Path
from argparse import ArgumentParser
from typing import Dict, List, Tuple, Optional

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    GenerationConfig,
    Trainer,
    TrainingArguments,
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

import sys, os
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
from data_baseline.train_gcn import MolGNN, load_gnn_from_checkpoint

_EVAL_AVAILABLE = True
try:
    from data_baseline.retrieval_answer import evaluate_retrieval_text_metrics  # :contentReference[oaicite:0]{index=0}
except Exception:
    _EVAL_AVAILABLE = False

# --------------------------------------------------------------------------------------
# Prompting utilities
# --------------------------------------------------------------------------------------
def build_prompt(query_card, neighbor_descs: List[str], neighbor_cards: List[str]) -> str:
    lines = []
    lines.append("Here are the descriptions of the K nearest molecules to the query molecule from nearest to furthest.")
    for d in neighbor_descs:
        lines.append(d)
    lines.append("Based on previous descriptions, generate the accurate description of the query molecule.")
    return "\n".join(lines)

# Here are Molecule Card and Description pairs for the K nearest neighbors to the query molecule in training data ....
# Molecule Card Neighbor 1
# Description Neighbor 1
# ...
# Now, here is the Molecule Card of the query molecule. Generate the accurate description of this molecule.
# Molecule Card Query
def build_prompt(query_card: str, neighbor_descs: List[str], neighbor_cards: List[str]) -> str:
    """
    New prompting approach:
      - Provide (card, description) pairs for each neighbor
      - Provide the query molecule card
      - Explicit instruction to treat MOL_FEATURES as constraints and neighbors as examples
    """
    lines: List[str] = []
    lines.append(
        "You are given molecule 'info cards' and descriptions for K nearest neighbor molecules from training data."
    )
    lines.append("Use them as examples and evidence, but follow the query molecule card as the factual constraint.")
    lines.append("Do not assert elements, charge sign, aromaticity, or ring presence that contradict the query card.")
    lines.append("")

    lines.append("[NEIGHBORS]")
    for i, (c, d) in enumerate(zip(neighbor_cards, neighbor_descs), start=1):
        lines.append(f"Neighbor {i} - Molecule Card:")
        lines.append(c.strip() if c else "[MOL_FEATURES]\nunknown\n[/MOL_FEATURES]")
        lines.append(f"Neighbor {i} - Description:")
        lines.append((d or "").strip())
        lines.append("")
    lines.append("[/NEIGHBORS]")
    lines.append("")

    lines.append("Query Molecule Card:")
    lines.append(query_card.strip() if query_card else "[MOL_FEATURES]\nunknown\n[/MOL_FEATURES]")
    lines.append("")
    lines.append("Task: Give an accurate, factual and concise description of the query molecule in the same style as the training descriptions.")
    return "\n".join(lines)

def prompt_as_chat(prompt_text: str) -> List[Dict[str, str]]:
    """
    Return a chat-style prompt in the format used by instruct chat models in TRL docs:
      [{"role": "user", "content": "..."}]
    """
    return [{"role": "user", "content": prompt_text}]


# --------------------------------------------------------------------------------------
# GNN encoding + KNN retrieval
# --------------------------------------------------------------------------------------
@torch.no_grad()
def encode_graphs(model: MolGNN, graph_pkl: str, device: str, batch_size: int = 64) -> Tuple[torch.Tensor, List[str]]:
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


def knn_topk(
    query_embs_norm: torch.Tensor,
    train_embs_norm: torch.Tensor,
    k: int,
    mask_self: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      top_idx: [Nq, k]
      top_sims: [Nq, k]
    mask_self:
      Optional boolean mask [Nq, Nt] where True positions are disallowed (set sim=-inf).
      Used for leave-one-out on training.
    """
    sims = query_embs_norm @ train_embs_norm.t()  # [Nq, Nt]
    if mask_self is not None:
        sims = sims.masked_fill(mask_self, float("-inf"))

    k_eff = min(k, sims.size(1))
    top_sims, top_idx = torch.topk(sims, k=k_eff, dim=-1, largest=True, sorted=True)
    return top_idx, top_sims


# --------------------------------------------------------------------------------------
# Rewards: BLEU-F1 + BERTScore(roberta-base) + exact match penalty
# --------------------------------------------------------------------------------------
_WS_RE = re.compile(r"\s+")


def _normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = _WS_RE.sub(" ", s)
    return s


def bleu_f1_corpus(preds: List[str], refs: List[str], tokenize: str = "13a") -> float:
    """
    BLEU-F1 in [0, 1].
      precision proxy: BLEU(preds, refs)
      recall proxy:    BLEU(refs, preds)
    """
    preds = [p.lower() for p in preds]
    refs = [r.lower() for r in refs]
    bleu_p = sacrebleu.corpus_bleu(preds, [refs], tokenize=tokenize).score
    bleu_r = sacrebleu.corpus_bleu(refs, [preds], tokenize=tokenize).score
    denom = (bleu_p + bleu_r)
    bleu_f1 = (2.0 * bleu_p * bleu_r / denom) if denom > 0 else 0.0
    return float(bleu_f1 / 100.0)


@torch.no_grad()
def bertscore_f1_roberta_base(
    preds: List[str],
    refs: List[str],
    device: str,
    batch_size: int = 32,
) -> float:
    """
    Mean BERTScore F1 in [0, 1], using roberta-base.
    """
    P, R, F1 = bertscore(
        cands=preds,
        refs=refs,
        model_type="roberta-base",
        lang="en",
        device=device,
        batch_size=batch_size,
        idf=False,
        verbose=False,
        rescale_with_baseline=False,
    )
    return float(F1.mean().item())

def bleu_f1_sentence(pred: str, ref: str, tokenize: str = "13a") -> float:
    """
    Per-sample BLEU-F1 in [0, 1].
      precision proxy: sentence_bleu(pred, [ref])
      recall proxy:    sentence_bleu(ref, [pred])
    """
    pred_l = (pred or "").lower()
    ref_l = (ref or "").lower()

    bleu_p = sacrebleu.sentence_bleu(pred_l, [ref_l], tokenize=tokenize).score
    bleu_r = sacrebleu.sentence_bleu(ref_l, [pred_l], tokenize=tokenize).score

    denom = bleu_p + bleu_r
    bleu_f1 = (2.0 * bleu_p * bleu_r / denom) if denom > 0 else 0.0
    return float(bleu_f1 / 100.0)

class CachedBERTScorer:
    """
    Cache RoBERTa model/tokenizer to avoid re-loading BERTScore at every RL step.
    Computes per-sample BERTScore F1.
    """
    def __init__(self, device: str, batch_size: int = 32):
        from bert_score import BERTScorer

        self.device = device
        self.batch_size = batch_size

        # BERTScorer internally caches the model/tokenizer after construction
        self.scorer = BERTScorer(
            model_type="roberta-base",
            lang="en",
            device=device,
            batch_size=batch_size,
            idf=False,
            rescale_with_baseline=False,
        )

    @torch.no_grad()
    def score_f1(self, preds: List[str], refs: List[str]) -> List[float]:
        """
        Returns per-sample BERTScore F1 in [0, 1].
        """
        P, R, F1 = self.scorer.score(preds, refs)
        return F1.detach().cpu().tolist()

def make_reward_fn(
    device: str,
    bleu_weight: float = 0.45,
    bert_weight: float = 0.55,
    exact_bonus: float = 1.0,
    exact_penalty: float = 1.0,
    bert_batch_size: int = 32,
):
    """
    GRPO reward with:
      - per-sample BLEU-F1
      - per-sample BERTScore F1
      - cached RoBERTa model
      - exact-match shaping

    This guarantees non-zero reward variance within GRPO groups.
    """
    bert_scorer = CachedBERTScorer(device=device, batch_size=bert_batch_size)

    def reward_fn(completions, reference, **kwargs):
        preds = [c[0]["content"] for c in completions]
        refs = list(reference)

        n = min(len(preds), len(refs))
        preds = preds[:n]
        refs = refs[:n]

        # Exact match (per sample)
        exact = [
            1 if _normalize_text(p) == _normalize_text(r) and len(_normalize_text(r)) > 0 else 0
            for p, r in zip(preds, refs)
        ]

        # Per-sample metrics
        bleu_scores = [bleu_f1_sentence(p, r) for p, r in zip(preds, refs)]
        bert_scores = bert_scorer.score_f1(preds, refs)

        rewards = []
        for ex, b, bf in zip(exact, bleu_scores, bert_scores):
            em_term = exact_bonus if ex == 1 else -exact_penalty
            shaped = bleu_weight * b + bert_weight * bf
            rewards.append(float(em_term + shaped))

        # Stabilization
        return [max(-2.0, min(2.0, r)) for r in rewards]

    return reward_fn


# --------------------------------------------------------------------------------------
# Build RL dataset: prompts from KNN, references from ground truth
# --------------------------------------------------------------------------------------
@torch.no_grad()
def build_knn_prompt_dataset(
    gnn_model: MolGNN,
    train_graphs: str,
    train_emb_csv: str,
    device: str,
    k: int,
    encode_batch_size: int,
    max_samples: Optional[int] = None,
    leave_one_out: bool = True,
) -> Dataset:
    """
    Dataset columns:
      - prompt:    chat-style messages (list[{"role","content"}])
      - reference: ground-truth description (string)

    Uses TRAIN split for RL prompts with leave-one-out KNN retrieval.
    Incorporates mol_card (query + neighbors) in prompts.
    """
    # Load train text embeddings (targets of GNN)
    train_id2emb = load_id2emb(train_emb_csv)
    train_ids = list(train_id2emb.keys())
    train_embs = torch.stack([train_id2emb[i] for i in train_ids]).to(device)
    train_embs = F.normalize(train_embs, dim=-1)

    # Load descriptions + mol_cards
    train_id2desc = load_descriptions_from_graphs(train_graphs)
    train_id2card = load_mol_cards_from_graphs(train_graphs)

    # Encode training graphs with GNN
    train_gnn_embs, train_ids_ordered = encode_graphs(
        gnn_model, train_graphs, device=device, batch_size=encode_batch_size
    )

    # Align indices for leave-one-out
    id_to_trainemb_index = {tid: idx for idx, tid in enumerate(train_ids)}

    mask_self = None
    if leave_one_out:
        n_q = len(train_ids_ordered)
        n_t = len(train_ids)
        mask_self = torch.zeros((n_q, n_t), dtype=torch.bool, device=train_gnn_embs.device)
        for i, qid in enumerate(train_ids_ordered):
            j = id_to_trainemb_index.get(qid, None)
            if j is not None:
                mask_self[i, j] = True

    top_idx, _top_sims = knn_topk(
        query_embs_norm=train_gnn_embs,
        train_embs_norm=train_embs,
        k=k,
        mask_self=mask_self,
    )
    top_idx = top_idx.cpu().tolist()

    prompts: List[List[Dict[str, str]]] = []
    references: List[str] = []

    n = len(train_ids_ordered)
    if max_samples is not None:
        n = min(n, int(max_samples))

    for i in range(n):
        qid = train_ids_ordered[i]

        neighbor_ids = [train_ids[j] for j in top_idx[i]]
        neighbor_descs = [train_id2desc.get(nid, "") for nid in neighbor_ids]
        neighbor_cards = [train_id2card.get(nid, "") for nid in neighbor_ids]

        query_card = train_id2card.get(qid, "")

        ptxt = build_prompt(query_card, neighbor_descs, neighbor_cards)
        prompts.append(prompt_as_chat(ptxt))
        references.append(train_id2desc.get(qid, ""))

    return Dataset.from_dict({"prompt": prompts, "reference": references})

# --------------------------------------------------------------------------------------
# Supervised Fine Tuning (SFT) on the KNN dataset
# --------------------------------------------------------------------------------------

def _render_prompt_for_sft(tokenizer, prompt_messages: List[Dict[str, str]]) -> str:
    """
    Convert a chat-style prompt (list of {role, content}) into a plain text string
    for supervised fine-tuning. Uses the tokenizer chat template if available; otherwise
    concatenates messages.
    """
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        # No generation prompt here; SFT will append target separately
        return tokenizer.apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,
            tokenize=False,
        )
    # Fallback: simple linearization
    parts = []
    for m in prompt_messages:
        role = (m.get("role", "user") or "user").upper()
        parts.append(f"{role}: {m.get('content','')}")
    parts.append("ASSISTANT:")
    return "\n".join(parts)


def sft_finetune_on_knn(
    base_model_name_or_path: str,
    output_dir: str,
    train_dataset: Dataset,  # expects columns: "prompt" (chat messages), "reference" (string)
    device: str,
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
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
) -> str:
    """
    Supervised fine-tuning (SFT) stage on the KNN prompt dataset.
    Produces an intermediate model directory to be used as the starting policy for RL.

    Works for:
      - decoder-only (CausalLM) models
      - encoder-decoder (Seq2SeqLM) models
    """
    os.makedirs(output_dir, exist_ok=True)

    config = AutoConfig.from_pretrained(base_model_name_or_path)
    torch_dtype = torch.bfloat16 if bf16 else (torch.float16 if fp16 else torch.float32)
    device_map = "auto" if torch.cuda.is_available() else None

    if config.is_encoder_decoder:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" if config.is_encoder_decoder else "left"

    if use_lora:
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="SEQ_2_SEQ_LM" if config.is_encoder_decoder else "CAUSAL_LM",
            target_modules="all-linear",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    # ----------------------------
    # Tokenization: build training examples
    # ----------------------------
    def _tokenize_row(ex):
        prompt_msgs = ex["prompt"]
        ref = ex["reference"] or ""

        if config.is_encoder_decoder:
            # Seq2Seq: encode prompt as input, reference as target
            prompt_text = _render_prompt_for_sft(tokenizer, prompt_msgs)
            model_inputs = tokenizer(
                prompt_text,
                max_length=max_prompt_length,
                truncation=True,
            )
            labels = tokenizer(
                                text_target=ref,
                                max_length=max_completion_length,
                                truncation=True,
                            )["input_ids"]
            model_inputs["labels"] = labels
            return model_inputs

        # CausalLM: encode (prompt + reference) and mask prompt tokens from loss
        prompt_text = _render_prompt_for_sft(tokenizer, prompt_msgs)
        full_text = prompt_text + " " + ref

        prompt_ids = tokenizer(
            prompt_text,
            max_length=max_prompt_length,
            truncation=True,
            add_special_tokens=False,
        )["input_ids"]

        full = tokenizer(
            full_text,
            max_length=max_prompt_length + max_completion_length,
            truncation=True,
            add_special_tokens=True,
        )

        labels = full["input_ids"].copy()
        # mask the prompt portion
        cut = min(len(prompt_ids), len(labels))
        for i in range(cut):
            labels[i] = -100

        full["labels"] = labels
        return full

    tokenized = train_dataset.map(
                                _tokenize_row,
                                remove_columns=train_dataset.column_names,  # IMPORTANT: drop prompt/reference entirely
                            )

    # ----------------------------
    # Collator (pad inputs + labels)
    # ----------------------------
    if config.is_encoder_decoder:
        # Correctly pads labels and sets label_pad_token_id to -100.
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            label_pad_token_id=-100,
        )
    else:
        # Decoder-only: pads input_ids/attention_mask. Labels already contain -100 for prompt tokens,
        # but they still need padding to the batch max length; pad with -100 to avoid loss on pads.
        data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding=True,
            return_tensors="pt",
        )

        # Wrap to also pad labels consistently with -100
        def _causal_collate(features):
            batch = data_collator(features)

            if "labels" in batch:
                # DataCollatorWithPadding will NOT pad labels; do it here.
                labels = [f["labels"] for f in features]
                max_len = max(len(l) for l in labels)
                padded = [l + [-100] * (max_len - len(l)) for l in labels]
                batch["labels"] = torch.tensor(padded, dtype=torch.long)

            return batch

        data_collator = _causal_collate

    # ----------------------------
    # Train
    # ----------------------------
    args = TrainingArguments(
        output_dir=output_dir,
        max_steps=num_train_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        logging_steps=20,
        save_steps=200,
        save_total_limit=2,
        bf16=bf16,
        fp16=(fp16 and not bf16),
        report_to=[],
        seed=seed,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    
    # If LoRA was used, merge adapter into the base model and save a full model
    if use_lora:
        # PeftModel supports merge_and_unload()
        model = model.merge_and_unload()

    model.save_pretrained(output_dir)      # saves config.json + model weights
    tokenizer.save_pretrained(output_dir)
    return output_dir

# --------------------------------------------------------------------------------------
# RL fine-tuning
# --------------------------------------------------------------------------------------
def finetune_base_model(
    base_model_name_or_path: str,
    output_dir: str,
    train_dataset: Dataset,
    device: str,
    rl_algo: str = "ppo",
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    max_prompt_length: int = 1024,
    max_completion_length: int = 192,
    num_train_steps: int = 500,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    num_generations: int = 4,  # GRPO only
    learning_rate: float = 5e-6,
    bf16: bool = False,
    fp16: bool = True,
    seed: int = 42,
    bert_reward_batch_size: int = 16,
) -> str:
    """
    Fine-tune with GRPO or PPO (user-selectable).

    Notes:
      - GRPO path expects TRL GRPOTrainer behavior.
      - PPO path uses TRL PPOTrainer and requires a value head (TRL wrapper).
      - For PPO, we linearize chat-style prompts to plain text (model-agnostic).
    """
    os.makedirs(output_dir, exist_ok=True)

    rl_algo = (rl_algo or "grpo").strip().lower()
    if rl_algo not in {"grpo", "ppo"}:
        raise ValueError(f"Unsupported rl_algo={rl_algo}. Expected 'grpo' or 'ppo'.")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    config = AutoConfig.from_pretrained(base_model_name_or_path)

    torch_dtype = torch.bfloat16 if bf16 else (torch.float16 if fp16 else torch.float32)
    device_map = "auto" if torch.cuda.is_available() else None

    # ----------------------------
    # Load model(s)
    # ----------------------------
    if rl_algo == "ppo":
        if config.is_encoder_decoder:
            base = AutoModelForSeq2SeqLM.from_pretrained(
                base_model_name_or_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
            ref_base = AutoModelForSeq2SeqLM.from_pretrained(
                base_model_name_or_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )

            # Apply LoRA BEFORE wrapping with value head
            if use_lora:
                lora_cfg = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type="SEQ_2_SEQ_LM",
                    target_modules="all-linear",
                )
                base = get_peft_model(base, lora_cfg)
                base.print_trainable_parameters()

            model = AutoModelForSeq2SeqLMWithValueHead(base)
            ref_model = AutoModelForSeq2SeqLMWithValueHead(ref_base)

        else:
            base = AutoModelForCausalLM.from_pretrained(
                base_model_name_or_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
            ref_base = AutoModelForCausalLM.from_pretrained(
                base_model_name_or_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )

            if use_lora:
                lora_cfg = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules="all-linear",
                )
                base = get_peft_model(base, lora_cfg)
                base.print_trainable_parameters()

            model = AutoModelForCausalLMWithValueHead(base)
            ref_model = AutoModelForCausalLMWithValueHead(ref_base)
    else:
        # GRPO (no value head)
        if config.is_encoder_decoder:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                base_model_name_or_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name_or_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )

    # ----------------------------
    # Optional LoRA
    # ----------------------------
    if use_lora:
        # For PPO, LoRA was already applied to `base` BEFORE wrapping with value head.
        # Applying LoRA again to the TRL ValueHead wrapper breaks PEFT expectations.
        if rl_algo == "grpo":
            lora_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="SEQ_2_SEQ_LM" if config.is_encoder_decoder else "CAUSAL_LM",
                target_modules="all-linear",
            )
            model = get_peft_model(model, lora_cfg)
            model.print_trainable_parameters()

    # ----------------------------
    # Reward function (reuse existing)
    # ----------------------------
    reward_fn = make_reward_fn(device=device, bert_batch_size=bert_reward_batch_size)

    # ----------------------------
    # Train
    # ----------------------------
    if rl_algo == "grpo":
        training_args = GRPOConfig(
            output_dir=output_dir,
            learning_rate=learning_rate,
            max_steps=num_train_steps,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_generations=num_generations,
            max_prompt_length=max_prompt_length,
            max_completion_length=max_completion_length,
            bf16=bf16,
            fp16=fp16 if not bf16 else False,
            logging_steps=10,
            save_steps=100,
            seed=seed,
            report_to=[],
        )

        trainer = GRPOTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            reward_funcs=reward_fn,
        )

        trainer.train()
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        return output_dir

    # ----------------------------
    # PPO path
    # ----------------------------
    def _prompt_to_text(p):
        if isinstance(p, str):
            return p
        if isinstance(p, list):
            # list[{"role","content"}] -> plain text
            parts = []
            for m in p:
                role = (m.get("role", "user") or "user").upper()
                content = m.get("content", "")
                parts.append(f"{role}: {content}")
            parts.append("ASSISTANT:")
            return "\n".join(parts)
        return str(p)

    # Ensure PPO sees plain text prompts
    if "prompt" in train_dataset.column_names:
        train_dataset = train_dataset.map(lambda ex: {"prompt": _prompt_to_text(ex["prompt"])})
    else:
        raise ValueError("PPO requires a 'prompt' column in train_dataset.")

    if "reference" not in train_dataset.column_names:
        raise ValueError("PPO path expects a 'reference' column for reward computation.")

    ppo_cfg = PPOConfig(
        batch_size=per_device_train_batch_size,
        mini_batch_size=max(1, per_device_train_batch_size // max(1, gradient_accumulation_steps)),
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        seed=seed,
    )

    def _collate(batch):
        prompts = [b["prompt"] for b in batch]
        refs = [b["reference"] for b in batch]
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_length,
        )
        enc["reference_text"] = refs  # keep raw strings for reward
        return enc

    trainer = PPOTrainer(
        config=ppo_cfg,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=train_dataset,
        data_collator=_collate,
    )

    gen_kwargs = dict(
        max_new_tokens=max_completion_length,
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    step = 0
    for batch in trainer.dataloader:
        if step >= num_train_steps:
            break

        query_tensors = batch["input_ids"].to(trainer.accelerator.device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(trainer.accelerator.device)

        response_tensors = trainer.generate(
            query_tensors,
            attention_mask=attention_mask,
            **gen_kwargs,
        )

        responses_dec = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        refs = batch["reference_text"]

        # Adapt to existing reward_fn signature: completions is list[list[{role,content}]]
        completions = [[{"role": "assistant", "content": r}] for r in responses_dec]
        reward_vals = reward_fn(completions=completions, reference=refs)
        rewards = [torch.tensor(r, device=trainer.accelerator.device) for r in reward_vals]

        trainer.step(query_tensors, response_tensors, rewards)
        step += 1

    trainer.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir


# --------------------------------------------------------------------------------------
# Generation on validation/test: retrieval -> prompt -> batch generate
# --------------------------------------------------------------------------------------
@torch.no_grad()
def generate_desc(
    gnn_model: MolGNN,
    llm_dir_or_name: str,
    train_graphs: str,
    query_graphs: str,
    train_emb_csv: str,
    device: str,
    k: int,
    out_csv: str,
    encode_batch_size: int = 64,
    gen_batch_size: int = 8,
    max_new_tokens: int = 192,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    evaluate: bool = False,
    ) -> pd.DataFrame:
    # Load train pool (descriptions + embeddings + mol_cards)
    train_id2desc = load_descriptions_from_graphs(train_graphs)
    train_id2card = load_mol_cards_from_graphs(train_graphs)

    train_id2emb = load_id2emb(train_emb_csv)
    train_ids = list(train_id2emb.keys())
    train_embs = torch.stack([train_id2emb[i] for i in train_ids]).to(device)
    train_embs = F.normalize(train_embs, dim=-1)

    # Load query mol_cards (from pickle if present; else compute fallback)
    query_id2card = load_mol_cards_from_graphs(query_graphs)

    # Encode query
    query_embs, query_ids = encode_graphs(gnn_model, query_graphs, device=device, batch_size=encode_batch_size)

    # Retrieve KNN
    top_idx, top_sims = knn_topk(query_embs, train_embs, k=k)
    top_idx = top_idx.cpu().tolist()
    top_sims = top_sims.cpu().tolist()

    prompts_text: List[str] = []
    nn_ids_all: List[List[str]] = []
    sims_all: List[List[float]] = []

    for i in range(len(query_ids)):
        qid = query_ids[i]

        nn_ids = [train_ids[j] for j in top_idx[i]]
        nn_descs = [train_id2desc.get(nid, "") for nid in nn_ids]
        nn_cards = [train_id2card.get(nid, "") for nid in nn_ids]

        query_card = query_id2card.get(qid, "")

        prompts_text.append(build_prompt(query_card, nn_descs, nn_cards))
        nn_ids_all.append(nn_ids)
        sims_all.append([float(s) for s in top_sims[i]])

    # Load LLM
    tokenizer = AutoTokenizer.from_pretrained(llm_dir_or_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        llm_dir_or_name,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    llm.eval()

    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Batch generation
    generations: List[str] = []
    for start in range(0, len(prompts_text), gen_batch_size):
        batch_prompts = prompts_text[start:start + gen_batch_size]

        # For instruct chat models, apply chat template if available
        if hasattr(tokenizer, "apply_chat_template"):
            batch_msgs = [prompt_as_chat(p) for p in batch_prompts]
            input_ids = tokenizer.apply_chat_template(
                batch_msgs,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(llm.device)
            attention_mask = (input_ids != tokenizer.pad_token_id).long()
            outputs = llm.generate(input_ids=input_ids, attention_mask=attention_mask, generation_config=gen_cfg)
            # Strip the prompt portion
            gen_only = outputs[:, input_ids.shape[1]:]
            texts = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
        else:
            enc = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(llm.device)
            outputs = llm.generate(**enc, generation_config=gen_cfg)
            gen_only = outputs[:, enc["input_ids"].shape[1]:]
            texts = tokenizer.batch_decode(gen_only, skip_special_tokens=True)

        generations.extend([t.strip() for t in texts])

    # Write CSV
    rows = []
    for i, qid in enumerate(query_ids):
        row = {
            "ID": qid,
            "Prompt": prompts_text[i],
            "generated_description": generations[i],
        }
        for j in range(len(nn_ids_all[i])):
            row[f"NN{j+1}"] = nn_ids_all[i][j]
            row[f"SIM{j+1}"] = sims_all[i][j]
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(str(Path(out_csv).parent), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved generations to: {out_csv}")

    df.to_csv(out_csv, index=False)
    print(f"Saved generations to: {out_csv}")

    # Optional evaluation (intended for validation split where references exist)
    if evaluate:
        if not _EVAL_AVAILABLE:
            print("WARNING: Evaluation requested but retrieval_answer.evaluate_retrieval_text_metrics is not available.")
        else:
            query_id2desc = load_descriptions_from_graphs(query_graphs)
            metrics_path = str(Path(out_csv).with_suffix(".metrics.json"))
            eval_df = df[["ID", "generated_description"]].rename(columns={"generated_description": "description"})
            evaluate_retrieval_text_metrics(
                results_df=eval_df,
                test_id2desc=query_id2desc,
                device=device,
                save_path=metrics_path,
            )

    return df


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def main():
    parser = ArgumentParser()
    parser.add_argument("-f_data", default="data_baseline/data", type=str, help="Folder containing *graphs.pkl files")
    parser.add_argument("-f", default="data_baseline/data", type=str, help="Folder containing embeddings/model checkpoint")
    parser.add_argument("--k", default=3, type=int)
    parser.add_argument("--encode_batch_size", default=128, type=int)
    parser.add_argument("--max_train_samples", default=None, type=int, help="Limit RL dataset size for quick prototyping")

    # LLM
    parser.add_argument(
        "--base_llm",
        default="QizhiPei/biot5-plus-base",
        type=str,
        help="HF model name/path (<=1B recommended).",
    )
    parser.add_argument("--out_llm_dir", default="knn_llm", type=str, help="Where to save the fine-tuned model")

    # Actions
    parser.add_argument("--do_finetune", action="store_true", help="Run GRPO fine-tuning")
    parser.add_argument("--do_generate", action="store_true", help="Run generation on a split")
    parser.add_argument("--do_sft", action="store_true", help="Run SFT on the KNN dataset before RL")
    
    # SFT params
    parser.add_argument("--sft_steps", default=800, type=int)
    parser.add_argument("--sft_lr", default=2e-5, type=float)

    # Generation split
    parser.add_argument("--split", default="test", choices=["validation", "val", "test"])
    parser.add_argument("--gen_batch_size", default=8, type=int)
    parser.add_argument("--max_new_tokens", default=192, type=int)

    parser.add_argument(
        "--rl_algo",
        default=None,
        choices=["grpo", "ppo"],
        help="RL algorithm: grpo or ppo (seq2seq-friendly).",
    )
    # RL params (prototype defaults)
    parser.add_argument("--steps", default=300, type=int)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--per_device_bs", default=2, type=int)
    parser.add_argument("--grad_accum", default=4, type=int)
    parser.add_argument("--num_generations", default=4, type=int)
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA adapters (recommended)")
    parser.add_argument("--no_lora", action="store_true", help="Disable LoRA")

    # Reward speed
    parser.add_argument("--bert_reward_bs", default=16, type=int, help="BERTScore batch size inside reward")

    args = parser.parse_args()

    # Resolve paths
    file_path = Path(os.path.abspath(__file__))
    parent_folder = file_path.parent

    data_path = parent_folder.parent / args.f_data
    base_path = parent_folder.parent / args.f

    TRAIN_GRAPHS = str(data_path / "train_graphs.pkl")
    VAL_GRAPHS = str(data_path / "validation_graphs.pkl")
    TEST_GRAPHS = str(data_path / "test_graphs.pkl")

    MODEL_PATH = str(base_path / "model_checkpoint.pt")
    TRAIN_EMB_CSV = str(base_path / "train_embeddings.csv")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    for p in [TRAIN_GRAPHS, MODEL_PATH, TRAIN_EMB_CSV]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file: {p}")

    # Load GNN
    train_id2emb = load_id2emb(TRAIN_EMB_CSV)
    emb_dim = len(next(iter(train_id2emb.values())))

    print(f"Loading GNN checkpoint: {MODEL_PATH}")
    gnn = load_gnn_from_checkpoint(
        model_path=MODEL_PATH,
        device=device,
        x_map=x_map,
        e_map=e_map,
    )
    gnn.eval()

    # Fine-tune
    llm_dir = args.out_llm_dir
    use_lora = args.use_lora and (not args.no_lora)

    if args.do_finetune:
        print("Building KNN dataset from TRAIN split (leave-one-out KNN)...")
        knn_ds = build_knn_prompt_dataset(
            gnn_model=gnn,
            train_graphs=TRAIN_GRAPHS,
            train_emb_csv=TRAIN_EMB_CSV,
            device=device,
            k=args.k,
            encode_batch_size=args.encode_batch_size,
            max_samples=args.max_train_samples,
            leave_one_out=True,
        )
        print(f"KNN dataset size: {len(knn_ds)}")

        llm_dir = args.base_llm

        if args.do_sft:
            sft_dir = str(base_path / (args.out_llm_dir + "_sft"))
            print(f"Running SFT -> {sft_dir}")
            llm_dir = sft_finetune_on_knn(
                base_model_name_or_path=args.base_llm,
                output_dir=sft_dir,
                train_dataset=knn_ds,
                device=device,
                use_lora=use_lora,
                num_train_steps=args.sft_steps,
                learning_rate=args.sft_lr,
                per_device_train_batch_size=args.per_device_bs,
                gradient_accumulation_steps=args.grad_accum,
                max_prompt_length=1024,
                max_completion_length=args.max_new_tokens,
                bf16=False,
                fp16=torch.cuda.is_available(),
            )
            print(f"SFT model saved to: {llm_dir}")

        if args.rl_algo is not None:
            print("Running RL fine-tuning...")
            llm_dir = finetune_base_model(
                base_model_name_or_path=start_model,  # NOTE: SFT output if --do_sft, else base_llm
                output_dir=str((base_path / args.out_llm_dir)),
                train_dataset=knn_ds,
                device=device,
                rl_algo=args.rl_algo,
                use_lora=use_lora,
                num_train_steps=args.steps,
                learning_rate=args.lr,
                per_device_train_batch_size=args.per_device_bs,
                gradient_accumulation_steps=args.grad_accum,
                num_generations=args.num_generations,
                bert_reward_batch_size=args.bert_reward_bs,
                bf16=False,
                fp16=torch.cuda.is_available(),
            )
            print(f"Saved fine-tuned model to: {llm_dir}")

    # Generate
    if args.do_generate:
        if args.split in ("validation", "val"):
            query_graphs = VAL_GRAPHS
            out_csv = str(base_path / f"val_knn_gen_k{args.k}.csv")
            eval_flag = True
        else:
            query_graphs = TEST_GRAPHS
            out_csv = str(base_path / f"test_knn_gen_k{args.k}.csv")
            eval_flag = False

        if not os.path.exists(query_graphs):
            raise FileNotFoundError(f"Missing split graphs: {query_graphs}")

        llm_path = llm_dir
        if not os.path.isabs(llm_path):
            candidate = base_path / llm_path
            if candidate.exists():
                llm_path = str(candidate)

        print(f"Generating with model: {llm_path}")
        generate_desc(
            gnn_model=gnn,
            llm_dir_or_name=llm_path,
            train_graphs=TRAIN_GRAPHS,
            query_graphs=query_graphs,
            train_emb_csv=TRAIN_EMB_CSV,
            device=device,
            k=args.k,
            out_csv=out_csv,
            encode_batch_size=args.encode_batch_size,
            gen_batch_size=args.gen_batch_size,
            max_new_tokens=args.max_new_tokens,
            evaluate=eval_flag,
        )

    if (not args.do_finetune) and (not args.do_generate):
        print("Nothing to do. Use --do_finetune and/or --do_generate.")


if __name__ == "__main__":
    main()