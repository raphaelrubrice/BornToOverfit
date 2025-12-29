#!/usr/bin/env python3
"""
Generate embeddings for molecular descriptions using various Transformer architectures.
Supports:
  - BERT/RoBERTa (Encoder-only, uses [CLS])
  - GPT/Llama/Qwen (Decoder-only, uses Last Token)
  - T5/BioT5 (Encoder-Decoder, uses Mean Pooling)
"""

import pickle
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, T5EncoderModel
from tqdm import tqdm
from pathlib import Path
import os
from argparse import ArgumentParser

# Set verbosity to avoid excessive warnings
import transformers
transformers.logging.set_verbosity_error()

def intelligent_pooling(outputs, inputs, config):
    """
    Dynamically extracts the best sentence embedding based on model architecture.
    """
    last_hidden_state = outputs.last_hidden_state
    attention_mask = inputs['attention_mask']
    model_type = config.model_type.lower()

    # 1. BERT / RoBERTa / DeBERTa / MPNet (Encoder-only)
    # Strategy: Use the first token ([CLS])
    if any(k in model_type for k in ["bert", "roberta", "mpnet", "electra", "deberta"]):
        return last_hidden_state[:, 0, :]

    # 2. GPT / Llama / Qwen / Mistral / Falcon (Causal Decoder-only)
    # Strategy: Use the last non-padding token
    elif any(k in model_type for k in ["gpt", "llama", "mistral", "qwen", "falcon"]):
        # sequence_lengths is the index of the last token (len - 1)
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_state.shape[0]
        # Select the vector at the last position for each batch element
        return last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]

    # 3. T5 / BioT5 (Encoder-Decoder Encoders)
    # Strategy: Mean Pooling (Average of all non-padding tokens)
    # Note: T5 does not have a [CLS] token.
    elif any(k in model_type for k in ["t5", "mt5"]):
        # Expand mask to match hidden dim: (B, Seq) -> (B, Seq, Hidden)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        
        # Sum embeddings, ignoring padded tokens
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        # Avoid division by zero
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return sum_embeddings / sum_mask

    # Fallback: Default to Mean Pooling if unknown
    else:
        print(f"Warning: Unknown model type '{model_type}'. Defaulting to Mean Pooling.")
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--base_model",
        type=str,
        default="bert-base-uncased",
        help="HF model name (e.g. bert-base-uncased, Qwen/Qwen2-0.5B-Instruct, mistralai/Mistral-7B-v0.1)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for embedding generation (default: 32)"
    )
    parser.add_argument("-f_data", default="data_baseline/data", type=str, help="Folder containing source .pkl graph files")
    parser.add_argument("-f", default="data_baseline/data", type=str, help="Output folder for embeddings")

    args = parser.parse_args()
    data_folder = args.f_data
    folder = args.f

    # =========================================================
    # CONFIG & PATHS
    # =========================================================
    file_path = Path(os.path.abspath(__file__))
    parent_folder = file_path.parent

    # Resolving paths relative to the script location
    data_path = parent_folder.parent / data_folder
    base_path = parent_folder.parent / folder
    os.makedirs(str(base_path), exist_ok=True)
    
    # Configuration
    MAX_TOKEN_LENGTH = 128

    print(f"Loading model: {args.base_model}")
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    
    # Set Pad Token if missing (common in GPT/Llama)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Model
    # Use T5EncoderModel for T5 family to avoid loading the decoder
    if "t5" in args.base_model.lower():
        model = T5EncoderModel.from_pretrained(
            args.base_model,
            output_hidden_states=True,
            return_dict=True
        )
    else:
        model = AutoModel.from_pretrained(
            args.base_model,
            output_hidden_states=True,
            return_dict=True
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    print(f"Model loaded on: {device}")

    # =========================================================
    # PROCESSING LOOP
    # =========================================================
    for split in ['train', 'validation']:
        print(f"\nProcessing {split}...")
        
        # Load graphs from pkl file
        pkl_path = str(data_path / f'{split}_graphs.pkl')
        if not os.path.exists(pkl_path):
            print(f"Skipping {split}, file not found: {pkl_path}")
            continue

        print(f"Loading from {pkl_path}...")
        with open(pkl_path, 'rb') as f:
            graphs = pickle.load(f)
        print(f"Loaded {len(graphs)} graphs")
        
        ids = []
        embeddings = []

        descriptions = [g.description for g in graphs]
        graph_ids = [g.id for g in graphs]

        # Batch Processing
        for i in tqdm(range(0, len(graphs), args.batch_size),
                      total=(len(graphs) + args.batch_size - 1) // args.batch_size):
            
            batch_text = descriptions[i:i + args.batch_size]
            batch_ids = graph_ids[i:i + args.batch_size]

            # Tokenize
            inputs = tokenizer(
                batch_text,
                return_tensors='pt',
                truncation=True,
                max_length=MAX_TOKEN_LENGTH,
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
            
            # --- INTELLIGENT POOLING ---
            seq_embedding = intelligent_pooling(outputs, inputs, model.config)
            # ---------------------------

            ids.extend(batch_ids)
            embeddings.extend(seq_embedding.cpu().numpy())
        
        # Save to CSV
        result = pd.DataFrame({
            'ID': ids,
            'embedding': [','.join(map(str, emb.tolist())) for emb in embeddings]
        })
        
        output_path = str(base_path / f'{split}_embeddings.csv')
        result.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")

    print("\nGeneration Complete!")