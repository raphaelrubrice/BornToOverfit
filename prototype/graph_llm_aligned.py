import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any, Union

# Try importing UMAP
try:
    import umap
except ImportError:
    print("Warning: 'umap-learn' library not found. UMAP plot will be skipped.")
    umap = None

# Try imports for PyG
try:
    from torch_geometric.data import Batch, Data
    from torch_geometric.utils import to_dense_batch, to_dense_adj
except ImportError:
    print("Warning: torch_geometric not found. Graph features require PyG.")

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoConfig,
    PreTrainedModel,
    AutoTokenizer
)

# =========================================================
# 1. Configuration & Feature Maps
# =========================================================

FEATURE_DIMS = {
    'atomic_num': 119, 'chirality': 9, 'degree': 11, 'formal_charge': 12,
    'num_hs': 9, 'num_radical_electrons': 5, 'hybridization': 8,
    'is_aromatic': 2, 'is_in_ring': 2, 'bond_type': 22, 'stereo': 7, 'is_conjugated': 2
}

@dataclass
class GraleConfig:
    hidden_dim: int = 48
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.08
    max_nodes: int = 574
    llm_model_path: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    feat_dims: Dict[str, int] = None
    def __post_init__(self):
        if self.feat_dims is None: self.feat_dims = FEATURE_DIMS

# =========================================================
# 2. RichGrALE Model
# =========================================================

class RichFeatureEmbedding(nn.Module):
    def __init__(self, config: GraleConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.node_embs = nn.ModuleDict({
            k: nn.Embedding(v, self.hidden_dim) for k, v in config.feat_dims.items() 
            if k not in ['bond_type', 'stereo', 'is_conjugated']
        })
    def forward_nodes(self, x):
        out = 0
        keys = ['atomic_num', 'chirality', 'degree', 'formal_charge', 'num_hs', 
                'num_radical_electrons', 'hybridization', 'is_aromatic', 'is_in_ring']
        for i, k in enumerate(keys):
            out += self.node_embs[k](x[..., i].long())
        return out

# class RichGrALE(nn.Module):
#     def __init__(self, config: GraleConfig):
#         super().__init__()
#         self.config = config
#         self.embedding = RichFeatureEmbedding(config)
#         self.max_dist = 20
#         self.spatial_bias = nn.Embedding(self.max_dist + 1, config.num_heads)
        
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=config.hidden_dim, nhead=config.num_heads,
#             dim_feedforward=config.hidden_dim * 4, dropout=config.dropout,
#             batch_first=True, norm_first=True
#         )
#         self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
#         self.out_norm = nn.LayerNorm(config.hidden_dim)
        
#         # Heads for AE
#         self.node_decoders = nn.ModuleDict({
#             k: nn.Linear(config.hidden_dim, v) for k, v in config.feat_dims.items()
#             if k not in ['bond_type', 'stereo', 'is_conjugated']
#         })
#         self.edge_decoder_dim = config.hidden_dim * 2
#         self.edge_classifiers = nn.ModuleDict({
#             k: nn.Sequential(nn.Linear(self.edge_decoder_dim, config.hidden_dim), nn.ReLU(), nn.Linear(config.hidden_dim, v))
#             for k, v in config.feat_dims.items() if k in ['bond_type', 'stereo', 'is_conjugated']
#         })
#         self.apply(self._init_weights)

#     def _init_weights(self, module):
#         if isinstance(module, (nn.Linear, nn.Embedding)):
#             module.weight.data.normal_(mean=0.0, std=0.02)
#         if isinstance(module, nn.Linear) and module.bias is not None:
#             module.bias.data.zero_()

#     def forward(self, batch_data):
#         x_dense, mask = to_dense_batch(batch_data.x, batch_data.batch, max_num_nodes=self.config.max_nodes)
#         h = self.embedding.forward_nodes(x_dense)
#         src_key_padding_mask = ~mask
#         z_nodes = self.encoder(h, src_key_padding_mask=src_key_padding_mask)
#         z_nodes = self.out_norm(z_nodes)
#         mask_expanded = mask.unsqueeze(-1).float()
#         z_graph = (z_nodes * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-6)
#         return z_graph, z_nodes, mask

#     def decode_edges(self, z_nodes):
#         B, N, H = z_nodes.shape
#         z_i = z_nodes.unsqueeze(2).expand(B, N, N, H)
#         z_j = z_nodes.unsqueeze(1).expand(B, N, N, H)
#         return torch.cat([z_i, z_j], dim=-1)

#     def training_step_ae(self, batch_data):
#         z_graph, z_nodes, mask = self(batch_data)
#         total_loss = 0.0
#         x_dense, _ = to_dense_batch(batch_data.x, batch_data.batch, max_num_nodes=self.config.max_nodes)
        
#         keys = ['atomic_num', 'chirality', 'degree', 'formal_charge', 'num_hs', 
#                 'num_radical_electrons', 'hybridization', 'is_aromatic', 'is_in_ring']
#         for i, k in enumerate(keys):
#             targets = x_dense[..., i].long()
#             logits = self.node_decoders[k](z_nodes)
#             total_loss += F.cross_entropy(logits[mask], targets[mask])

#         z_pairs = self.decode_edges(z_nodes)
#         edge_keys = ['bond_type', 'stereo', 'is_conjugated']
#         for i, k in enumerate(edge_keys):
#             target_adj = to_dense_adj(batch_data.edge_index, batch_data.batch, 
#                                       edge_attr=batch_data.edge_attr[..., i], 
#                                       max_num_nodes=self.config.max_nodes).long()
#             logits = self.edge_classifiers[k](z_pairs)
#             pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)
#             total_loss += F.cross_entropy(logits[pair_mask], target_adj[pair_mask])
            
#         return total_loss

class RichGrALE(nn.Module):
    def __init__(self, config: GraleConfig):
        super().__init__()
        self.config = config
        self.embedding = RichFeatureEmbedding(config)
        self.max_dist = 20
        self.spatial_bias = nn.Embedding(self.max_dist + 1, config.num_heads)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim, nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4, dropout=config.dropout,
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.out_norm = nn.LayerNorm(config.hidden_dim)
        
        # Heads for AE
        # Note: In GrALE, decoders often predict logits for 'max_nodes' 
        # independent of input order, but we keep your structure for compatibility.
        self.node_decoders = nn.ModuleDict({
            k: nn.Linear(config.hidden_dim, v) for k, v in config.feat_dims.items()
            if k not in ['bond_type', 'stereo', 'is_conjugated']
        })
        self.edge_decoder_dim = config.hidden_dim * 2
        self.edge_classifiers = nn.ModuleDict({
            k: nn.Sequential(
                nn.Linear(self.edge_decoder_dim, config.hidden_dim), 
                nn.GELU(), 
                nn.Linear(config.hidden_dim, v)
            ) for k, v in config.feat_dims.items() if k in ['bond_type', 'stereo', 'is_conjugated']
        })
        
        # Sinkhorn regularization parameter (epsilon)
        self.sinkhorn_epsilon = 0.05 
        self.sinkhorn_iters = 20

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, batch_data):
        # x_dense: [B, max_nodes, F_in]
        x_dense, mask = to_dense_batch(batch_data.x, batch_data.batch, max_num_nodes=self.config.max_nodes)
        
        h = self.embedding.forward_nodes(x_dense)
        src_key_padding_mask = ~mask
        z_nodes = self.encoder(h, src_key_padding_mask=src_key_padding_mask)
        z_nodes = self.out_norm(z_nodes)
        
        mask_expanded = mask.unsqueeze(-1).float()
        z_graph = (z_nodes * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-6)
        
        return z_graph, z_nodes, mask

    def decode_edges(self, z_nodes):
        B, N, H = z_nodes.shape
        z_i = z_nodes.unsqueeze(2).expand(B, N, N, H)
        z_j = z_nodes.unsqueeze(1).expand(B, N, N, H)
        return torch.cat([z_i, z_j], dim=-1)

    def sinkhorn_logsumexp(self, C, mask, iters=20, epsilon=0.05):
        """
        Computes the soft matching matrix T using the LogSumExp Sinkhorn algorithm.
        C: Cost matrix [B, N, N]
        mask: Valid nodes mask [B, N]
        """
        B, N, _ = C.shape
        
        # Initialize dual variables (u, v)
        u = torch.zeros(B, N, device=C.device)
        v = torch.zeros(B, N, device=C.device)
        
        # Mask invalid entries in cost with infinity
        # We only want to match valid nodes to valid nodes
        valid_mask_matrix = mask.unsqueeze(2) & mask.unsqueeze(1) # [B, N, N]
        C = C.masked_fill(~valid_mask_matrix, 1e9)

        # Sinkhorn iterations in log-domain for stability
        for _ in range(iters):
            # Update u
            # u = -epsilon * logsumexp( (v - C)/epsilon )
            u = -epsilon * torch.logsumexp((v.unsqueeze(1) - C) / epsilon, dim=2)
            u = u.masked_fill(~mask, 0.0) # Reset duals for padded nodes
            
            # Update v
            # v = -epsilon * logsumexp( (u - C)/epsilon )
            v = -epsilon * torch.logsumexp((u.unsqueeze(2) - C) / epsilon, dim=1)
            v = v.masked_fill(~mask, 0.0)
            
        # Compute final transport plan T = exp((u + v - C) / epsilon)
        T = torch.exp((u.unsqueeze(2) + v.unsqueeze(1) - C) / epsilon)
        
        # Zero out padding matches ensuring strict validity
        T = T * valid_mask_matrix.float()
        return T

    def training_step_ae(self, batch_data):
        z_graph, z_nodes, mask = self(batch_data)
        
        # 1. Prepare Targets
        x_dense, _ = to_dense_batch(batch_data.x, batch_data.batch, max_num_nodes=self.config.max_nodes)
        
        # 2. Compute Node Cost Matrix C_nodes [B, N_pred, N_target]
        # We calculate the cost of matching predicted node i to target node j.
        # For simplicity, we sum the cross-entropy losses for all features.
        
        node_keys = ['atomic_num', 'chirality', 'degree', 'formal_charge', 'num_hs', 
                     'num_radical_electrons', 'hybridization', 'is_aromatic', 'is_in_ring']
        
        B, N, _ = z_nodes.shape
        C_nodes = torch.zeros(B, N, N, device=z_nodes.device)
        
        # Pre-compute logits for all nodes
        node_logits = {k: self.node_decoders[k](z_nodes) for k in node_keys}
        
        for k in node_keys:
            logits = node_logits[k]  # [B, N, num_classes]
            targets = x_dense[..., node_keys.index(k)].long() # [B, N]
            
            # Expand to form pairwise costs: Cost(i, j) = Loss(Pred_i, Target_j)
            # Pred_i shape: [B, N, 1, C], Target_j shape: [B, 1, N]
            l = logits.unsqueeze(2).expand(-1, -1, N, -1) # [B, N, N, Classes]
            t = targets.unsqueeze(1).expand(-1, N, -1)    # [B, N, N]
            
            # Calculate Cross Entropy for every pair (i, j)
            # Flatten to [B*N*N, Classes] for F.cross_entropy
            flat_l = l.reshape(-1, l.size(-1))
            flat_t = t.reshape(-1)
            pair_loss = F.cross_entropy(flat_l, flat_t, reduction='none').view(B, N, N)
            
            C_nodes += pair_loss

        # 3. Compute Transport Plan T (Alignment)
        # In GrALE, we align based on reconstruction cost.
        T = self.sinkhorn_logsumexp(C_nodes, mask, iters=self.sinkhorn_iters, epsilon=self.sinkhorn_epsilon)
        
        # 4. Compute Loss Node (Weighted by T)
        # Loss = sum_{i,j} T_{ij} * C_{ij}
        loss_nodes = (T * C_nodes).sum() / B

        # 5. Compute Edge Loss (Gromov-Wasserstein Style)
        # We need to compare Pred_Adjacency to (T @ Target_Adjacency @ T.t())
        # Or more simply: Loss_edge = sum_{i,j,k,l} T_{ik} T_{jl} Loss(PredEdge_{ij}, TargetEdge_{kl})
        
        z_pairs = self.decode_edges(z_nodes)
        edge_keys = ['bond_type', 'stereo', 'is_conjugated']
        
        total_edge_loss = 0.0
        
        for i_feat, k in enumerate(edge_keys):
            # Target Adjacency for this feature
            target_adj = to_dense_adj(
                batch_data.edge_index, batch_data.batch, 
                edge_attr=batch_data.edge_attr[..., i_feat], 
                max_num_nodes=self.config.max_nodes
            ).long() # [B, N, N]
            
            pred_logits = self.edge_classifiers[k](z_pairs) # [B, N, N, Classes]
            
            # To compute GW loss efficiently:
            # We want to match Pred edge (i,j) with Target edge (k,l) using weights T[i,k] * T[j,l]
            # It implies we align the targets using T: aligned_target = T @ OneHot(target_adj) @ T.T
            # Since standard CE is hard with soft targets, we can use the "Permuted Target" approximation 
            # or simply project the logits.
            
            # Standard GrALE Approach: 
            # Reconstruct the target adjacency using the transport plan T
            # Soft Permuted Target: T * Target * T^T
            # Note: Since targets are categorical indices, we ideally permute the one-hot encoding.
            
            num_classes = pred_logits.size(-1)
            target_one_hot = F.one_hot(target_adj, num_classes=num_classes).float() # [B, N, N, C]
            
            # Permute target: B x C x N x N (permute spatial dims)
            target_permuted = torch.einsum('bik,bklc,bjl->bijc', T, target_one_hot, T)
            
            # Compute Cross Entropy with Soft Targets (KL Divergence or Soft CE)
            # Pred logits are [B, N, N, C], Target is probability distribution [B, N, N, C]
            log_probs = F.log_softmax(pred_logits, dim=-1)
            
            # Mask for valid edges (i,j both valid)
            pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)
            
            # Soft CE: - sum( p_target * log(p_pred) )
            feat_loss = - (target_permuted * log_probs).sum(dim=-1)
            total_edge_loss += (feat_loss * pair_mask.float()).sum() / B

        return loss_nodes + total_edge_loss

# =========================================================
# 3. Multi-Token Projector & Graph Augmented LLM
# =========================================================

class MultiTokenProjector(nn.Module):
    """
    Projects the graph embedding into K distinct tokens using an MLP.
    """
    def __init__(self, input_dim, output_dim, num_tokens=8):
        super().__init__()
        self.num_tokens = num_tokens
        self.output_dim = output_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.GELU(),
            nn.Linear(output_dim * 2, num_tokens * output_dim)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        out = self.net(x)
        return out.view(batch_size, self.num_tokens, self.output_dim)

class GraphAugmentedLLM(nn.Module):
    def __init__(
        self, 
        llm_model_path: str, 
        graph_encoder: Union['RichGrALE', str], 
        use_lora: bool = False, 
        projector: Union[nn.Module, str, None] = None,
        num_tokens: int = 8,
        lora_r: int = 128,
        lora_alpha: float = None,
        device=None,
    ):
        super().__init__()
        
        # 1. Load Configuration & Tokenizer
        print(f"Loading LLM: {llm_model_path}")
        config = AutoConfig.from_pretrained(llm_model_path, trust_remote_code=True)
        self.is_encoder_decoder = config.is_encoder_decoder
        
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_path, trust_remote_code=True)
        
        # 2. Universal Graph Token Setup
        self.graph_token = "[GRAPH_VECTOR]"
        self.must_resize = False
        
        if self.graph_token not in self.tokenizer.get_vocab():
            print(f"Adding special token '{self.graph_token}' to tokenizer...")
            self.tokenizer.add_tokens([self.graph_token], special_tokens=True)
            self.must_resize = True
            
        self.graph_token_id = self.tokenizer.convert_tokens_to_ids(self.graph_token)
        print(f"Graph Anchor Token: '{self.graph_token}' (ID: {self.graph_token_id})")

        # 3. Load LLM Backbone
        if self.is_encoder_decoder:
            model_cls = AutoModelForSeq2SeqLM
        else:
            model_cls = AutoModelForCausalLM

        self.llm = model_cls.from_pretrained(
            llm_model_path, 
            trust_remote_code=True, 
            device_map="auto", 
            torch_dtype=torch.float16
        )
        
        if self.must_resize:
            print(f"Resizing model embeddings to {len(self.tokenizer)}...")
            self.llm.resize_token_embeddings(len(self.tokenizer))

        if use_lora:
            from peft import get_peft_model, LoraConfig
            lora_alpha = lora_r * 2 if lora_alpha is None else lora_alpha
            peft_config = LoraConfig(
                r=lora_r, 
                lora_alpha=lora_alpha, 
                target_modules=["q_proj", "v_proj"], 
                task_type="CAUSAL_LM" if not self.is_encoder_decoder else "SEQ_2_SEQ_LM"
            )
            self.llm = get_peft_model(self.llm, peft_config)
            
        # 4. Load Graph Encoder
        if isinstance(graph_encoder, str):
            ckpt = torch.load(graph_encoder, map_location='cpu')
            from graph_llm_aligned import GraleConfig, RichGrALE 
            enc_config = ckpt.get('config', GraleConfig())
            self.graph_encoder = RichGrALE(enc_config)
            self.graph_encoder.load_state_dict(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt)
        else:
            self.graph_encoder = graph_encoder

        # 5. Load/Init Projector
        if isinstance(projector, str):
            ckpt = torch.load(projector, map_location='cpu')
            from graph_llm_aligned import MultiTokenProjector 
            self.projector = MultiTokenProjector(ckpt['input_dim'], ckpt['output_dim'], ckpt.get('num_tokens', 8))
            self.projector.load_state_dict(ckpt['state_dict'])
        elif projector is not None:
            self.projector = projector
        else:
            from graph_llm_aligned import MultiTokenProjector
            self.projector = MultiTokenProjector(
                input_dim=self.graph_encoder.config.hidden_dim, 
                output_dim=self.llm.config.hidden_size, 
                num_tokens=num_tokens,
            )
        
        # Gen config
        self.generation_config = None

        if device is not None:
            self.to(device)

    def _replace_graph_token(self, input_ids, inputs_embeds, graph_emb, attention_mask=None, labels=None):
        """
        Splices graph_emb into inputs_embeds AND expands labels/masks accordingly.
        """
        is_graph_token = (input_ids == self.graph_token_id)
        
        if not is_graph_token.any():
            return inputs_embeds, attention_mask, labels

        batch_size = inputs_embeds.shape[0]
        new_embeds_list = []
        new_mask_list = []
        new_labels_list = []
        
        for i in range(batch_size):
            indices = torch.nonzero(is_graph_token[i], as_tuple=True)[0]
            
            # Case: No graph token in this specific sample
            if len(indices) == 0:
                new_embeds_list.append(inputs_embeds[i])
                if attention_mask is not None:
                    new_mask_list.append(attention_mask[i])
                if labels is not None:
                    new_labels_list.append(labels[i])
                continue

            # Replace the FIRST occurrence
            idx = indices[0]
            
            # 1. Splice Embeddings
            combined_emb = torch.cat([
                inputs_embeds[i, :idx],
                graph_emb[i],                 
                inputs_embeds[i, idx+1:]
            ], dim=0)
            new_embeds_list.append(combined_emb)
            
            # 2. Splice Mask
            if attention_mask is not None:
                graph_mask = torch.ones((graph_emb.shape[1],), dtype=attention_mask.dtype, device=attention_mask.device)
                combined_mask = torch.cat([
                    attention_mask[i, :idx],
                    graph_mask,
                    attention_mask[i, idx+1:]
                ], dim=0)
                new_mask_list.append(combined_mask)

            # 3. Splice Labels (CRITICAL FIX)
            if labels is not None:
                # We insert -100 (ignore index) for the graph tokens
                ignore_tokens = torch.full((graph_emb.shape[1],), -100, dtype=labels.dtype, device=labels.device)
                combined_labels = torch.cat([
                    labels[i, :idx],
                    ignore_tokens,
                    labels[i, idx+1:]
                ], dim=0)
                new_labels_list.append(combined_labels)

        from torch.nn.utils.rnn import pad_sequence
        
        final_embeds = pad_sequence(new_embeds_list, batch_first=True, padding_value=0.0)
        
        final_mask = None
        if attention_mask is not None:
            final_mask = pad_sequence(new_mask_list, batch_first=True, padding_value=0)
            
        final_labels = None
        if labels is not None:
            # Pad labels with -100 so they are ignored in loss
            final_labels = pad_sequence(new_labels_list, batch_first=True, padding_value=-100)
            
        return final_embeds, final_mask, final_labels

    def forward(self, input_ids, attention_mask=None, labels=None, graph_data=None, **kwargs):
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        
        if graph_data is not None:
            target_device = inputs_embeds.device
            self.graph_encoder.to(target_device)
            self.projector.to(target_device)
            if hasattr(graph_data, 'to'): 
                graph_data = graph_data.to(target_device)

            z_graph, _, _ = self.graph_encoder(graph_data)
            graph_emb = self.projector(z_graph).to(dtype=inputs_embeds.dtype)
            
            # --- UPDATED CALL: Pass and receive labels ---
            inputs_embeds, attention_mask, labels = self._replace_graph_token(
                input_ids, inputs_embeds, graph_emb, attention_mask, labels
            )
            
        return self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, **kwargs)

    def generate(self, input_ids, graph_data=None, **kwargs):
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        attention_mask = kwargs.get('attention_mask', None)

        if graph_data is not None:
            target_device = inputs_embeds.device
            self.graph_encoder.to(target_device)
            self.projector.to(target_device)
            if hasattr(graph_data, 'to'):
                graph_data = graph_data.to(target_device)

            z_graph, _, _ = self.graph_encoder(graph_data)
            graph_emb = self.projector(z_graph).to(dtype=inputs_embeds.dtype)
            
            # Note: generate doesn't use labels, so we ignore the 3rd return
            inputs_embeds, attention_mask, _ = self._replace_graph_token(
                input_ids, inputs_embeds, graph_emb, attention_mask
            )
            
            if attention_mask is not None:
                kwargs['attention_mask'] = attention_mask
            
        return self.llm.generate(inputs_embeds=inputs_embeds, **kwargs)

    def enable_input_require_grads(self):
        self.llm.enable_input_require_grads()
        self.graph_encoder.requires_grad_(True)
        self.projector.requires_grad_(True)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def print_trainable_parameters(self):
        """Prints breakdown of trainable parameters."""
        llm_trainable = sum(p.numel() for p in self.llm.parameters() if p.requires_grad)
        llm_all = sum(p.numel() for p in self.llm.parameters())
        
        graph_trainable = sum(p.numel() for p in self.graph_encoder.parameters() if p.requires_grad)
        proj_trainable = sum(p.numel() for p in self.projector.parameters() if p.requires_grad)
        
        total_trainable = llm_trainable + graph_trainable + proj_trainable
        total_all = llm_all + sum(p.numel() for p in self.graph_encoder.parameters()) + sum(p.numel() for p in self.projector.parameters())

        print(f"\nGraphAugmentedLLM Trainable Params: {total_trainable:,} / {total_all:,} ({100 * total_trainable / total_all:.4f}%)")
        print(f"  - LLM (LoRA?): {llm_trainable:,}")
        print(f"  - GraphEnc:    {graph_trainable:,}")
        print(f"  - Projector:   {proj_trainable:,}\n")
        
    @property
    def device(self):
        return self.llm.device

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        return self

# =========================================================
# 4. Save/Load Utilities
# =========================================================

def save_graph_components(
    graph_encoder: RichGrALE, 
    projector: MultiTokenProjector, 
    save_dir: str,
    llm_model_path: str
):
    """
    Saves the Graph Encoder and Projector separately for easy reloading.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    graph_encoder.config.llm_model_path = llm_model_path

    # 1. Save RichGrALE
    grale_path = os.path.join(save_dir, "rich_grale.pt")
    torch.save({
        'config': graph_encoder.config,
        'state_dict': graph_encoder.state_dict()
    }, grale_path)
    print(f"Saved RichGrALE (LLM Backbone: {graph_encoder.config.llm_model_path}) to {grale_path}")
    
    # 2. Save Projector
    proj_path = os.path.join(save_dir, "projector.pt")
    # Infer input dim from first layer weight (self.net[0] is Linear)
    input_dim = projector.net[0].weight.shape[1] 
    torch.save({
        'input_dim': input_dim,
        'output_dim': projector.output_dim,
        'num_tokens': projector.num_tokens,
        'state_dict': projector.state_dict()
    }, proj_path)
    print(f"Saved Projector to {proj_path}")

def load_graph_components(save_dir: str, device='cpu') -> Tuple[RichGrALE, MultiTokenProjector]:
    """
    Loads RichGrALE and Projector from disk (Static Utility).
    Useful if you need them isolated from the LLM.
    """
    grale_path = os.path.join(save_dir, "rich_grale.pt")
    if not os.path.exists(grale_path):
        raise FileNotFoundError(f"RichGrALE checkpoint not found at {grale_path}")
        
    grale_ckpt = torch.load(grale_path, map_location=device, weights_only=False)
    grale_model = RichGrALE(grale_ckpt['config'])
    grale_model.load_state_dict(grale_ckpt['state_dict'])
    grale_model.to(device)
    grale_model.eval() 
    
    proj_path = os.path.join(save_dir, "projector.pt")
    if not os.path.exists(proj_path):
        raise FileNotFoundError(f"Projector checkpoint not found at {proj_path}")
        
    proj_ckpt = torch.load(proj_path, map_location=device, weights_only=False)
    projector = MultiTokenProjector(
        input_dim=proj_ckpt['input_dim'],
        output_dim=proj_ckpt['output_dim'],
        num_tokens=proj_ckpt['num_tokens']
    )
    projector.load_state_dict(proj_ckpt['state_dict'])
    projector.to(device)
    projector.eval()
    
    return grale_model, projector

# =========================================================
# 5. Training Utilities
# =========================================================

class GraphPickleDataset(Dataset):
    def __init__(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            self.data_list = pickle.load(f)
    def __len__(self): return len(self.data_list)
    def __getitem__(self, idx): return self.data_list[idx]

def graph_collate_fn(data_list):
    return Batch.from_data_list(data_list)

def visualize_latent_space(model, loader, device, output_dir=".", step_suffix=""):
    model.eval()
    all_z = []
    max_batches = 10
    print(f"Extracting latent vectors for visualization...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches: break
            if hasattr(batch, 'to'): batch = batch.to(device)
            z_graph, _, _ = model(batch)
            all_z.append(z_graph.cpu())
    if not all_z: return

    all_z = torch.cat(all_z, dim=0).numpy()
    num_samples = all_z.shape[0]
    
    if num_samples > 30:
        perp = min(30, num_samples - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perp, init='pca', learning_rate='auto')
        z_tsne = tsne.fit_transform(all_z)
        
        plt.figure(figsize=(8, 6))
        plt.scatter(z_tsne[:, 0], z_tsne[:, 1], alpha=0.6, c='#1f77b4', s=20)
        plt.title(f"t-SNE Latent Space")
        plt.savefig(os.path.join(output_dir, f"latent_space_tsne_{step_suffix}.png"))
        plt.close()
        
        if umap is not None:
            reducer = umap.UMAP(n_components=2)
            z_umap = reducer.fit_transform(all_z)
            plt.figure(figsize=(8, 6))
            plt.scatter(z_umap[:, 0], z_umap[:, 1], alpha=0.6, c='#2ca02c', s=20)
            plt.title(f"UMAP Latent Space")
            plt.savefig(os.path.join(output_dir, f"latent_space_umap_{step_suffix}.png"))
            plt.close()

def train_grale_scratch(train_pkl_path, 
                        output_dir, 
                        epochs=1, 
                        batch_size=16, 
                        max_steps=None,
                        config_grale={},
                        ):
    import torch.optim as optim
    
    # 1. Setup
    dataset = GraphPickleDataset(train_pkl_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=graph_collate_fn)
    
    config = GraleConfig(**config_grale)
    model = RichGrALE(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    
    # 2. Calculate Milestones (10%, 50%, 100%)
    # Ensure they are at least epoch 1 and unique
    milestones = {max(1,int(len(loader) * epochs * 0.1)), int(len(loader) * epochs * 0.5), len(loader) * epochs}
    
    print(f"--- Stage 1: Training RichGrALE Autoencoder ({device}) ---")
    print(f"Visualization milestones (Epochs): {sorted(list(milestones))}")
    step = 0
    for epoch in range(epochs):
        current_epoch = epoch + 1
        total_loss = 0
        pbar = tqdm(loader, desc=f"GrALE Epoch {current_epoch}/{epochs}")
        
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = model.training_step_ae(batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            step += 1

            # Trigger Visualization at Milestones
            if step in milestones:
                visualize_latent_space(
                    model, 
                    loader, 
                    device, 
                    output_dir=output_dir, 
                    step_suffix=f"step_{step}"
                )
                model.train()

            if max_steps is not None and step >= max_steps:
                break
        # Save checkpoint
        save_graph_components(model, MultiTokenProjector(1,1), output_dir, config.llm_model_path)
            
    return model

def train_projector_alignment(
    model: GraphAugmentedLLM, 
    train_pkl_path: str, 
    tokenizer, 
    epochs=1, 
    batch_size=8, 
    lr=1e-3, 
    output_dir=".",
    max_steps=None,
):
    import torch.optim as optim
    print("\n--- Stage 2: Aligning MultiToken Projector (Graph -> Tokens) ---")
    
    # 1. Freeze everything except Projector
    model.graph_encoder.eval()
    model.llm.eval()
    for param in model.graph_encoder.parameters(): 
        param.requires_grad = False
    for param in model.llm.parameters(): 
        param.requires_grad = False
    
    model.projector.train()
    for param in model.projector.parameters(): 
        param.requires_grad = True
    
    # 2. Setup Data
    dataset = GraphPickleDataset(train_pkl_path)
    # Filter for valid descriptions
    dataset.data_list = [d for d in dataset.data_list if hasattr(d, 'description') and d.description]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=graph_collate_fn)
    
    optimizer = optim.AdamW(model.projector.parameters(), lr=lr)
    device = model.llm.device 
    
    # Retrieve the token string dynamically
    graph_token = model.graph_token 

    step_count = 0
    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(loader, desc=f"Align Epoch {epoch+1}")
        
        for step, batch in enumerate(pbar):
            # Extract Data
            descriptions = [g.description for g in batch.to_data_list()]
            mol_cards = [g.mol_card for g in batch.to_data_list()]
            
            # --- 1. Construct Texts with Chat Template ---
            full_texts = []
            prompt_texts_for_masking = []

            for card, desc in zip(mol_cards, descriptions):
                # Construct the User Content containing the Graph Token
                user_content = f"\n\n[QUERY MOLECULE]\n{graph_token}\nCard: {card}\nDescribe this molecule."
                
                # Check for Chat Template
                if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                    # Full Conversation: User + Assistant (Target)
                    full_conv = [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": desc}
                    ]
                    full_text = tokenizer.apply_chat_template(full_conv, tokenize=False)
                    
                    # Prompt Only: User + Generation Prompt (for masking)
                    prompt_conv = [{"role": "user", "content": user_content}]
                    prompt_text = tokenizer.apply_chat_template(prompt_conv, tokenize=False, add_generation_prompt=True)
                else:
                    # Fallback for models without templates
                    full_text = f"{user_content}\n{desc}"
                    prompt_text = f"{user_content}\n"
                
                full_texts.append(full_text)
                prompt_texts_for_masking.append(prompt_text)
            
            # --- 2. Tokenize & Masking ---
            original_padding_side = tokenizer.padding_side
            tokenizer.padding_side = "right"
            
            # Tokenize Full Text (add_special_tokens=False because template handles structure usually, 
            # but we allow tokenizer to add BOS if it detects start)
            # We follow SFT logic: add_special_tokens=True usually safe with string templates
            inputs = tokenizer(
                full_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=256,
                add_special_tokens=True 
            ).to(device)
            
            labels = inputs.input_ids.clone()
            
            # Calculate length of the prompt part for masking
            # We tokenize prompts separately without special tokens to measure length correctly relative to full
            prompt_tokens = tokenizer(
                prompt_texts_for_masking, 
                add_special_tokens=False, # Template string already has tokens, we just want count
                padding=False,
                truncation=True,
                max_length=256
            )
            
            for i, p_ids in enumerate(prompt_tokens['input_ids']):
                p_len = len(p_ids)
                
                # Correction: If full input has BOS but our prompt tokenization didn't add it, offset by 1
                if tokenizer.bos_token_id is not None and inputs.input_ids[i][0] == tokenizer.bos_token_id:
                     # Check if our simple tokenization missed the BOS
                     if len(p_ids) > 0 and p_ids[0] != tokenizer.bos_token_id:
                         p_len += 1

                # Mask tokens [0 ... p_len]
                if p_len < labels.shape[1]:
                    labels[i, :p_len] = -100
                else:
                    labels[i, :] = -100
            
            # Restore padding side
            tokenizer.padding_side = original_padding_side
            
            # --- 3. Forward Pass ---
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                labels=labels, # Masked labels passed here
                graph_data=batch.to(device)
            )
            
            loss = outputs.loss
            
            # Backward
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            step_count += 1

            if max_steps is not None and step_count >= max_steps:
                break
        if max_steps is not None and step_count >= max_steps:
            break
    
    print("\nSaving Aligned Components...")
    config = model.graph_encoder.config
    save_graph_components(model.graph_encoder, model.projector, output_dir, config.llm_model_path)

    return model

# =========================================================
# 6. Main Execution
# =========================================================

if __name__ == "__main__":
    from argparse import ArgumentParser
    import os
    from pathlib import Path

    parser = ArgumentParser()
    parser.add_argument("-f_data", default="data_baseline/data", type=str)
    parser.add_argument("-f", default="data_baseline/data", type=str)
    parser.add_argument("-base_llm", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", type=str)
    parser.add_argument("-steps", default=None, type=int)
    parser.add_argument("-do_grale", action="store_true")
    parser.add_argument("-grale_batch_size", default=16, type=int)
    parser.add_argument("-grale_epochs", default=2, type=int)
    parser.add_argument("-do_proj", action="store_true")
    parser.add_argument("-proj_batch_size", default=8, type=int)
    parser.add_argument("-proj_epochs", default=2, type=int)
    args = parser.parse_args()

    file_path = Path(os.path.abspath(__file__))
    parent_folder = file_path.parent
    data_path = parent_folder.parent / args.f_data
    base_path = parent_folder.parent / args.f
    os.makedirs(str(base_path), exist_ok=True)
    TRAIN_DATA_PATH = data_path / "train_graphs.pkl"

    MAX_STEPS = args.steps

    LLM_ID = args.base_llm
    print(f"Using {LLM_ID}")

    # 1. Train RichGrALE
    if args.do_grale:
        trained_grale = train_grale_scratch(
            train_pkl_path=TRAIN_DATA_PATH,
            output_dir=base_path,
            epochs=args.grale_epochs, 
            batch_size=args.grale_batch_size,
            max_steps=MAX_STEPS,
            config_grale={'llm_model_path':LLM_ID},
        )
    else:
        trained_grale, projector_model = load_graph_components(base_path, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    if args.do_proj:
        # 2. Init LLM
        print("\nInitializing LLM...")
        llm_model = GraphAugmentedLLM(
            llm_model_path=LLM_ID,
            graph_encoder=trained_grale,
            use_lora=False 
        )
        
        # FIX: Use the tokenizer from the model (which has [GRAPH_VECTOR])
        tokenizer = llm_model.tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 3. Align Projector
        llm_model = train_projector_alignment(
            model=llm_model,
            train_pkl_path=TRAIN_DATA_PATH,
            tokenizer=tokenizer,
            epochs=args.proj_epochs, 
            batch_size=args.proj_batch_size,
            lr=5e-4, 
            output_dir=base_path,
            max_steps=MAX_STEPS,
        )
        
        # 4. Inference Test
        print("\n--- Running Inference Test ---")
        dataset = GraphPickleDataset(TRAIN_DATA_PATH)
        sample_graph = dataset[0]
        
        if hasattr(sample_graph, 'description'):
            print(f"Ground Truth: {sample_graph.description}...")

        graph_batch = Batch.from_data_list([sample_graph])
        device = llm_model.llm.device
        graph_batch = graph_batch.to(device)

        # --- FIX 1: Construct Prompt with [GRAPH_VECTOR] ---
        # --- FIX 2: Use Chat Template to suppress "Thinking" rambling ---
        user_content = f"\n\n[QUERY MOLECULE]\n[GRAPH_VECTOR]\nCard: {sample_graph.mol_card}\nDescribe this molecule."
        
        messages = [{"role": "user", "content": user_content}]
        
        # Apply chat template
        prompt_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        print(f"Prompt: {prompt_text}")

        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = llm_model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                graph_data=graph_batch,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.6, 
                pad_token_id=tokenizer.eos_token_id
            )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Optional: Strip <think> tags if they still appear
        if "</think>" in decoded:
            decoded = decoded.split("</think>")[-1].strip()
            
        print("\n--- Model Output (Aligned) ---")
        print(decoded)
        print("------------------------------")