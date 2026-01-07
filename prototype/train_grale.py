import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any, Union

# Try imports for PyG (torch_geometric)
try:
    from torch_geometric.data import Batch, Data
    from torch_geometric.utils import to_dense_batch, to_dense_adj
except ImportError:
    print("Warning: torch_geometric not found. Graph features require PyG.")

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoConfig,
    PreTrainedModel
)
from transformers.utils import ModelOutput

# =========================================================
# 1. Configuration & Feature Maps (from data_utils.py)
# =========================================================

# Dimensions based on your provided maps
FEATURE_DIMS = {
    # Node Features
    'atomic_num': 119,
    'chirality': 9,
    'degree': 11,
    'formal_charge': 12, # Range -5 to 6 mapped to 0-11
    'num_hs': 9,
    'num_radical_electrons': 5,
    'hybridization': 8,
    'is_aromatic': 2,
    'is_in_ring': 2,
    # Edge Features
    'bond_type': 22,
    'stereo': 7,
    'is_conjugated': 2
}

@dataclass
class GraleConfig:
    # Model Hyperparameters
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    max_nodes: int = 575  # (max is 574 from training, val and test inspection) Fixed size for dense batching (adjust as needed)
    
    # Feature Config
    feat_dims: Dict[str, int] = None

    def __post_init__(self):
        if self.feat_dims is None:
            self.feat_dims = FEATURE_DIMS

# =========================================================
# 2. Rich Feature Embedder
# =========================================================

class RichFeatureEmbedding(nn.Module):
    """
    Embeds the 9 node features and 3 edge features into a unified latent space.
    """
    def __init__(self, config: GraleConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        
        # --- Node Embeddings ---
        # We create separate embeddings for each categorical feature
        self.node_embs = nn.ModuleDict({
            'atomic_num': nn.Embedding(config.feat_dims['atomic_num'], self.hidden_dim),
            'chirality': nn.Embedding(config.feat_dims['chirality'], self.hidden_dim),
            'degree': nn.Embedding(config.feat_dims['degree'], self.hidden_dim),
            'formal_charge': nn.Embedding(config.feat_dims['formal_charge'], self.hidden_dim),
            'num_hs': nn.Embedding(config.feat_dims['num_hs'], self.hidden_dim),
            'num_radical': nn.Embedding(config.feat_dims['num_radical_electrons'], self.hidden_dim),
            'hybridization': nn.Embedding(config.feat_dims['hybridization'], self.hidden_dim),
            'is_aromatic': nn.Embedding(config.feat_dims['is_aromatic'], self.hidden_dim),
            'is_in_ring': nn.Embedding(config.feat_dims['is_in_ring'], self.hidden_dim),
        })
        
        # Project concatenated embeddings back to hidden_dim? 
        # Or sum them? Summing is parameter efficient and standard in Graphormer.
        # We will use summation of embeddings.

        # --- Edge Embeddings ---
        self.edge_embs = nn.ModuleDict({
            'bond_type': nn.Embedding(config.feat_dims['bond_type'], self.hidden_dim),
            'stereo': nn.Embedding(config.feat_dims['stereo'], self.hidden_dim),
            'is_conjugated': nn.Embedding(config.feat_dims['is_conjugated'], self.hidden_dim),
        })

    def forward_nodes(self, x: torch.Tensor):
        """
        x shape: [Batch, Max_Nodes, 9] (Dense) or [Total_Nodes, 9] (Sparse)
        Indices in x correspond to the order in data_utils.x_map
        """
        # x[..., 0] -> atomic_num
        # x[..., 1] -> chirality
        # ...
        
        # Helper to sum embeddings safely
        out = 0
        out += self.node_embs['atomic_num'](x[..., 0].long())
        out += self.node_embs['chirality'](x[..., 1].long())
        out += self.node_embs['degree'](x[..., 2].long())
        
        # Formal charge: map range -5..6 to 0..11. 
        # Assuming input is already mapped indices from data_utils, otherwise adjust here.
        # data_utils.x_map['formal_charge'] is a list. The dataset likely stores the *index*.
        out += self.node_embs['formal_charge'](x[..., 3].long())
        
        out += self.node_embs['num_hs'](x[..., 4].long())
        out += self.node_embs['num_radical'](x[..., 5].long())
        out += self.node_embs['hybridization'](x[..., 6].long())
        out += self.node_embs['is_aromatic'](x[..., 7].long())
        out += self.node_embs['is_in_ring'](x[..., 8].long())
        
        return out

    def forward_edges(self, edge_attr: torch.Tensor):
        """
        edge_attr shape: [..., 3]
        """
        out = 0
        out += self.edge_embs['bond_type'](edge_attr[..., 0].long())
        out += self.edge_embs['stereo'](edge_attr[..., 1].long())
        out += self.edge_embs['is_conjugated'](edge_attr[..., 2].long())
        return out

# =========================================================
# 3. RichGrALE: The New Base Model (Graph Transformer)
# =========================================================

class RichGrALE(nn.Module):
    """
    A Graph Transformer trained from scratch to encode rich molecular graphs.
    It uses dense batching (padding to max_nodes) to allow standard Attention.
    """
    def __init__(self, config: GraleConfig):
        super().__init__()
        self.config = config
        self.embedding = RichFeatureEmbedding(config)
        
        # Spatial Encoding (Shortest Path Bias)
        # We learn a scalar bias for each distance up to max_dist
        self.max_dist = 20
        self.spatial_bias = nn.Embedding(self.max_dist + 1, config.num_heads)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        # Projector to generic latent (optional, can just use output)
        self.out_norm = nn.LayerNorm(config.hidden_dim)
        
        # --- Heads for Pre-training (AutoEncoder Task) ---
        # Predict atomic_num (classification) from latent
        self.atom_pred_head = nn.Linear(config.hidden_dim, config.feat_dims['atomic_num'])
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, batch_data):
        """
        Input: PyG Batch object containing:
          - x: [N_total, 9]
          - edge_index: [2, E_total]
          - edge_attr: [E_total, 3]
          - batch: [N_total]
        """
        device = batch_data.x.device
        
        # 1. Convert to Dense (Batch, Max_Nodes, Feat)
        # x_dense: [B, N_max, 9], mask: [B, N_max]
        x_dense, mask = to_dense_batch(batch_data.x, batch_data.batch, max_num_nodes=self.config.max_nodes)
        
        # 2. Embed Nodes
        # h: [B, N_max, Hidden]
        h = self.embedding.forward_nodes(x_dense)
        
        # 3. Compute/Process Adjacency & Spatial Bias
        # We need to reconstruct adjacency to compute shortest paths on the fly 
        # or assume they are pre-computed. For Training speed, we do on-the-fly coarse check or 
        # rely on edge_attr embeddings added to attention.
        
        # Simplification for "GrALE-like" attention:
        # We will simply pass `h` through Transformer. 
        # Ideally, we inject edge info into the attention matrix (Graphormer style).
        # Here, we stick to a robust Transformer on node features + simple adjacency bias if possible.
        # Given constraints, we'll assume the rich node features carry most info, 
        # and simple padding mask for the transformer.
        
        # Invert mask for Transformer (True = Ignored/Padding)
        # mask is True for real nodes, False for padding in PyG to_dense_batch.
        # Transformer expects True for padding (if boolean).
        src_key_padding_mask = ~mask
        
        # 4. Pass through Encoder
        # Out: [B, N_max, Hidden]
        z_nodes = self.encoder(h, src_key_padding_mask=src_key_padding_mask)
        z_nodes = self.out_norm(z_nodes)
        
        # 5. Global Graph Embedding
        # Average pooling over valid nodes
        # mask: [B, N_max] -> [B, N_max, 1]
        mask_expanded = mask.unsqueeze(-1).float()
        z_graph = (z_nodes * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-6)
        
        return z_graph, z_nodes, mask

    def training_step_ae(self, batch_data):
        """
        Autoencoder training step: Reconstruct atomic numbers.
        """
        z_graph, z_nodes, mask = self(batch_data)
        
        # Predict atomic numbers
        logits = self.atom_pred_head(z_nodes) # [B, N_max, 119]
        
        # Targets
        targets, _ = to_dense_batch(batch_data.x[..., 0], batch_data.batch, max_num_nodes=self.config.max_nodes)
        targets = targets.long()
        
        # Flatten
        active_logits = logits[mask]
        active_targets = targets[mask]
        
        loss = F.cross_entropy(active_logits, active_targets)
        return loss

# =========================================================
# 4. Graph Augmented LLM Wrapper
# =========================================================

class GraphAugmentedLLM(nn.Module):
    """
    Wraps a Hugging Face LLM and the Custom RichGrALE encoder.
    """
    def __init__(
        self, 
        llm_name_or_path: str,
        graph_encoder: RichGrALE,
        use_lora: bool = False
    ):
        super().__init__()
        
        # Load LLM
        print(f"Loading LLM: {llm_name_or_path}")
        self.llm = AutoModelForCausalLM.from_pretrained(llm_name_or_path)
        
        if use_lora:
            from peft import get_peft_model, LoraConfig
            peft_config = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
            self.llm = get_peft_model(self.llm, peft_config)
            
        # Graph Encoder
        self.graph_encoder = graph_encoder
        
        # Projector (Graph Dim -> LLM Dim)
        self.projector = nn.Linear(graph_encoder.config.hidden_dim, self.llm.config.hidden_size)
        
    def forward(self, input_ids, attention_mask=None, labels=None, graph_data=None, **kwargs):
        
        # 1. Get Text Embeddings
        # [Batch, Seq, LLM_Dim]
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        
        # 2. Inject Graph (if present)
        if graph_data is not None:
            # Encode Graph -> [Batch, Graph_Dim]
            z_graph, _, _ = self.graph_encoder(graph_data)
            
            # Project -> [Batch, 1, LLM_Dim]
            graph_emb = self.projector(z_graph).unsqueeze(1)
            
            # Concat at start
            inputs_embeds = torch.cat([graph_emb, inputs_embeds], dim=1)
            
            # Adjust Attention Mask
            if attention_mask is not None:
                pad = torch.ones((attention_mask.shape[0], 1), device=attention_mask.device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([pad, attention_mask], dim=1)
                
            # Adjust Labels (Shift for Causal LM training)
            if labels is not None:
                ignore = torch.full((labels.shape[0], 1), -100, device=labels.device, dtype=labels.dtype)
                labels = torch.cat([ignore, labels], dim=1)

        # 3. Forward LLM
        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

    def generate(self, input_ids, graph_data=None, **kwargs):
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        
        if graph_data is not None:
            z_graph, _, _ = self.graph_encoder(graph_data)
            graph_emb = self.projector(z_graph).unsqueeze(1)
            inputs_embeds = torch.cat([graph_emb, inputs_embeds], dim=1)
            
        return self.llm.generate(inputs_embeds=inputs_embeds, **kwargs)

# =========================================================
# 5. Data Loading & Training Utilities
# =========================================================

class GraphPickleDataset(Dataset):
    def __init__(self, pkl_path):
        print(f"Loading graphs from {pkl_path}...")
        with open(pkl_path, 'rb') as f:
            self.data_list = pickle.load(f)
        print(f"Loaded {len(self.data_list)} graphs.")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

def graph_collate_fn(data_list):
    """Collates a list of PyG Data objects into a Batch."""
    return Batch.from_data_list(data_list)

def train_grale_scratch(
    train_pkl_path: str,
    output_dir: str = "./custom_grale_ckpt",
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4
):
    """
    Trains the RichGrALE model from scratch using the provided graph data.
    """
    import torch.optim as optim
    
    # 1. Setup Data
    dataset = GraphPickleDataset(train_pkl_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=graph_collate_fn)
    
    # 2. Setup Model
    config = GraleConfig()
    model = RichGrALE(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    # 3. Training Loop
    model.train()
    print(f"Starting GrALE training on {device}...")
    
    for epoch in range(epochs):
        total_loss = 0
        for step, batch in enumerate(loader):
            batch = batch.to(device)
            
            optimizer.zero_grad()
            loss = model.training_step_ae(batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if step % 50 == 0:
                print(f"Epoch {epoch+1} | Step {step} | Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}")
        
        # Save Checkpoint
        os.makedirs(output_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(output_dir, f"grale_epoch_{epoch+1}.pt"))

    print("Training finished.")
    return model

if __name__ == "__main__":
    from argparse import ArgumentParser
    from transformers import AutoTokenizer
    import os
    import pickle
    from pathlib import Path

    parser = ArgumentParser()
    parser.add_argument("-f_data", default="data_baseline/data", type=str, help="Folder containing source .pkl graph files")
    parser.add_argument("-f", default="data_baseline/data", type=str, help="Output folder for model")

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

    # Paths (assumed from context)
    TRAIN_DATA_PATH = data_path / "train_graphs.pkl"  # Adjust path if needed based on your environment
    
    print("="*50)
    print("STEP 1: Training RichGrALE from Scratch")
    print("="*50)
    
    # 1. Train the Graph Encoder (GrALE)
    # We use a small number of epochs for the demo
    trained_grale = train_grale_scratch(
        train_pkl_path=TRAIN_DATA_PATH,
        output_dir=base_path,
        epochs=1, 
        batch_size=32
    )
    
    if trained_grale is None:
        print("GrALE training failed or data missing. Exiting.")
        exit()

    print("\n" + "="*50)
    print("STEP 2: Initializing Graph-Augmented LLM (DeepSeekR1-Distill-Qwen 1.5B)")
    print("="*50)

    # 2. Initialize LLM Wrapper with DeepSeek
    # Note: DeepSeek-R1-Distill-Qwen-1.5B is the model ID on HF
    LLM_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    
    try:
        # Load the composite model
        # We put the model on CPU first to merge, or Auto device map handles it inside the class
        llm_model = GraphAugmentedLLM(
            llm_name_or_path=LLM_ID,
            graph_encoder=trained_grale,
            use_lora=False # Set to True if you want to initialize LoRA adapters immediately
        )
        
        tokenizer = AutoTokenizer.from_pretrained(LLM_ID, trust_remote_code=True)
        
        print("\n" + "="*50)
        print("STEP 3: Running Integration Test")
        print("="*50)

        # 3. Test Generation
        # Load one graph to use as input
        dataset = GraphPickleDataset(TRAIN_DATA_PATH)
        sample_graph = dataset[0] # PyG Data object
        
        # Batch it (as the model expects a batch)
        graph_batch = Batch.from_data_list([sample_graph])
        graph_batch = graph_batch.to(llm_model.llm.device)
        
        # Create a text prompt
        prompt = "Describe this molecule:\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(llm_model.llm.device)
        
        print(f"Prompt: {prompt}")
        print("Generating response (with graph injection)...")
        
        # Generate
        with torch.no_grad():
            outputs = llm_model.generate(
                input_ids=inputs.input_ids,
                graph_data=graph_batch,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7
            )
            
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("\n--- Model Output ---")
        print(decoded)
        print("--------------------")
        
        print("\nSuccess! The Graph Encoder and LLM are integrated and executable.")

    except Exception as e:
        print(f"\nIntegration Test Failed: {e}")
        import traceback
        traceback.print_exc()