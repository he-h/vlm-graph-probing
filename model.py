from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data
from transformers import LlavaForConditionalGeneration, AutoProcessor, Qwen2_5_VLForConditionalGeneration, Gemma3ForConditionalGeneration, AutoModel, AutoModelForImageTextToText
import warnings
import numpy as np
import scipy.sparse as sp
import os
import re
from typing import List, Tuple, Dict, Any, Optional

from utils import *

class GCNPredictor(nn.Module):
    def __init__(self, 
                 num_nodes, 
                 embedding_dim=128, 
                 hidden_dim=256,
                 fc_hidden_dim=256,
                 num_layers=2,
                 activation='relu',
                 use_activation_final=False,
                 dropout=0.0,
                 edge_weighted=True,
                 task='regression'):
        """
        GCN Predictor with configurable layers and activation
        
        Args:
            num_nodes: Number of unique nodes in the graph
            embedding_dim: Dimension of node embeddings
            hidden_dim: Dimension of hidden layers
            num_layers: Number of GCN layers (minimum 1)
            activation: Activation function ('relu', 'elu', 'leaky_relu', 'tanh', 'none')
            use_activation_final: Whether to apply activation after the final GCN layer
            dropout: Dropout probability (0.0 means no dropout)
        """
        super(GCNPredictor, self).__init__()
        
        self.num_layers = num_layers
        self.use_activation_final = use_activation_final
        self.dropout = dropout
        self.edge_weighted = edge_weighted
        
        # Node embedding
        self.node_embedding = nn.Embedding(num_nodes, embedding_dim)
        
        # Define activation function
        self.activation = get_activation(activation)
        
        # Build GCN layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(embedding_dim, hidden_dim))
        
        # Additional layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Fully connected layers
        # After pooling we have hidden_dim * 2 (avg + max pool)
        self.fc1 = nn.Linear(hidden_dim * 2, fc_hidden_dim)
        self.fc2 = nn.Linear(fc_hidden_dim, 1)
        
        # Optional: Batch normalization layers
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        
    
    def forward(self, node_ids, edge_index, edge_weight=None, batch=None):
        # Node embeddings
        x = self.node_embedding(node_ids)
        
        # Apply GCN layers with activation
        for i, conv in enumerate(self.convs):
            if self.edge_weighted and edge_weight is not None:
                x = conv(x, edge_index, edge_weight)
            else:
                x = conv(x, edge_index)
            
            # Apply batch normalization (optional)
            # Uncomment if you want to use batch norm
            # x = self.bn_layers[i](x)
            
            # Apply activation (except possibly on last layer)
            if i < self.num_layers - 1 or self.use_activation_final:
                x = self.activation(x)
            
            # Apply dropout (except on last layer)
            if self.dropout > 0 and i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        avg_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        x = torch.cat([avg_pool, max_pool], dim=1)
        
        # Fully connected layers
        x = self.fc1(x)
        # x = self.activation(x)
        
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer (no activation for regression)
        x = self.fc2(x)
        
        return x

class MLPPredictor(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hidden_dim=256,
                 num_layers=2,
                 activation='relu',
                 dropout=0.0,
                 task='regression'):
        """
        MLP Predictor with configurable layers and activation
        
        Args:
            hidden_dim: Dimension of hidden layers
            num_layers: Number of hidden layers (minimum 1)
            activation: Activation function ('relu', 'elu', 'leaky_relu', 'tanh', 'none')
            dropout: Dropout probability (0.0 means no dropout)
        """
        super(MLPPredictor, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.task = task
        self.activation = get_activation(activation)

        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim, 1))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.Linear(hidden_dim, 1))
        
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.num_layers - 1:
                x = self.activation(x)
                if self.dropout > 0:
                    x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


def load_model(model_ckpt: str, device: str = "cuda:0"):
    lower = model_ckpt.lower()
    # check do we need eval() on the model
    if "llava" in lower:
        model_type = "llava"
        model = LlavaForConditionalGeneration.from_pretrained(
            model_ckpt, dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(device).eval()
    elif "qwen" in lower:
        model_type = "qwen"
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_ckpt, dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(device).eval()
    elif "gemma" in lower:
        model_type = "gemma"
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_ckpt#, dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(device).eval()
    elif "internvl" in lower:
        model_type = "internvl"
        model = AutoModelForImageTextToText.from_pretrained(
            model_ckpt, dtype=torch.float16 # TODO: should be bfloat16 if on A100 or H100
        ).to(device).eval()
    else:
        raise ValueError(f"Unsupported model family for: {model_ckpt} (expected llava/qwen/gemma)")

    processor = AutoProcessor.from_pretrained(model_ckpt, use_fast=True)
    return model_type, model.eval(), processor


@torch.inference_mode()
def corr_graph_torch(hs: torch.Tensor, sparse_level: float = 0.9) -> Dict[str, Any]:
    """
    Fast correlation graph on GPU.

    Args:
      hs: [seq, hidden] float tensor on CUDA (or CPU; CUDA is much faster)
      sparse_level: keep top (1 - sparse_level) fraction of |corr| edges

    Returns:
      dict with COO graph arrays on CPU:
        {
          "num_nodes": H,
          "edge_index": np.ndarray [2, E],
          "edge_weight": np.ndarray [E],
        }
    """
    if hs.dim() != 2:
        raise ValueError(f"Expected 2D [seq, hidden], got {tuple(hs.shape)}")
    x = hs - hs.mean(dim=0, keepdim=True)                 # [S, H]
    denom = x.norm(dim=0).clamp_min(1e-8)                 # [H]
    x = x / denom
    C = (x.t() @ x).clamp(-1, 1)                          # [H, H] ~ Pearson if centered
    H = C.shape[0]
    # keep top (1 - sparse_level) edges by |corr|
    k = max(1, int((1.0 - sparse_level) * H * H))
    thr = torch.topk(C.abs().flatten(), k).values.min()
    mask = (C.abs() >= thr)
    # remove self loops
    # mask.fill_diagonal_(False)
    ei = mask.nonzero(as_tuple=False)                     # [E, 2]
    ew = C[mask]                                          # [E]
    edge_index = ei.t().contiguous().cpu().numpy()        # [2, E]
    edge_weight = ew.float().cpu().numpy()                # [E]
    return {"num_nodes": int(H), "edge_index": edge_index, "edge_weight": edge_weight}



class NeuronGraphExtractor:
    """
    Generic extractor for VLM families (LLaVA, Qwen2.5-VL, Gemma-3).

    - Batched multimodal generation for ANY text task (caption, VQA, etc.)
    - Returns hidden states (per layer) and decoded texts
    - Provides fast GPU correlation graph utility
    """

    def __init__(self, model_ckpt: str, device: str = "cuda:0"):
        self.device = device
        self.model_ckpt = model_ckpt
        self.model_type, self.model, self.processor = load_model(model_ckpt, device)
        self.tokenizer = ensure_tokenizer_defaults(self.processor)
        self.num_layers, self.hidden_dim = get_layers_dims(self.model, self.model_type)

        # Optional perf knobs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            from torch.backends.cuda import sdp_kernel
            sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True)
        except Exception:
            pass  # not critical

        print(f"[Extractor] Loaded {self.model_ckpt} ({self.model_type})")
        print(f"[Extractor] Layers: {self.num_layers}, Hidden dim: {self.hidden_dim}")

    @torch.inference_mode()
    def _prepare_inputs(
        self, images: List[Any], texts: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Turn (images, texts) -> model inputs for different families.
        """
        if len(images) != len(texts):
            raise ValueError(f"images ({len(images)}) and texts ({len(texts)}) must have same length")

        # if self.model_type == "qwen":
        all_texts = []
        for img, txt in zip(images, texts):
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": txt},
                    ],
                }
            ]
            t = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            all_texts.append(t)

        inputs = self.processor(
            text=all_texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )

        inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
        return inputs

    @torch.inference_mode()
    def process_batch(
        self,
        images: List[Any],
        texts: List[str],
        max_new_tokens: int = 50,
        min_new_tokens: int = 1,
        do_sample: bool = False,
        num_beams: int = 1,
        no_repeat_ngram_size: Optional[int] = None,
    ) -> Tuple[List[torch.Tensor], List[str]]:
        """
        One-stop batched forward+generate:
          - returns (hidden_states_per_layer, decoded_texts)
            where hidden_states_per_layer is a list of [B, T, H] tensors
        """
        # Prepare inputs (handles Qwen chat template vs others)
        inputs = self._prepare_inputs(images, texts)

        # 1) Forward pass to capture hidden states
        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states

        # 2) Generate batched continuations
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=do_sample,
            num_beams=num_beams,
            use_cache=True,
        )
        if self.tokenizer is not None:
            gen_kwargs["eos_token_id"] = self.tokenizer.eos_token_id
            gen_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        if no_repeat_ngram_size is not None:
            gen_kwargs["no_repeat_ngram_size"] = int(no_repeat_ngram_size)

        with amp_ctx():
            out_ids = self.model.generate(**inputs, **gen_kwargs)

        # 3) Decode
        if hasattr(self.processor, "batch_decode"):
            decoded = self.processor.batch_decode(out_ids, skip_special_tokens=True)
        else:
            decoded = [self.processor.decode(t, skip_special_tokens=True) for t in out_ids]
        decoded = [norm_text(s) for s in decoded]

        return hidden_states, decoded


    @torch.inference_mode()
    def process_single(
        self,
        image: Any,
        text: str,
        max_new_tokens: int = 50,
        min_new_tokens: int = 1,
        do_sample: bool = False,
        num_beams: int = 1,
        no_repeat_ngram_size: Optional[int] = None,
    ) -> Tuple[List[torch.Tensor], str]:
        """
        Returns:
        - hidden_states_per_layer: list of [1, T_prompt, H] for vision+prompt only
        - generated_text: decoded string of continuation
        """

        message = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text},
            ],
        }]

        prompts = self.processor.apply_chat_template(
            [message], tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=prompts, 
            images=[image],
            return_tensors="pt",
        )

        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}

        forward_out = self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=True
        )

        hidden_states_per_layer = list(forward_out.hidden_states) 

        gen_out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=do_sample,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            eos_token_id=getattr(self.processor.tokenizer, "eos_token_id", None),
            pad_token_id=getattr(self.processor.tokenizer, "pad_token_id", None),
            return_dict_in_generate=True,
            output_scores=False,           # flip to True if you want token scores
            output_hidden_states=False     # set True if you also want per-step gen hidden states
        )

        tokenizer = getattr(self.processor, "tokenizer", None) or getattr(self, "tokenizer", None)

        # gen_out.sequences includes the prompt + generated tokens
        sequences = gen_out.sequences
        # Compute prompt length from inputs (assumes causal LM with input_ids)
        prompt_len = inputs["input_ids"].shape[-1]
        generated_ids = sequences[:, prompt_len:]
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return hidden_states_per_layer, generated_text



    @torch.inference_mode()
    def compute_correlation_graph(
        self,
        hidden_states_per_layer: List[torch.Tensor],
        sample_index: int,
        layer_indices: Dict[str, int],
        sparse_level: float = 0.9,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Build correlation graphs for a specific sample across selected layers.

        Args:
          hidden_states_per_layer: list of tensors [B, T, H]
          sample_index: which sample in the batch
          layer_indices: mapping {"layer_0": 0, "layer_mid": X, "layer_last": L-1}
          sparse_level: sparsification quantile on |corr|

        Returns:
          { layer_name: {num_nodes, edge_index, edge_weight} }
        """
        graphs = {}
        L = len(hidden_states_per_layer)
        for name, idx in layer_indices.items():
            idx = max(0, min(idx, L - 1))
            hs_bt = hidden_states_per_layer[idx]           # [B, T, H]
            hs = hs_bt[sample_index]                       # [T, H]
            if hs.device.type != "cuda" and torch.cuda.is_available():
                hs = hs.to("cuda", non_blocking=True)
            graph = corr_graph_torch(hs, sparse_level=sparse_level)
            graphs[name] = graph
        return graphs
