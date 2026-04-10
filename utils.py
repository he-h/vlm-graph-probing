from __future__ import annotations

import unicodedata as ud
import re
import numpy as np
import math

import torch
import torch.nn.functional as F
from torch import inference_mode

model_list = [
    "llava-hf/llava-1.5-7b-hf",
    "llava-hf/llava-1.5-13b-hf",
    "llava-hf/llava-v1.6-mistral-7b-hf",

    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen2.5-VL-32B-Instruct",

    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-it",

    "OpenGVLab/InternVL3-1B-hf",
    "OpenGVLab/InternVL3-2B-hf",
    "OpenGVLab/InternVL3-4B-hf",
    "OpenGVLab/InternVL3-8B-hf",
    "OpenGVLab/InternVL3-14B-hf",
    "OpenGVLab/InternVL3-38B-hf",
]


def require_model_deps():
    """No-op guard kept for call-site compatibility."""
    pass

def require_dataset_deps():
    """No-op guard kept for call-site compatibility."""
    pass


def model_ckpt2name(model_ckpt):
    '''Convert model path to a more user-friendly model name.'''
    
    mapping = {
        "llava-hf/llava-1.5-7b-hf": "LLaVA-1.5-7B",
        "llava-hf/llava-1.5-13b-hf": "LLaVA-1.5-13B",
        "llava-hf/llava-v1.6-mistral-7b-hf": "LLaVA-v1.6-Mistral-7B",
        "Qwen/Qwen2.5-VL-3B-Instruct": "Qwen2.5-VL-3B",
        "Qwen/Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B",
        "Qwen/Qwen2.5-VL-32B-Instruct": "Qwen2.5-VL-32B",
        "google/gemma-3-4b-it": "Gemma-3-4B",
        "google/gemma-3-12b-it": "Gemma-3-12B",
        "google/gemma-3-27b-it": "Gemma-3-27B",
        "OpenGVLab/InternVL3-1B-hf": "InternVL3-1B",
        "OpenGVLab/InternVL3-2B-hf": "InternVL3-2B",
        "OpenGVLab/InternVL3-4B-hf": "InternVL3-4B",
        "OpenGVLab/InternVL3-8B-hf": "InternVL3-8B",
        "OpenGVLab/InternVL3-14B-hf": "InternVL3-14B",
        "OpenGVLab/InternVL3-38B-hf": "InternVL3-38B",
    }

    # Handle direct matches first
    if model_ckpt in mapping:
        return mapping[model_ckpt]

    # Default: return last part of the path
    return model_ckpt.split("/")[-1]



def get_activation(activation):
    """Get activation function based on string input"""
    activations = {
        'relu': F.relu,
        'elu': F.elu,
        'leaky_relu': F.leaky_relu,
        'tanh': torch.tanh,
        'sigmoid': torch.sigmoid,
        'none': lambda x: x  # No activation (linear)
    }
    return activations.get(activation.lower(), F.relu)


def norm_text(s: str) -> str:
    """Simple normalization that extracts just the final response."""
    s = ud.normalize("NFKC", str(s)).strip()
    
    parts = re.split(r'(?:assistant|ASSISTANT|Assistant)[\n:]', s)
    if len(parts) > 1:
        s = parts[-1].strip()
    
    s = s.split("\n")[-1].strip()
    s = s.replace("<image>", "").replace("Caption:", "").strip()
    
    return s


def sanitize(s: str) -> str:
    """Normalize unicode and collapse whitespace."""
    if not isinstance(s, str):
        s = str(s)
    s = ud.normalize("NFKC", s)
    s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def sanitize_preds_refs(preds, refs):
    """Sanitize prediction strings and nested reference strings."""
    preds = [sanitize(p) for p in preds]
    refs = [[sanitize(r) for r in ref_list] for ref_list in refs]
    return preds, refs




SUPPORTED_FAMILIES = {"llava", "qwen", "gemma", "internvl"}

def get_layers_dims(model, model_family):
    if model_family not in SUPPORTED_FAMILIES:
        raise ValueError(f"Unknown model type: {model_family}")
    num_layers = model.config.text_config.num_hidden_layers
    hidden_dim = model.config.text_config.hidden_size
    return num_layers, hidden_dim

class _NullCtx:
    """No-op context manager used when CUDA is unavailable."""
    def __enter__(self): return None
    def __exit__(self, *args): return False


def amp_ctx():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    elif torch.cuda.is_available():
        return torch.autocast("cuda", dtype=torch.float16)
    else:
        return _NullCtx()


def ensure_tokenizer_defaults(processor):
    tok = getattr(processor, "tokenizer", None)
    if tok is None:
        return None
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    return tok


def intervene(hidden, indices, mode='mask', scale=0.5, donor=None):
    mask = torch.zeros_like(hidden)
    mask[:, indices] = 1
    if mode == 'mask':
        hidden = hidden * (1 - mask)
    elif mode == 'scale':
        hidden = hidden * (1 + scale * mask)
    elif mode == 'replace' and donor is not None:
        hidden = hidden * (1 - mask) + donor * mask
    return hidden


def evenly_spaced_layers(num_layers: int, layer_slices: int):
    if layer_slices == -1:
        return [int(num_layers/2 - 1)]
    idxs = np.linspace(0, num_layers - 1, layer_slices + 1)
    idxs = np.floor(idxs).astype(int)
    idxs = np.unique(idxs)
    return idxs.tolist()

def get_blocks(model, model_family):
    if model_family not in SUPPORTED_FAMILIES:
        raise ValueError(f"Unknown model type: {model_family}")
    return model.language_model.layers

def sparsify_graph(num_nodes, edge_index, edge_weight, sparse_level=0.5):
    """
    Sparsify by selecting top-k edges by |weight| without constructing N×N.
    Assumes: k_target = floor((1 - sparse_level) * N * N) < E  (so there are enough edges to choose from)

    - Zeros are included in the ranking but not saved.
    - Final kept count K <= k_target (if many zeros among top-ranked).
    
    Args:
        num_nodes (int): N
        edge_index (LongTensor [2, E])
        edge_weight (Tensor [E])
        sparse_level (float in [0, 1]): e.g., 0.9 -> keep top 10% of N^2

    Returns:
        kept_edge_index (LongTensor [2, K])
        kept_edge_weight (Tensor [K])
    """
    assert 0.0 <= sparse_level <= 1.0, f"sparse_level={sparse_level} must be within [0,1]"


    N = int(num_nodes)

    # Desired number of entries to keep based on full N^2 space 
    k_target = int(math.floor((1.0 - float(sparse_level)) * N * N))

    # Rank all existing edges by |w| (zeros included in ranking)
    order = torch.argsort(edge_weight.abs(), descending=True)

    # Select top nonzero edges in that order up to k_target
    ordered_weights = edge_weight[order]
    nonzero_mask_ordered = (ordered_weights != 0)
    chosen_ordered = order[nonzero_mask_ordered][:k_target]

    kept_edge_index = edge_index[:, chosen_ordered]
    kept_edge_weight = edge_weight[chosen_ordered]

    return kept_edge_index, kept_edge_weight
