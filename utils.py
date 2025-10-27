import torch
import torch.nn.functional as F
import unicodedata as ud
import re
import numpy as np

model_list = [
    "llava-hf/llava-1.5-7b-hf",
    "llava-hf/llava-1.5-13b-hf",

    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen2.5-VL-32B-Instruct",

    "Salesforce/blip2-opt-2.7b",
    "Salesforce/blip2-opt-6.7b",
    "Salesforce/blip2-flan-t5-xxl", #12B

    "google/gemma-3-4b-it", # 2560
    "google/gemma-3-12b-it", # 3840
    "google/gemma-3-27b-it", # 3840

    "OpenGVLab/InternVL3-1B-hf",
    "OpenGVLab/InternVL3-2B-hf",
    "OpenGVLab/InternVL3-4B-hf",
    "OpenGVLab/InternVL3-8B-hf",
    "OpenGVLab/InternVL3-14B-hf",
    "OpenGVLab/InternVL3-38B-hf",
]


def model_ckpt2name(model_ckpt):
    '''Convert model path to a more user-friendly model name.'''
    
    mapping = {
        "llava-hf/llava-1.5-7b-hf": "LLaVA-1.5-7B",
        "llava-hf/llava-1.5-13b-hf": "LLaVA-1.5-13B",
        "Qwen/Qwen2.5-VL-3B-Instruct": "Qwen2.5-VL-3B",
        "Qwen/Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B",
        "Qwen/Qwen2.5-VL-32B-Instruct": "Qwen2.5-VL-32B",
        "Salesforce/blip2-opt-2.7b": "BLIP2-OPT-2.7B",
        "Salesforce/blip2-opt-6.7b": "BLIP2-OPT-6.7B",
        "Salesforce/blip2-flan-t5-xxl": "BLIP2-FLAN-T5-XXL",
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


def prompt_for_model(base_prompt, model_type="llava"):
    '''Return the caption prompt formatted for VLM input.'''
    if model_type == "llava":
        return f"USER: <image>\n{base_prompt} ASSISTANT:"
    elif model_type == "qwen":
        return (
        "<|im_start|>system\n"
        "You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"<|vision_start|><|image_pad|><|vision_end|>{base_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    elif model_type == "gemma":
        return f"<start_of_image> {base_prompt}"
    elif model_type == "internvl":
        return (
        "<|im_start|>system\n"
        "你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。<|im_end|>\n"
        "<|im_start|>user\n"
        "<image>\n"
        f"{base_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    else:
        raise ValueError(f"Unsupported model type for prompt: {model_type}")

    
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
    
    # Split by any form of "assistant" marker and take the last part
    # This regex matches: assistant\n, ASSISTANT:, assistant:, Assistant:, etc.
    parts = re.split(r'(?:assistant|ASSISTANT|Assistant)[\n:]', s)
    if len(parts) > 1:
        s = parts[-1].strip()
    
    # split s by \n
    s = s.split("\n")[-1].strip()
    
    # Clean up any remaining artifacts
    s = s.replace("<image>", "").replace("Caption:", "").strip()
    
    return s

    

def get_layers_dims(model, model_type):
    if model_type == "qwen" :
        num_layers = model.config.text_config.num_hidden_layers
        hidden_dim = model.config.hidden_size
    elif model_type == "llava" or model_type == "gemma":
        num_layers = model.config.text_config.num_hidden_layers
        hidden_dim = model.config.text_config.hidden_size
    elif model_type == "internvl":
        num_layers = model.config.text_config.num_hidden_layers
        hidden_dim = model.config.text_config.hidden_size
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return num_layers, hidden_dim

def amp_ctx():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    elif torch.cuda.is_available():
        return torch.autocast("cuda", dtype=torch.float16)
    else:
        class _NullCtx:
            def __enter__(self): return None
            def __exit__(self, *args): return False
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
    idxs = np.linspace(0, num_layers - 1, layer_slices + 1)
    idxs = np.floor(idxs).astype(int)
    idxs = np.unique(idxs)
    return idxs.tolist()
