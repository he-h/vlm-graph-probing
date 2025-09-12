import torch
import torch.nn.functional as F
import unicodedata as ud
import re

model_list = [
    "llava-hf/llava-interleave-qwen-0.5b-hf", # hidden size 1024
    "bczhou/tiny-llava-v1-hf",

    "llava-hf/llava-1.5-7b-hf",
    "llava-hf/llava-1.5-13b-hf",

    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen2.5-VL-32B-Instruct",

    "Salesforce/blip2-opt-2.7b",
    "Salesforce/blip2-opt-6.7b",
    "Salesforce/blip2-flan-t5-xxl", #12B

    "openai/gpt-oss-20b",

    "google/gemma-3-4b-it", # 2560. issue: cannot batch processing
    "google/gemma-3-12b-it", # 3840
    "google/gemma-3-27b-it", # 3840
]


def model_path2name(model_path):
    '''Convert model path to a more user-friendly model name.'''
    
    if model_path == "llava-hf/llava-1.5-7b-hf":
        return "LLaVA-1.5-7B"
    elif model_path == "llava-hf/llava-1.5-13b-hf":
        return "LLaVA-1.5-13B"
    elif model_path == "Qwen/Qwen2.5-VL-3B-Instruct":
        return "Qwen2.5-VL-3B"
    elif model_path == "Qwen/Qwen2.5-VL-7B-Instruct":
        return "Qwen2.5-VL-7B"
    elif model_path == "Qwen/Qwen2.5-VL-32B-Instruct":
        return "Qwen2.5-VL-32B"
    elif model_path == "llava-hf/llava-1.5-13b-hf":
        return "LLaVA-1.5-13B"
    elif model_path == "bczhou/tiny-llava-v1-hf":
        return "Tiny-LLaVA"
    elif model_path == "Salesforce/blip2-opt-2.7b":
        return "BLIP2-OPT-2.7B"
    elif model_path == "Salesforce/blip2-opt-6.7b":
        return "BLIP2-OPT-6.7B"
    elif model_path == "Salesforce/blip2-flan-t5-xxl":
        return "BLIP2-FLAN-T5-XXL"
    elif model_path == "openai/gpt-oss-20b":
        return "GPT-OSS-20B"
    elif model_path == "google/gemma-3-4b-it":
        return "Gemma-3-4B"
    elif model_path == "google/gemma-3-12b-it":
        return "Gemma-3-12B"
    elif model_path == "google/gemma-3-27b-it":
        return "Gemma-3-27B"
    else:
        return model_path.split("/")[-1]


def caption_prompt(choice=0, add_1sent_constraint=False):
    '''Return the caption prompt based on the choice. From short to long. range 0-2'''
    prompts = [
        "Describe the image.",
        "Provide a caption for this image in one sentence.",
        "Provide a detailed caption describing the objects, colors, and relationships in the image."
    ]
    
    candidates = [
        "Caption the image focusing on spatial positions (left, right, top, bottom).",
        "Write a caption using only what is clearly visible. Do not guess or infer.",
    ]

    if add_1sent_constraint:
        return prompts[choice] + " Respond with only one sentence, nothing else."
    return prompts[choice]


def build_1_image_prompt4vlm(base_prompt, model_type="llava"):
    '''Return the caption prompt formatted for VLM input.'''
    if model_type == "llava":
        return f"USER: <image>\n{base_prompt}\nASSISTANT:"
    elif model_type == "qwen":
        return (
            f"<|im_start|>user\n"
            f"<|image|>\n{base_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    elif model_type == "blip" or model_type == "gemma":
        return base_prompt
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
    if model_type == "qwen":
        num_layers = len(model.model.layers)
        hidden_dim = model.config.hidden_size
    elif model_type == "llava" or model_type == "gemma":
        num_layers = len(model.language_model.model.layers)
        hidden_dim = model.config.text_config.hidden_size
    elif model_type == "blip":
        num_layers = len(model.language_model.model.decoder.layers)
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
    # IMPORTANT for batched generation on decoder-only LMs
    tok.padding_side = "left"
    return tok