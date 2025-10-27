import os
import json
import argparse
import warnings
import pickle
from typing import Optional, Tuple, Any

import torch
import numpy as np
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from utils import model_path2name
from model import NeuronGraphExtractor as GraphExtractor

# ---------- CLEVR helpers ----------
CLEVR_COLORS = ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow']
CLEVR_SHAPES = ['cube', 'sphere', 'cylinder']

def split_clevr_question_answer(qa_str: str):
    parts = qa_str.split("?")
    if len(parts) != 2:
        raise ValueError(f"Unexpected QA format: {qa_str}")
    return parts[0] + "?", parts[1].strip().lower()

def classify_clevr_question(question: str, answer: str):
    q = question.lower(); a = answer.lower()
    try:
        v = int(a)
        if 0 <= v < 10: return "counting"
    except Exception:
        pass
    if "color" in q and a in CLEVR_COLORS: return "color"
    if a in CLEVR_SHAPES: return "shape"
    return "unknown"

def constrain_clevr_prompt(question: str, task: str) -> str:
    if task == "color":
        return f"{question} Answer with one word from: {', '.join(CLEVR_COLORS)}. Output exactly one word."
    if task == "counting":
        return f"{question} Answer with a single integer 0-9. Output only the number."
    if task == "existence":
        return f"{question} Answer with 'yes' or 'no' only. Output exactly one word."
    if task == "comparison":
        return f"{question} Answer with 'more', 'fewer', or 'equal' only. Output exactly one word."
    if task == "shape":
        return f"{question} Answer with one word from: {', '.join(CLEVR_SHAPES)}. Output exactly one word."
    return question

# ---------- hook helpers ----------
def _extract_first_tensor(out: Any) -> Tuple[Optional[torch.Tensor], str, Optional[int]]:
    if isinstance(out, torch.Tensor):
        return out, "tensor", None
    if isinstance(out, (tuple, list)):
        for i, x in enumerate(out):
            if isinstance(x, torch.Tensor):
                return x, "tuple" if isinstance(out, tuple) else "list", i
        return None, "tuple" if isinstance(out, tuple) else "list", None
    if hasattr(out, "last_hidden_state") and isinstance(out.last_hidden_state, torch.Tensor):
        return out.last_hidden_state, "other", None
    return None, "other", None

def _replace_tensor(out: Any, new_t: torch.Tensor, kind: str, idx: Optional[int]) -> Any:
    if kind == "tensor":
        return new_t
    if kind in ("tuple", "list"):
        seq = list(out)
        if idx is not None:
            seq[idx] = new_t
        return tuple(seq) if kind == "tuple" else seq
    return new_t

@torch.inference_mode()
def make_token_selection_hook_from_lastvec(last_vec: torch.Tensor, pct: float):
    """
    Build a per-sample hook:
      - Computes token scores |h_i · v| at this block's output
      - Zeros the top-pct token positions (all hidden dims)
    Assumes last_vec is 1D [H] on the correct device/dtype.
    """
    def hook(_m, _inp, out):
        t, kind, idx = _extract_first_tensor(out)
        if t is None or not isinstance(t, torch.Tensor) or t.dim() != 3:
            return out

        # Shapes: either [B,S,H] or [S,B,H]; H is last dim
        H = t.shape[-1]
        if last_vec.numel() != H:
            # dimension mismatch; do nothing
            return out

        # Align dtype/device
        v = last_vec.to(device=t.device, dtype=t.dtype)

        # Compute scores per token: |h_i · v|
        # handle both layouts without transpose
        if t.shape[0] == 1:  # [B,S,H]
            # (1,S,H) · (H,) -> (1,S)
            scores = (t * v).sum(dim=-1).abs().squeeze(0)  # [S]
            S = scores.shape[0]
            k = max(1, int(np.ceil(pct * S)))
            k = min(k, S)
            topk = torch.topk(scores, k=k, largest=True).indices  # [k]
            # Build a token mask (1,S,1)
            token_mask = torch.ones(S, device=t.device, dtype=t.dtype)
            token_mask.index_fill_(0, topk, 0)
            token_mask = token_mask.view(1, S, 1)
            # In-place multiply (fast, no cloning)
            t.mul_(token_mask)
        else:  # assume [S,B,H]
            S = t.shape[0]
            scores = (t * v).sum(dim=-1).abs().mean(dim=1)  # [S], avg over batch dim
            k = max(1, int(np.ceil(pct * S)))
            k = min(k, S)
            topk = torch.topk(scores, k=k, largest=True).indices  # [k]
            token_mask = torch.ones(S, device=t.device, dtype=t.dtype)
            token_mask.index_fill_(0, topk, 0)
            token_mask = token_mask.view(S, 1, 1)
            t.mul_(token_mask)

        return _replace_tensor(out, t, kind, idx)
    return hook

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser("Qwen intervention: select TOKENS by |h·v_last| at chosen layer and zero them")
    ap.add_argument("--data_dir", required=True, help="Folder with complete_dataset.pkl and metadata.json")
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--layer_tag", type=str, default="layer_50",
                    choices=["layer_0", "layer_25", "layer_50", "layer_75", "layer_100"])
    ap.add_argument("--topk_pct", type=float, default=0.05, help="Fraction of tokens to ablate.")
    ap.add_argument("--task", type=str, default=None)
    ap.add_argument("--max_eval", type=int, default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    # Load metadata + saved samples
    meta_path = os.path.join(args.data_dir, "metadata.json")
    pkl_path  = os.path.join(args.data_dir, "complete_dataset.pkl")
    if not (os.path.exists(meta_path) and os.path.exists(pkl_path)):
        raise FileNotFoundError("Missing metadata.json or complete_dataset.pkl in --data_dir")

    with open(meta_path, "r") as f:
        meta = json.load(f)
    with open(pkl_path, "rb") as f:
        saved = pickle.load(f)
    if len(saved) == 0:
        print("No samples in dataset.")
        return

    task = args.task or meta.get("task", "color")
    layer_indices = meta.get("layer_indices")
    if layer_indices is None:
        raise ValueError("metadata.json missing 'layer_indices'.")

    # Rebuild CLEVR (same order as creator)
    print("Reloading CLEVR validation split...")
    clevr = load_dataset("laion/clevr-webdataset", split="validation")
    images, questions, answers = [], [], []
    for s in clevr:
        try:
            q, a = split_clevr_question_answer(s["txt"])
            if classify_clevr_question(q, a) == task:
                images.append(s["jpg"].convert("RGB"))
                questions.append(q)
                answers.append(a)
                if len(images) >= len(saved): break
        except Exception:
            continue

    if len(images) < len(saved):
        warnings.warn("Reconstructed fewer samples than saved; truncating.")
    n_eval = min(len(saved), len(images))
    if args.max_eval is not None:
        n_eval = min(n_eval, args.max_eval)

    # Init model (Qwen)
    torch.set_grad_enabled(False)
    print(f"Loading Qwen model: {args.model_name}")
    extractor = GraphExtractor(model_name=args.model_name, device=args.device)
    extractor.model.eval()

    try:
        qwen_layers = extractor.model.model.layers
    except Exception as e:
        raise RuntimeError("Could not access Qwen blocks at extractor.model.model.layers.") from e

    target_idx = min(layer_indices[args.layer_tag], len(qwen_layers) - 1)
    target_block = qwen_layers[target_idx]

    last_token_key = {
        "layer_0":   "last_token_layer_0",
        "layer_25":  "last_token_layer_25",
        "layer_50":  "last_token_layer_50",
        "layer_75":  "last_token_layer_75",
        "layer_100": "last_token_layer_100",
    }[args.layer_tag]

    print("\n=== INTERVENTION SETTINGS ===")
    print(f"Task: {task}")
    print(f"Layer: {args.layer_tag} -> index {target_idx}")
    print(f"Ablation target: top {args.topk_pct*100:.1f}% TOKENS by |h·v_last| (per sample @ selected layer)")
    print(f"Eval samples: {n_eval}\n")

    correct = 0
    total = 0

    for i in tqdm(range(n_eval), desc="Intervening"):
        sample = saved[i]

        # last-token vector v (1D [H]); keep as torch on CPU first
        last_vec = sample[last_token_key]
        if not torch.is_tensor(last_vec):
            last_vec = torch.as_tensor(last_vec)

        # Build per-sample hook (v will be moved to GPU inside the hook)
        h = target_block.register_forward_hook(make_token_selection_hook_from_lastvec(last_vec, args.topk_pct))

        try:
            img = images[i]; q = questions[i]; ref = answers[i]
            prompt = constrain_clevr_prompt(q, task)
            with torch.inference_mode():
                _, gen = extractor.process_single(img, prompt, max_new_tokens=1)
            pred = gen.lower().strip() if isinstance(gen, str) else str(gen).lower().strip()
            if pred == ref:
                correct += 1
            total += 1
            if args.verbose and (i % 20 == 0):
                print(f"\n[{i}] Q: {q}\nPred: {pred} | Ref: {ref}")
        except Exception as e:
            if args.verbose:
                print(f"Error on sample {i}: {e}")
        finally:
            try: h.remove()
            except Exception: pass
            torch.cuda.empty_cache()

    acc = (correct / total * 100.0) if total > 0 else 0.0
    print("\n" + "="*60)
    print("FINAL INTERVENTION RESULTS (Qwen, token-level via last-token similarity)")
    print("="*60)
    print(f"Samples evaluated: {total}")
    print(f"Accuracy (ablate @ {args.layer_tag}, top {args.topk_pct*100:.1f}% tokens by |h·v_last|): {acc:.2f}%")
    print("="*60)

if __name__ == "__main__":
    main()
