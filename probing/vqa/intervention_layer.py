import os
import json
import argparse
import warnings
import pickle
from typing import List

import torch
import numpy as np
import scipy.sparse as sp
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
    question = question.lower()
    answer = answer.lower()
    try:
        v = int(answer)
        if 0 <= v < 10:
            return "counting"
    except Exception:
        pass
    if "color" in question and answer in CLEVR_COLORS:
        return "color"
    elif answer in CLEVR_SHAPES:
        return "shape"
    else:
        return "unknown"

def constrain_clevr_prompt(question: str, task: str) -> str:
    if task == "color":
        return f"{question} Answer with one word from: {', '.join(CLEVR_COLORS)}. Output exactly one word."
    elif task == "counting":
        return f"{question} Answer with a single integer 0-9. Output only the number."
    elif task == "existence":
        return f"{question} Answer with 'yes' or 'no' only. Output exactly one word."
    elif task == "comparison":
        return f"{question} Answer with 'more', 'fewer', or 'equal' only. Output exactly one word."
    elif task == "shape":
        return f"{question} Answer with one word from: {', '.join(CLEVR_SHAPES)}. Output exactly one word."
    return question

# ---------- graph / hook utils ----------
def _to_numpy(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def topk_degree_nodes(adj_dict, pct: float = 0.05, undirected: bool = True) -> np.ndarray:
    """
    adj_dict: {"num_nodes": int, "edge_index": (2,E), "edge_weight": (E,) or None}
    Returns indices of top-pct nodes by (weighted) degree.
    """
    if not isinstance(adj_dict, dict) or "edge_index" not in adj_dict or "num_nodes" not in adj_dict:
        raise ValueError("Expected dict with keys {'num_nodes','edge_index',('edge_weight')}")

    N = int(adj_dict["num_nodes"])
    ei = _to_numpy(adj_dict["edge_index"])   # shape (2, E)
    ew = _to_numpy(adj_dict.get("edge_weight"))

    if ei.ndim != 2 or ei.shape[0] != 2:
        raise ValueError(f"edge_index expected shape (2, E), got {ei.shape}")

    E = ei.shape[1]
    if ew is None:
        ew = np.ones(E, dtype=np.float32)
    else:
        ew = ew.astype(np.float32, copy=False)

    src = ei[0].astype(np.int64, copy=False)
    dst = ei[1].astype(np.int64, copy=False)

    # degree = sum of incident edge weights; treat graph as undirected by default
    deg = np.bincount(src, weights=ew, minlength=N)
    if undirected:
        deg += np.bincount(dst, weights=ew, minlength=N)

    k = max(1, int(np.ceil(pct * N)))
    idx = np.argpartition(-deg, k - 1)[:k]
    idx = idx[np.argsort(-deg[idx])]  # optional, stable order
    return idx.astype(int)

def make_ablation_hook(token_idx: np.ndarray):
    token_idx = np.unique(np.asarray(token_idx, dtype=int))

    def hook(_module, _inp, out):
        tensor = out[0] if isinstance(out, tuple) else out
        if not isinstance(tensor, torch.Tensor) or tensor.dim() != 3:
            return out

        # Handle [B,S,H] or [S,B,H]
        if tensor.shape[0] == 1:          # [B, S, H]
            seq_len = tensor.shape[1]
            keep = token_idx[token_idx < seq_len]
            if keep.size == 0:
                return out
            out_ = tensor.clone()
            out_[:, keep, :] = 0
            return (out_,) + out[1:] if isinstance(out, tuple) else out_
        else:                              # assume [S, B, H]
            seq_len = tensor.shape[0]
            keep = token_idx[token_idx < seq_len]
            if keep.size == 0:
                return out
            out_ = tensor.clone()
            out_[keep, :, :] = 0
            return (out_,) + out[1:] if isinstance(out, tuple) else out_

    return hook


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser("Qwen intervention on CLEVR (ablate top-degree tokens at one layer)")
    ap.add_argument("--data_dir", required=True, help="Folder containing complete_dataset.pkl and metadata.json")
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")  # or your exact Qwen path
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--layer_tag", type=str, default="layer_50",
                    choices=["layer_0", "layer_25", "layer_50", "layer_75", "layer_100"])
    ap.add_argument("--topk_pct", type=float, default=0.05)
    ap.add_argument("--task", type=str, default=None)
    ap.add_argument("--max_eval", type=int, default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

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

    # rebuild CLEVR (same filtering/order as creator)
    print("Reloading CLEVR validation split...")
    clevr = load_dataset("laion/clevr-webdataset", split="validation")

    images, questions, answers = [], [], []
    valid = 0
    for s in clevr:
        try:
            q, a = split_clevr_question_answer(s["txt"])
            if classify_clevr_question(q, a) == task:
                images.append(s["jpg"].convert("RGB"))
                questions.append(q)
                answers.append(a)
                valid += 1
                if valid >= len(saved):
                    break
        except Exception:
            continue

    if len(images) < len(saved):
        warnings.warn("Reconstructed fewer samples than saved; truncating evaluations.")
    n_eval = min(len(saved), len(images))
    if args.max_eval is not None:
        n_eval = min(n_eval, args.max_eval)

    # init model
    print(f"Loading Qwen model: {args.model_name}")
    extractor = GraphExtractor(model_name=args.model_name, device=args.device)

    # Qwen blocks live at model.model.layers
    try:
        qwen_layers = extractor.model.model.layers
    except Exception as e:
        raise RuntimeError(
            "Could not access Qwen blocks at extractor.model.model.layers. "
            "Please adjust if your wrapper differs."
        ) from e

    target_idx = min(layer_indices[args.layer_tag], len(qwen_layers)-1)
    target_block = qwen_layers[target_idx]

    print("\n=== INTERVENTION SETTINGS ===")
    print(f"Task: {task}")
    print(f"Layer: {args.layer_tag} -> index {target_idx}")
    print(f"Ablation: top {args.topk_pct*100:.1f}% tokens by graph degree (per-sample)")
    print(f"Eval samples: {n_eval}\n")

    correct = 0
    total = 0

    graph_key = {
        "layer_0": "graph_layer_0",
        "layer_25": "graph_layer_25",
        "layer_50": "graph_layer_50",
        "layer_75": "graph_layer_75",
        "layer_100": "graph_layer_100",
    }[args.layer_tag]

    for i in tqdm(range(n_eval), desc="Intervening"):
        sample = saved[i]
        adj = sample[graph_key]
        top_idx = topk_degree_nodes(adj, pct=args.topk_pct)  # now works with your dict format
        h = target_block.register_forward_hook(make_ablation_hook(top_idx))

        try:
            img = images[i]
            q = questions[i]
            ref = answers[i]
            prompt = constrain_clevr_prompt(q, task)
            _, gen = extractor.process_single(img, prompt)
            pred = gen.lower().strip() if isinstance(gen, str) else str(gen).lower().strip()
            if pred == ref:
                correct += 1
            total += 1
            if args.verbose and (i % 20 == 0):
                print(f"\n[{i}] Q: {q}\nPred: {pred} | Ref: {ref} | zeroed={len(top_idx)} tokens")
        except Exception as e:
            if args.verbose:
                print(f"Error on sample {i}: {e}")
        finally:
            try:
                h.remove()
            except Exception:
                pass
            torch.cuda.empty_cache()

    acc = (correct / total * 100.0) if total > 0 else 0.0
    print("\n" + "="*60)
    print("FINAL INTERVENTION RESULTS (Qwen)")
    print("="*60)
    print(f"Samples evaluated: {total}")
    print(f"Accuracy (ablate @ {args.layer_tag}, top {args.topk_pct*100:.1f}%): {acc:.2f}%")
    print("="*60)

if __name__ == "__main__":
    main()
