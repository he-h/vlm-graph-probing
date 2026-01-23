import os
import json
import argparse
import pickle
import gc
import numpy as np
import torch
from tqdm import tqdm

from utils import *  
from model import NeuronGraphExtractor as GraphExtractor
from dataset import prepare_vlm_data, get_candidate_answers


# -------------------------------
# Minimal helpers
# -------------------------------

def topk_indices_abs(v: np.ndarray, pct: float) -> np.ndarray:
    """Return indices of top-k% by absolute value from a 1D array."""
    v = np.asarray(v, dtype=np.float32).ravel()
    H = v.shape[0]
    k = max(1, int(np.ceil(pct * H)))
    idx = np.argpartition(np.abs(v), -k)[-k:]
    idx = idx[np.argsort(-np.abs(v[idx]))]
    return idx.astype(int)

def _parse_graph_item(graph_item):
    """
    Parse a per-sample neuron–neuron graph stored as:
        [H, edge_index(2,E), edge_weight(E or None)]

    Returns:
        (H, edge_index[np.int64 (2,E)], edge_weight[np.float32 (E,)])
    """
    H = int(graph_item[0])
    ei = np.asarray(graph_item[1], dtype=np.int64)
    ew = np.asarray(graph_item[2], dtype=np.float32) if graph_item[2] is not None else np.ones(ei.shape[1], dtype=np.float32)
    return H, ei, ew

def topk_neurons_by_degree(graph_item, pct: float) -> np.ndarray:
    """Top-k% neurons by (undirected, weighted) degree from a per-sample neuron–neuron graph."""
    H, ei, ew = _parse_graph_item(graph_item)  # ei: (2,E)
    deg = np.zeros(H, dtype=np.float32)
    np.add.at(deg, ei[0], ew)
    np.add.at(deg, ei[1], ew)
    k = max(1, int(np.ceil(pct * H)))
    idx = np.argpartition(deg, -k)[-k:]
    idx = idx[np.argsort(-deg[idx])]
    return idx.astype(int)


# -------------------------------
# Hook utilities
# -------------------------------

@torch.inference_mode()
def make_mask_hook(mask_1d: torch.Tensor):
    """
    Zero fixed hidden dims using a prebuilt 1×H mask (broadcasts to all tokens).
    Assumes block output is a Tensor whose last dim is H (e.g., [B,S,H] or [S,B,H]).
    """
    def hook(_m, _inp, out):
        t = out[0] if isinstance(out, tuple) else out
        if not isinstance(t, torch.Tensor) or t.dim() < 2:
            return out
        t.mul_(mask_1d)  # in-place broadcast over all leading dims
        return (t,) if isinstance(out, tuple) else t
    return hook

class HookCtx:
    """Context manager to ensure hook removal even on exceptions."""
    def __init__(self, module, hook_fn):
        self.module = module
        self.hook_fn = hook_fn
        self.handle = None
    def __enter__(self):
        self.handle = self.module.register_forward_hook(self.hook_fn)
        return self.handle
    def __exit__(self, exc_type, exc, tb):
        try:
            if self.handle is not None:
                self.handle.remove()
        except Exception:
            pass


# -------------------------------
# Main
# -------------------------------

def main():
    ap = argparse.ArgumentParser("Per-sample neuron intervention (random / last_abs / graph_degree) with layer slices.")
    ap.add_argument("--prepared_data_path", required=True,
                    help="Folder with last_token_layer_{L}.npy and graphs_layer_{L}.pkl (+ metadata.json)")
    ap.add_argument("--dataset", required=True, help="Passed to prepare_vlm_data (e.g., 'clevr')")
    ap.add_argument("--category", default="color", help="e.g., color/counting for VQA")
    ap.add_argument("--model_ckpt", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", default="cuda:0")

    # Keep your slices-based layer selection
    ap.add_argument("--layer_slices", type=int, default=4,
                    help="K slices -> K+1 evenly spaced layers (incl. first & last)")

    ap.add_argument("--selector", choices=["random", "last_abs", "graph_degree"], default="last_abs",
                    help="All are per-sample, neuron-level ablations.")
    ap.add_argument("--graph_type", choices=["all", "text"], default="all", help="If using graph_degree, which graph to use.")
    ap.add_argument("--topk_pct", type=float, default=0.10, help="Fraction of neurons to ablate (0-1).")
    ap.add_argument("--max_eval", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # (Optional) read metadata if you need info; we don't rely on it here.
    meta_path = os.path.join(args.prepared_data_path, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            _ = json.load(f)

    torch.set_grad_enabled(False)
    extractor = GraphExtractor(model_ckpt=args.model_ckpt, device=args.device)
    model = extractor.model
    blocks = get_blocks(model, extractor.model_family) 

    # Evenly spaced absolute layers (assumes files exist)
    selected_layers = evenly_spaced_layers(len(blocks), args.layer_slices)

    # Infer N,H from the first selected layer’s last-token file
    probe_L = selected_layers[0]
    lt_probe = np.load(os.path.join(args.prepared_data_path, f"last_token_layer_{probe_L}.npy"))
    N, H = lt_probe.shape
    if args.max_eval is not None:
        N = min(N, args.max_eval)

    # Prepare triplets (image, prompt, ref)
    data = prepare_vlm_data(dataset=args.dataset, num_samples=N, category=args.category)
    if len(data) < N:
        N = len(data)

    results = {}

    # Single inference context for speed
    with torch.inference_mode():
        for L in selected_layers:
            print(f"\n== Layer {L} | selector={args.selector} | topk={args.topk_pct*100:.1f}% ==")

            # Load per-sample last-token matrix for this layer
            last_token_mat = np.load(os.path.join(args.prepared_data_path, f"last_token_layer_{L}.npy"))[:N]  # [N,H]

            # Load per-sample graphs if needed
            graph_list = None
            if args.selector == "graph_degree":
                if args.graph_type == "all":
                    with open(os.path.join(args.prepared_data_path, f"graphs_layer_{L}.pkl"), "rb") as f:
                        graph_list = pickle.load(f)[:N]  # list length N
                elif args.graph_type == "text":
                    with open(os.path.join(args.prepared_data_path, f"text_graphs_layer_{L}.pkl"), "rb") as f:
                        graph_list = pickle.load(f)[:N]

            block = blocks[L]
            correct = total = 0
            k = max(1, int(np.ceil(args.topk_pct * H)))
            device = args.device

            # --------- Precompute per-sample neuron indices & masks (CPU -> CUDA once) ----------
            idx_list = [None] * N
            if args.selector == "random":
                for i in range(N):
                    # different RNG per sample for reproducibility
                    idx_list[i] = np.random.default_rng(args.seed + i).choice(H, size=k, replace=False)
            elif args.selector == "last_abs":
                for i in range(N):
                    idx_list[i] = topk_indices_abs(last_token_mat[i], args.topk_pct)
            else:  # graph_degree
                for i in range(N):
                    idx_list[i] = topk_neurons_by_degree(graph_list[i], args.topk_pct)

            mask_tensors = []
            for i in range(N):
                idx = np.asarray(idx_list[i], dtype=np.int64)
                # fp16 is fine for mask; last dim broadcast
                mask = torch.ones(H, dtype=torch.float16, device=device)
                if idx.size > 0:
                    mask.index_fill_(0, torch.as_tensor(idx, device=device), 0)
                mask_tensors.append(mask)
            # ------------------------------------------------------------------------------------

            candidates = get_candidate_answers(args.dataset, category=args.category)

            for i in tqdm(range(N), desc=f"Layer {L}", disable=True):
                img, prompt, ref = data[i]

                hook = make_mask_hook(mask_tensors[i])
                with HookCtx(block, hook):
                    _, gen, logits, _ = extractor.process_single(img, prompt, max_new_tokens=1, output_logits=True)

                output = gen.lower().strip() if isinstance(gen, str) else str(gen).lower().strip()
                probs = torch.nn.functional.log_softmax(logits[:, -1, :], dim=-1)
                tok = extractor.processor.tokenizer
                ids = [tok(c, add_special_tokens=False)["input_ids"][0] for c in candidates]
                scores = [probs[0, i].item() for i in ids]
                pred = candidates[int(np.argmax(scores))]

                gold = ref.lower().strip() if isinstance(ref, str) else str(ref).lower().strip()
                correct += int(pred == gold)
                total += 1

                if args.verbose and (i % 100 == 0):
                    zeroed = int((mask_tensors[i] == 0).sum().item())
                    print(f"[{i}] zeroed={zeroed} | output={output} | pred={pred} | ref={gold}")

            acc = (100.0 * correct / total) if total > 0 else 0.0
            results[L] = acc
            print(f"[Layer {L}] Accuracy: {acc:.2f}% over {total} samples")

            # Free big per-layer arrays promptly
            del last_token_mat, graph_list, idx_list, mask_tensors
            gc.collect()

    print(f"\n=== SUMMARY ({args.selector} per-sample neuron hooks) ===")
    for L in sorted(results):
        print(f"Layer {L}: {results[L]:.2f}%")


if __name__ == "__main__":
    main()
