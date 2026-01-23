import torch
import numpy as np
from datasets import load_dataset
from PIL import Image
import json
from tqdm import tqdm
import pickle
import scipy.sparse as sp
import os
import warnings
import argparse

from utils import *
from metrics import spice_scores, meteor_scores, rougeL_scores, bertscore_f1, sanitize_preds_refs
from model import NeuronGraphExtractor as GraphExtractor
from model import build_corr_graph
from dataset import prepare_vlm_data


def create_coco_dataset(
    num_samples=1000,
    model_ckpt="llava-hf/llava-1.5-7b-hf",
    output_dir="probing_dataset",
    prompt_choice=0,
    verbose=False,
    device="cuda:0",
    sparse_level=0.9,
    log_every=25,
    layer_slices=4,
):
    """
    Create a graph probing dataset for a VLM with COCO captions.

    - layer selection uses `layer_slices` to produce K+1 evenly-spaced layers
    - per-layer artifacts are saved in separate files:
        graphs_layer_<L>.pkl : list[ [num_nodes, edge_index, edge_weight] ]
        last_token_layer_<L>.npy : float32 array [N, H]
    - metrics saved as separate arrays: meteor.npy, rougeL.npy, spice.npy
    """
    model_prefix = model_ckpt2name(model_ckpt)
    out_dir = f"data/{model_prefix}_prompt_{prompt_choice}_sparsity_{int(sparse_level * 100)}_{output_dir}"
    os.makedirs(out_dir, exist_ok=True)

    data = prepare_vlm_data(dataset="coco", num_samples=num_samples, task="caption", prompt_choice=prompt_choice)

    print(f"Initializing {model_ckpt} model...")
    extractor = GraphExtractor(model_ckpt=model_ckpt, device=device)

    # Pick evenly spaced layers
    selected_layers = evenly_spaced_layers(extractor.num_layers, layer_slices)
    if verbose:
        print(f"Selected layers (total {extractor.num_layers}): {selected_layers}")

    # Per-layer collectors
    graphs_by_layer = {L: [] for L in selected_layers}           # list of dicts per sample
    last_token_by_layer = {L: [] for L in selected_layers}       # list of vectors per sample

    # For metrics
    all_preds, all_refs = [], []
    missing_captions = 0

    for sample_idx, (image, prompt, references) in enumerate(tqdm(data, desc="Processing samples")):
        try:
            hidden_states_all, gen_caption = extractor.process_single(image, prompt)
            if not gen_caption:
                missing_captions += 1
                if verbose:
                    print(f"Sample {sample_idx}: No caption generated")

            # hidden_states_all: list of [1, seq, hidden]
            hidden_states_all = [h.detach() for h in hidden_states_all]
            _, seq_len, hidden_dim = hidden_states_all[-1].shape

            # Build graphs & last-token vectors for selected layers
            for L in selected_layers:
                L = min(L, len(hidden_states_all) - 1)
                hs = hidden_states_all[L][0]  # [seq, hidden]

                # Graph (sparse by correlation thresholding)
                g = build_corr_graph(hs, sparse_level=sparse_level)
                # Convert to portable numpy for saving
                num_nodes = g["num_nodes"]
                edge_index = g["edge_index"].astype(np.int64)
                edge_weight = g["edge_weight"].astype(np.float32)

                graphs_by_layer[L].append([num_nodes, edge_index, edge_weight])

                # Last-token vector from this layer
                last_vec = hs[-1, :].contiguous().detach().cpu().numpy().astype(np.float32)  # [hidden]
                last_token_by_layer[L].append(last_vec)

            # Metrics bookkeeping
            if gen_caption:
                all_preds.append(gen_caption)
                all_refs.append(references)

            # Logging
            if sample_idx > 0 and (sample_idx % log_every == 0):
                print(f"\n--- Sample {sample_idx} ---")
                for i, ref in enumerate(references[:2]):
                    print(f"  Ref {i+1}: {ref[:80]}")
                print(f"  Generated: {gen_caption[:80] if gen_caption else '<empty>'}")
                if verbose:
                    L0 = selected_layers[0]
                    num_nodes, edge_index, edge_weight = graphs_by_layer[L0][-1]
                    print(f"  [Layer {L0}] edges: {edge_index.shape[1]}, weights in [{edge_weight.min():.4f}, {edge_weight.max():.4f}]")
            
            del hidden_states_all
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error processing sample {sample_idx}: {str(e)}")
            continue

    N = len(all_preds)  # samples with captions
    print(f"\nTotal samples with captions: {N}, missing: {missing_captions}")

    # ---------- Compute & SAVE metrics (arrays) ----------
    meteor_arr = np.array([], dtype=np.float32)
    rougeL_arr = np.array([], dtype=np.float32)
    spice_arr  = np.array([], dtype=np.float32)

    if N > 0:
        print(f"\n{'='*60}\nCOMPUTING METRICS\n{'='*60}")
        preds_s, refs_s = sanitize_preds_refs(all_preds, all_refs)
        meteor_arr = np.array(meteor_scores(preds_s, refs_s), dtype=np.float32)
        rougeL_arr = np.array(rougeL_scores(preds_s, refs_s), dtype=np.float32)
        spice_arr  = np.array(spice_scores(preds_s, refs_s), dtype=np.float32)

        np.save(os.path.join(out_dir, "meteor.npy"), meteor_arr)
        np.save(os.path.join(out_dir, "rougeL.npy"), rougeL_arr)
        np.save(os.path.join(out_dir, "spice.npy"), spice_arr)

        print(f"Saved metrics: meteor.npy, rougeL.npy, spice.npy")
    else:
        print("No captions were generated; metrics not computed.")

    # ---------- SAVE per-layer artifacts ----------
    print(f"\n{'='*60}\nSAVING PER-LAYER ARTIFACTS\n{'='*60}")
    for L in selected_layers:
        # Graphs per layer
        graphs_path = os.path.join(out_dir, f"graphs_layer_{L}.pkl")
        with open(graphs_path, "wb") as f:
            pickle.dump(graphs_by_layer[L], f)

        # Last-token matrix per layer
        if len(last_token_by_layer[L]) > 0:
            last_mat = np.stack(last_token_by_layer[L], axis=0)  # [N, H]
        else:
            last_mat = np.zeros((0, extractor.hidden_dim), dtype=np.float32)
        np.save(os.path.join(out_dir, f"last_token_layer_{L}.npy"), last_mat)

    # ---------- Metadata ----------
    metadata = {
        "num_samples_total": len(data),
        "num_samples_with_captions": N,
        "missing_captions": missing_captions,
        "model": model_ckpt,
        "model_family": extractor.model_family,
        "num_layers": extractor.num_layers,
        "hidden_dim": extractor.hidden_dim,
        "selected_layers": selected_layers,
        "prompt_choice": prompt_choice,
        "sparse_level": float(sparse_level),
        "files": {
            "metrics": ["meteor.npy", "rougeL.npy", "spice.npy"],
            "graphs": [f"graphs_layer_{L}.pkl" for L in selected_layers],
            "last_tokens": [f"last_token_layer_{L}.npy" for L in selected_layers],
        }
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nOutput directory: {out_dir}")
    return graphs_by_layer, last_token_by_layer, meteor_arr, rougeL_arr, spice_arr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run probing caption coco dataset creation (single sample processing)")

    parser.add_argument("--model_ckpt", type=str, default="llava-hf/llava-1.5-7b-hf",
                        help="Model checkpoint name (HuggingFace repo ID or path)")
    parser.add_argument("--prompt_choice", type=int, default=0, choices=[0, 1, 2],
                        help="Prompt index (0, 1, or 2)")
    parser.add_argument("--num_samples", type=int, default=2500,
                        help="Number of samples from COCO val set")
    parser.add_argument("--sparse_level", type=float, default=0.9,
                        help="Quantile threshold for sparsifying correlation graph")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device string, e.g., 'cuda:0' or 'cpu'")
    parser.add_argument("--output_dir", type=str, default="probing_dataset",
                        help="Output directory for results")
    parser.add_argument("--verbose", action="store_true",
                        help="Print extra debugging information")
    parser.add_argument("--log_every", type=int, default=200,
                        help="Log progress every N samples")
    parser.add_argument("--layer_slices", type=int, default=4,
                        help="K slices -> K+1 evenly spaced layers (incl. first & last)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"COCO Dataset Creation")
    print(f"Model: {args.model_ckpt}")
    print(f"Prompt choice: {args.prompt_choice}")
    print(f"Num samples: {args.num_samples}")
    print(f"Sparse level: {args.sparse_level}")
    print(f"Device: {args.device}")
    print(f"{'='*60}\n")

    dataset = create_coco_dataset(
        num_samples=args.num_samples,
        model_ckpt=args.model_ckpt,
        output_dir=args.output_dir,
        sparse_level=args.sparse_level,
        prompt_choice=args.prompt_choice,
        verbose=args.verbose,
        device=args.device,
        log_every=args.log_every,
    )