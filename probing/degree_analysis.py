"""
Layer-wise degree and activation analysis.

Processes VQA samples across ALL layers of a VLM and computes:
1. Average node degree for each neuron (from correlation graphs)
2. Average absolute activation of the last token

Output shapes: [num_layers, hidden_dim]
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import numpy as np
from tqdm import tqdm
import json
import os
import argparse

from utils import model_ckpt2name
from model import NeuronGraphExtractor as GraphExtractor
from model import build_corr_graph
from dataset import prepare_vlm_data


def compute_node_degrees(corr_graph):
    """
    Compute node degrees from correlation graph in COO format.

    Args:
        corr_graph: dict with keys:
            - "num_nodes": int
            - "edge_index": np.ndarray [2, E]
            - "edge_weight": np.ndarray [E]

    Returns:
        degrees: np.ndarray [num_nodes] - degree of each node
    """
    num_nodes = corr_graph["num_nodes"]
    edge_index = corr_graph["edge_index"]  # [2, E]

    degrees = np.zeros(num_nodes, dtype=np.float64)
    source_nodes = edge_index[0, :]
    np.add.at(degrees, source_nodes, 1)
    target_nodes = edge_index[1, :]
    np.add.at(degrees, target_nodes, 1)

    return degrees


def create_layer_analysis(
    dataset="clevr",
    num_samples=1000,
    model_ckpt="llava-hf/llava-1.5-7b-hf",
    output_root="data",
    verbose=False,
    device="cuda:0",
    category="color",
    sparse_level=0.9,
    log_every=200,
    data_root=None,
    tdiuc_root=None,
    coco_val_root=None,
):
    """
    Process VQA samples and compute per-layer neuron statistics.

    For each layer computes:
    1. Average node degree across all samples  [num_layers, hidden_dim]
    2. Average |last-token activation|          [num_layers, hidden_dim]

    Saves:
      - avg_node_degree_all_layers.npy
      - avg_last_token_activation_all_layers.npy
      - metadata.json
    """
    model_prefix = model_ckpt2name(model_ckpt)
    output_dir = os.path.join(
        output_root,
        f"{model_prefix}_{dataset}_{category}_sparsity_{int(sparse_level * 100)}_layer_analysis",
    )
    os.makedirs(output_dir, exist_ok=True)

    # ---- Load data ----
    print("=" * 60)
    print(f"Preparing {dataset} samples, category={category}")
    data = prepare_vlm_data(
        dataset=dataset,
        num_samples=num_samples,
        category=category,
        balance=True,
        data_root=data_root,
        tdiuc_root=tdiuc_root,
        coco_val_root=coco_val_root,
    )
    if not data:
        print("ERROR: No samples loaded!")
        return

    # ---- Initialize model ----
    print("=" * 60)
    print(f"Initializing {model_ckpt} model...")
    extractor = GraphExtractor(model_ckpt=model_ckpt, device=device)

    num_layers = extractor.num_layers
    hidden_dim = extractor.hidden_dim

    # Initialize accumulators: [num_layers, hidden_dim]
    degree_accumulator = np.zeros((num_layers, hidden_dim), dtype=np.float64)
    activation_accumulator = np.zeros((num_layers, hidden_dim), dtype=np.float64)

    correct, total = 0, 0
    samples_processed = 0

    print(f"\nProcessing {len(data)} samples across ALL {num_layers} layers...")
    print(f"Output shapes will be: [{num_layers}, {hidden_dim}]")
    print("=" * 60)

    # ---- Main loop ----
    for sample_idx, (image, prompt, answer) in enumerate(
        tqdm(data, desc="Processing samples")
    ):
        try:
            hidden_states_all, gen, [image_token_start, text_token_start] = (
                extractor.process_single(image, prompt)
            )

            for layer_idx in range(num_layers):
                if layer_idx >= len(hidden_states_all):
                    continue

                hs = hidden_states_all[layer_idx][0]  # [seq, hidden]

                # 1. Correlation graph → node degrees
                corr_graph = build_corr_graph(hs, sparse_level=sparse_level)
                degrees = compute_node_degrees(corr_graph)
                if len(degrees) != hidden_dim:
                    print(
                        f"Warning: degree shape mismatch at layer {layer_idx}: "
                        f"{len(degrees)} vs {hidden_dim}"
                    )
                    continue
                degree_accumulator[layer_idx] += degrees

                # 2. Last token absolute activation
                last_token = hs[-1, :].float().detach().cpu().numpy()
                activation_accumulator[layer_idx] += np.abs(last_token)

            # Track accuracy
            pred = gen.lower().strip() if isinstance(gen, str) else str(gen)
            ref = answer.lower().strip() if isinstance(answer, str) else str(answer)
            correct += int(pred == ref)
            total += 1
            samples_processed += 1

            if sample_idx > 0 and sample_idx % log_every == 0:
                print(f"\n--- Sample {sample_idx} ---")
                print(f"Q: {prompt}")
                print(f"Pred: {gen} | Ref: {answer}")
                print(f"Current accuracy: {correct}/{total} = {correct/total*100:.2f}%")

        except Exception as e:
            print(f"Error processing sample {sample_idx}: {str(e)}")
            if verbose:
                import traceback
                traceback.print_exc()
            continue

        del hidden_states_all
        torch.cuda.empty_cache()

    # ---- Compute averages ----
    if samples_processed == 0:
        print("ERROR: No samples processed successfully!")
        return

    avg_degree = degree_accumulator / samples_processed
    avg_activation = activation_accumulator / samples_processed

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print("FINAL STATISTICS")
    print("=" * 60)
    print(f"Samples processed: {samples_processed}")
    acc = correct / total if total > 0 else 0.0
    print(f"Accuracy: {correct}/{total} = {acc * 100:.2f}%")
    print(f"Average degree shape: {avg_degree.shape}")
    print(f"Average activation shape: {avg_activation.shape}")
    print(f"Degree range: [{avg_degree.min():.4f}, {avg_degree.max():.4f}]")
    print(f"Activation range: [{avg_activation.min():.4f}, {avg_activation.max():.4f}]")

    # ---- Save ----
    np.save(os.path.join(output_dir, "avg_node_degree_all_layers.npy"), avg_degree)
    np.save(
        os.path.join(output_dir, "avg_last_token_activation_all_layers.npy"),
        avg_activation,
    )

    metadata = {
        "num_samples": samples_processed,
        "model": model_ckpt,
        "model_family": extractor.model_family,
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "sparse_level": sparse_level,
        "dataset": dataset,
        "category": category,
        "accuracy": acc,
        "output_shapes": {
            "avg_node_degree": list(avg_degree.shape),
            "avg_activation": list(avg_activation.shape),
        },
        "statistics": {
            "degree_min": float(avg_degree.min()),
            "degree_max": float(avg_degree.max()),
            "degree_mean": float(avg_degree.mean()),
            "degree_std": float(avg_degree.std()),
            "activation_min": float(avg_activation.min()),
            "activation_max": float(avg_activation.max()),
            "activation_mean": float(avg_activation.mean()),
            "activation_std": float(avg_activation.std()),
        },
    }

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nData saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run layer-wise degree and activation analysis"
    )
    parser.add_argument("--dataset", type=str, default="clevr")
    parser.add_argument("--model_ckpt", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--num_samples", type=int, default=2500)
    parser.add_argument("--sparse_level", type=float, default=0.9)
    parser.add_argument("--category", type=str, default="color")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--output_root",
        type=str,
        default="data",
        help="Directory where generated artifacts are written.",
    )
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--tdiuc_root", type=str, default=None)
    parser.add_argument("--coco_val_root", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log_every", type=int, default=200)
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"Layer-Wise Analysis (ALL layers)")
    print(f"Model: {args.model_ckpt}")
    print(f"Dataset: {args.dataset}, Category: {args.category}")
    print(f"{'=' * 60}\n")

    create_layer_analysis(
        dataset=args.dataset,
        num_samples=args.num_samples,
        model_ckpt=args.model_ckpt,
        output_root=args.output_root,
        sparse_level=args.sparse_level,
        category=args.category,
        verbose=args.verbose,
        device=args.device,
        log_every=args.log_every,
        data_root=args.data_root,
        tdiuc_root=args.tdiuc_root,
        coco_val_root=args.coco_val_root,
    )
