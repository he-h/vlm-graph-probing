#!/usr/bin/env python3
import os
import json
import math
import pickle
import numpy as np
from glob import glob

def prune_edges_top_frac(edge_index: np.ndarray,
                         edge_weight: np.ndarray,
                         keep_frac: float = 0.10):
    """
    Keep the top `keep_frac` fraction of edges by absolute weight.

    Args:
        edge_index: shape [2, E], dtype=int64
        edge_weight: shape [E], dtype=float32/float64
        keep_frac: fraction to keep (0.10 = top 10%)

    Returns:
        (new_edge_index, new_edge_weight) pruned arrays with same dtypes
    """
    E = edge_weight.shape[0]
    if E == 0 or keep_frac >= 1.0:
        return edge_index, edge_weight
    k = max(1, int(math.ceil(keep_frac * E)))

    abs_w = np.abs(edge_weight)
    # Get indices of the top-k (unsorted), then sort them descending by abs weight
    topk_unsorted = np.argpartition(abs_w, -k)[-k:]
    order = topk_unsorted[np.argsort(abs_w[topk_unsorted])[::-1]]

    pruned_edge_index = edge_index[:, order]
    pruned_edge_weight = edge_weight[order]
    return pruned_edge_index, pruned_edge_weight

def process_pickle_file(in_path: str, out_path: str, keep_frac: float = 0.10):
    """
    Load a dataset pickle (list of sample dicts), prune edges per graph, save to out_path.
    """
    with open(in_path, "rb") as f:
        samples = pickle.load(f)

    for s in samples:
        graphs = s.get("graphs", {})
        for layer_name, g in graphs.items():
            ei = g.get("edge_index", None)
            ew = g.get("edge_weight", None)
            if ei is None or ew is None:
                continue

            # Ensure correct dtypes
            ei = ei.astype(np.int64, copy=False)
            ew = ew.astype(np.float32, copy=False)

            # Prune to top 10%
            new_ei, new_ew = prune_edges_top_frac(ei, ew, keep_frac=keep_frac)

            # Write back
            g["edge_index"] = new_ei
            g["edge_weight"] = new_ew
            # num_nodes stays the same
            # other fields unchanged

    with open(out_path, "wb") as f:
        pickle.dump(samples, f)
    return len(samples)

def main():
    # INPUT: your dense (or less sparse) dataset directory
    in_dir  = "./llava_graph_probing_dataset_sparsity_0"
    # OUTPUT: new directory with top-10% edges retained
    out_dir = "./llava_graph_probing_dataset_sparsity_90"
    keep_frac = 0.10  # keep top 10% by |weight|

    os.makedirs(out_dir, exist_ok=True)

    # Copy & update metadata.json if present
    meta_in = os.path.join(in_dir, "metadata.json")
    meta_out = os.path.join(out_dir, "metadata.json")
    if os.path.exists(meta_in):
        with open(meta_in, "r") as f:
            meta = json.load(f)
        meta["sparsity"] = 0.9
        note = meta.get("note", "")
        add = "Edges pruned to keep top 10% by absolute weight."
        meta["note"] = (note + " " + add).strip() if note else add
        with open(meta_out, "w") as f:
            json.dump(meta, f, indent=2)

    # Find all .pkl files (batches + complete)
    pkl_files = sorted(glob(os.path.join(in_dir, "*.pkl")))
    if not pkl_files:
        print(f"No .pkl files found in {in_dir}")
        return

    total = 0
    for p in pkl_files:
        fname = os.path.basename(p)
        out_p = os.path.join(out_dir, fname)
        n = process_pickle_file(p, out_p, keep_frac=keep_frac)
        total += n
        print(f"Processed {fname}: {n} samples")

    print("\nDone.")
    print(f"Input dir : {in_dir}")
    print(f"Output dir: {out_dir}")
    print(f"Total samples processed across files: {total}")
    print("All other fields kept the same; only edges were pruned and metadata.sparsity set to 0.9.")

if __name__ == "__main__":
    main()
