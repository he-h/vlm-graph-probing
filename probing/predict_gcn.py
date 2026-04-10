"""
probing/predict_gcn.py — Train and evaluate GCNPredictor on pre-extracted correlation graphs.

Inputs  (from probing/extract_graphs.py output dir):
  graphs_layer_{L}.pkl      list of [num_nodes, edge_index, edge_weight]
  text_graphs_layer_{L}.pkl same format, text-token subgraph only
  preds.json                list of model predictions (strings)
  refs.json                 list of ground-truth answers (strings)

Label:  binary — 1 if pred == ref (model answered correctly), 0 otherwise.
Task:   binary classification with BCEWithLogitsLoss.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import json
import pickle
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

from model import GCNPredictor


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def resolve_layer(data_dir: str, graph_type: str, requested: int) -> int:
    """
    Return the layer index to use.
    If requested == -1, pick the first layer found in the directory.
    """
    prefix = "graphs_layer_" if graph_type == "full" else "text_graphs_layer_"
    if requested != -1:
        return requested

    # scan for available layers
    candidates = sorted(
        int(p.stem.replace(prefix, ""))
        for p in Path(data_dir).glob(f"{prefix}*.pkl")
    )
    if not candidates:
        raise FileNotFoundError(
            f"No {prefix}*.pkl files found in {data_dir}"
        )
    chosen = candidates[0]
    print(f"[INFO] --layer not set; auto-selected layer {chosen} "
          f"(available: {candidates})")
    return chosen


def load_graphs_and_labels(data_dir: str, layer: int, graph_type: str):
    """
    Load graph data and binary correctness labels.

    Returns
    -------
    graphs : list of (num_nodes, edge_index_np, edge_weight_np)
    labels : list of int  (1 = correct, 0 = wrong)
    """
    prefix = "graphs_layer_" if graph_type == "full" else "text_graphs_layer_"
    pkl_path = os.path.join(data_dir, f"{prefix}{layer}.pkl")
    preds_path = os.path.join(data_dir, "preds.json")
    refs_path  = os.path.join(data_dir, "refs.json")

    for p in (pkl_path, preds_path, refs_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")

    with open(pkl_path, "rb") as f:
        graphs = pickle.load(f)      # list of [num_nodes, edge_index, edge_weight]

    with open(preds_path) as f:
        preds = json.load(f)
    with open(refs_path) as f:
        refs = json.load(f)

    n = min(len(graphs), len(preds), len(refs))
    if len(graphs) != len(preds) or len(preds) != len(refs):
        print(f"[WARN] Length mismatch: graphs={len(graphs)}, "
              f"preds={len(preds)}, refs={len(refs)}. Using first {n}.")

    labels = [
        1 if str(preds[i]).lower().strip() == str(refs[i]).lower().strip() else 0
        for i in range(n)
    ]
    graphs = graphs[:n]

    pos = sum(labels)
    print(f"[INFO] Loaded {n} samples | correct={pos} ({100*pos/n:.1f}%) "
          f"| incorrect={n-pos} ({100*(n-pos)/n:.1f}%)")
    return graphs, labels


def build_pyg_dataset(graphs, labels) -> list:
    """Convert raw graph tuples to torch_geometric Data objects."""
    dataset = []
    for (num_nodes, edge_index_np, edge_weight_np), label in zip(graphs, labels):
        num_nodes = int(num_nodes)
        node_ids  = torch.arange(num_nodes, dtype=torch.long)
        ei        = torch.from_numpy(edge_index_np).long()
        ew        = torch.from_numpy(edge_weight_np).float()
        y         = torch.tensor([label], dtype=torch.float)
        dataset.append(Data(x=node_ids, edge_index=ei, edge_attr=ew, y=y,
                            num_nodes=num_nodes))
    return dataset


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def run_epoch(model, loader, criterion, optimizer, device, train: bool,
              grad_clip: float = 1.0):
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0
    nan_batches = 0

    for batch in loader:
        batch = batch.to(device)

        # node_ids stored in batch.x (long); edge weights in batch.edge_attr
        out = model(
            node_ids=batch.x,
            edge_index=batch.edge_index,
            edge_weight=batch.edge_attr,
            batch=batch.batch,
        ).squeeze(-1)   # [B]

        # guard against NaN outputs (can happen with degenerate graphs)
        out = torch.nan_to_num(out, nan=0.0, posinf=10.0, neginf=-10.0)

        loss = criterion(out, batch.y.squeeze(-1))

        if train:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        loss_val = loss.item()
        if not np.isfinite(loss_val):
            nan_batches += 1
            loss_val = 0.0

        total_loss += loss_val * batch.num_graphs
        preds = (torch.sigmoid(out) >= 0.5).long()
        correct += (preds == batch.y.squeeze(-1).long()).sum().item()
        total   += batch.num_graphs

    if nan_batches > 0:
        print(f"  [WARN] {nan_batches} batch(es) had non-finite loss (skipped in total).")

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train GCNPredictor on pre-extracted VLM correlation graphs"
    )

    # --- data ---
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Output directory from probing/extract_graphs.py")
    parser.add_argument("--layer", type=int, default=-1,
                        help="Layer index to use (-1 = auto-select first found)")
    parser.add_argument("--graph_type", type=str, default="full",
                        choices=["full", "text"],
                        help="Graph variant: 'full' (all tokens) or 'text' (text tokens only)")

    # --- split / seed ---
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Fraction of data for test set")

    # --- training ---
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Max gradient norm for clipping (0 = disabled)")

    # --- model ---
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--hidden_dim",    type=int, default=128)
    parser.add_argument("--fc_hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers",    type=int, default=2)
    parser.add_argument("--dropout",       type=float, default=0.0)
    parser.add_argument("--activation",    type=str, default="relu",
                        choices=["relu", "elu", "leaky_relu", "tanh", "none"])
    parser.add_argument("--edge_weighted", action="store_true", default=True,
                        help="Pass edge weights to GCN convolutions")

    # --- output ---
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default="results/gcn_predictor")
    parser.add_argument("--save_model", action="store_true",
                        help="Save best model checkpoint")
    parser.add_argument("--log_every", type=int, default=10,
                        help="Print metrics every N epochs")

    args = parser.parse_args()

    # ------------------------------------------------------------------ setup
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if str(device) != args.device:
        print(f"[WARN] CUDA unavailable; falling back to CPU")

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------ data
    layer = resolve_layer(args.data_dir, args.graph_type, args.layer)

    print(f"\n{'='*60}")
    print(f"GCN Predictor Training")
    print(f"  data_dir   : {args.data_dir}")
    print(f"  layer      : {layer}  graph_type: {args.graph_type}")
    print(f"  seed       : {args.seed}  device: {device}")
    print(f"{'='*60}\n")

    graphs, labels = load_graphs_and_labels(args.data_dir, layer, args.graph_type)

    # num_nodes is the same for all graphs (= hidden_dim of the VLM)
    num_nodes = int(graphs[0][0])
    print(f"[INFO] num_nodes (hidden_dim) = {num_nodes}")

    dataset = build_pyg_dataset(graphs, labels)

    # stratified split to maintain class balance
    indices = list(range(len(dataset)))
    try:
        train_idx, test_idx = train_test_split(
            indices, test_size=args.test_size, random_state=args.seed,
            stratify=labels
        )
    except ValueError:
        # fallback if stratification fails (e.g. only one class)
        print("[WARN] Stratified split failed; using random split.")
        train_idx, test_idx = train_test_split(
            indices, test_size=args.test_size, random_state=args.seed
        )

    train_data = [dataset[i] for i in train_idx]
    test_data  = [dataset[i] for i in test_idx]

    print(f"[INFO] Train: {len(train_data)}  Test: {len(test_data)}\n")

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_data,  batch_size=args.batch_size, shuffle=False)

    # ------------------------------------------------------------------ model
    model = GCNPredictor(
        num_nodes=num_nodes,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        fc_hidden_dim=args.fc_hidden_dim,
        num_layers=args.num_layers,
        activation=args.activation,
        dropout=args.dropout,
        edge_weighted=args.edge_weighted,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] GCNPredictor  params={n_params:,}")
    print(f"       embedding_dim={args.embedding_dim}  hidden_dim={args.hidden_dim}  "
          f"fc_hidden_dim={args.fc_hidden_dim}  num_layers={args.num_layers}\n")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # ----------------------------------------------------------------- train
    history = []
    best_test_acc  = 0.0
    best_epoch     = 0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device,
                                    train=True,  grad_clip=args.grad_clip)
        te_loss, te_acc = run_epoch(model, test_loader,  criterion, optimizer, device,
                                    train=False, grad_clip=args.grad_clip)

        history.append({
            "epoch": epoch,
            "train_loss": round(tr_loss, 6), "train_acc": round(tr_acc, 6),
            "test_loss":  round(te_loss, 6), "test_acc":  round(te_acc, 6),
        })

        if te_acc > best_test_acc:
            best_test_acc = te_acc
            best_epoch    = epoch
            if args.save_model:
                ckpt_path = os.path.join(
                    args.output_dir,
                    f"best_gcn_layer{layer}_{args.graph_type}.pt"
                )
                torch.save(model.state_dict(), ckpt_path)

        if epoch % args.log_every == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{args.epochs} | "
                  f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
                  f"test  loss={te_loss:.4f} acc={te_acc:.4f}")

    # ----------------------------------------------------------------- save
    results = {
        "config": vars(args),
        "layer": layer,
        "num_nodes": num_nodes,
        "num_train": len(train_data),
        "num_test":  len(test_data),
        "best_test_acc": round(best_test_acc, 6),
        "best_epoch": best_epoch,
        "history": history,
    }
    out_path = os.path.join(
        args.output_dir, f"results_layer{layer}_{args.graph_type}.json"
    )
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete.")
    print(f"  Best test accuracy : {best_test_acc:.4f}  (epoch {best_epoch})")
    print(f"  Results saved to   : {out_path}")
    if args.save_model:
        print(f"  Best checkpoint    : {ckpt_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
