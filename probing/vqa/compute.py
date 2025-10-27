from __future__ import annotations
import argparse
import pickle
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from sklearn.model_selection import train_test_split


# -----------------------------
# CLEVR label vocabularies
# -----------------------------
CLEVR_COLORS = ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow']
CLEVR_SHAPES = ['cube', 'sphere', 'cylinder']
CLEVR_RELATIONS = ['left', 'right', 'front', 'behind', 'above', 'below']
CLEVR_EXISTENCE = ['yes', 'no']              # normalized answers
CLEVR_COUNTING = [str(i) for i in range(0, 11)]

TASK_TO_CLASSES = {
    "color": CLEVR_COLORS,
    "shape": CLEVR_SHAPES,
    "relation": CLEVR_RELATIONS,
    "existence": CLEVR_EXISTENCE,
    "counting": CLEVR_COUNTING,
}


# -----------------------------
# Small helpers
# -----------------------------
def get_activation(name: str):
    name = (name or "relu").lower()
    if name == "relu":
        return nn.ReLU()
    if name == "elu":
        return nn.ELU()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.1)
    if name == "tanh":
        return nn.Tanh()
    if name == "none":
        return nn.Identity()
    raise ValueError(f"Unknown activation: {name}")


def label_to_index(label: str, classes: List[str]) -> int:
    """Map string label to class index; raises if not found."""
    lab = (label or "").strip().lower()
    # normalize 'grey' -> 'gray'
    if lab == "grey":
        lab = "gray"
    if lab not in classes:
        raise ValueError(f"Label '{label}' not in classes {classes}")
    return classes.index(lab)


# -----------------------------
# Model
# -----------------------------
class GCNPredictor(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        num_classes: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        fc_hidden_dim: int = 128,
        num_layers: int = 2,
        activation: str = 'relu',
        use_activation_final: bool = False,
        dropout: float = 0.0,
        edge_weighted: bool = True,
    ):
        """
        GCN Classifier with configurable layers and activation.
        Args:
            num_nodes: number of unique nodes (hidden units) in the graph
            num_classes: number of target classes
        """
        super().__init__()
        self.num_layers = num_layers
        self.use_activation_final = use_activation_final
        self.dropout = dropout
        self.edge_weighted = edge_weighted

        # Node embedding (indices 0..num_nodes-1)
        self.node_embedding = nn.Embedding(num_nodes, embedding_dim)

        # Activation
        self.activation = get_activation(activation)

        # GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(embedding_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Head: mean+max pooling -> FC -> logits
        self.fc1 = nn.Linear(hidden_dim * 2, fc_hidden_dim)
        self.fc2 = nn.Linear(fc_hidden_dim, num_classes)

        # (Optional) BatchNorm if you want it; currently unused
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])

    def forward(self, node_ids, edge_index, edge_weight=None, batch=None):
        x = self.node_embedding(node_ids)  # [N, emb]

        for i, conv in enumerate(self.convs):
            if self.edge_weighted and edge_weight is not None:
                x = conv(x, edge_index, edge_weight)
            else:
                x = conv(x, edge_index)

            # Optionally use BN: uncomment if you see training instability
            # x = self.bn_layers[i](x)

            if i < self.num_layers - 1 or self.use_activation_final:
                x = self.activation(x)

            if self.dropout > 0 and self.training and i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout)

        # Global mean + max pool over nodes per graph
        avg_pool = global_mean_pool(x, batch)  # [B, H]
        max_pool = global_max_pool(x, batch)   # [B, H]
        x = torch.cat([avg_pool, max_pool], dim=1)  # [B, 2H]

        x = self.fc1(x)
        if self.dropout > 0 and self.training:
            x = F.dropout(x, p=self.dropout)
        logits = self.fc2(x)  # [B, C]
        return logits


# -----------------------------
# Data loading
# -----------------------------
def load_graph_dataset(
    dataset_path: str,
    layer_name: str,
    task: str,
) -> Tuple[List[Data], int, List[str]]:
    """
    Load graphs from your saved pickle:
      each sample should contain:
        - sample[layer_name]['num_nodes']
        - sample[layer_name]['edge_index'] (shape [2, E])
        - sample[layer_name]['edge_weight'] (shape [E])
        - sample['reference_answer'] (string)
    Returns:
      data_list: list of PyG Data objects with x (node indices), edge_index, edge_attr, y (class index)
      num_nodes: number of nodes for embedding layer
      classes:  class vocabulary used
    """
    with open(dataset_path, "rb") as f:
        samples = pickle.load(f)
    assert len(samples) > 0, "Empty dataset."

    classes = TASK_TO_CLASSES[task]
    num_nodes = samples[0][layer_name]['num_nodes']

    data_list: List[Data] = []
    dropped = 0

    for s in samples:
        g = s[layer_name]
        ref = s.get("reference_answer", None)
        if ref is None:
            dropped += 1
            continue

        try:
            y_idx = label_to_index(ref, classes)
        except ValueError:
            dropped += 1
            continue

        edge_index = torch.tensor(g["edge_index"], dtype=torch.long)          # [2, E]
        edge_attr  = torch.tensor(np.abs(g["edge_weight"]), dtype=torch.float16)  # [E]

        data = Data(
            x=torch.arange(g["num_nodes"], dtype=torch.long),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor(y_idx, dtype=torch.long),
        )
        data_list.append(data)

    if dropped > 0:
        print(f"[load_graph_dataset] Dropped {dropped} samples due to missing/out-of-vocab labels.")

    return data_list, num_nodes, classes


def split_data(data_list: List[Data], seed: int = 42):
    """7:1:2 split for train/val/test."""
    n = len(data_list)
    idx = list(range(n))
    train_val_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=seed, shuffle=True)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.125, random_state=seed, shuffle=True)
    return ([data_list[i] for i in train_idx],
            [data_list[i] for i in val_idx],
            [data_list[i] for i in test_idx])


# -----------------------------
# Train / Eval
# -----------------------------
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)  # [B, C]
        loss = criterion(logits, batch.y)  # CE expects class indices
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += float(loss.item()) * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)  # [B, C]
        loss = criterion(logits, batch.y)
        total_loss += float(loss.item()) * batch.num_graphs

        pred = logits.argmax(dim=-1)  # [B]
        total_correct += int((pred == batch.y).sum().item())
        total_count += int(batch.num_graphs)

    avg_loss = total_loss / max(1, total_count)
    acc = total_correct / max(1, total_count)
    return avg_loss, acc


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Train GCN classifier on CLEVR graph dataset")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the saved pickle (e.g., complete_dataset.pkl)")
    parser.add_argument("--layer_name", type=str, default="graph_layer_0",
                        help="Which layer graph to use (graph_layer_0 | graph_layer_middle | graph_layer_last)")
    parser.add_argument("--task", type=str, default="color",
                        choices=list(TASK_TO_CLASSES.keys()),
                        help="Classification task to determine label space")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--fc_hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    # Repro
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading dataset...")
    data_list, num_nodes, classes = load_graph_dataset(
        dataset_path=args.dataset_path,
        layer_name=args.layer_name,
        task=args.task,
    )
    print(f"Loaded {len(data_list)} graphs | nodes per graph: {num_nodes} | num_classes: {len(classes)}")

    train_data, val_data, test_data = split_data(data_list, seed=args.seed)
    print(f"Split -> Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_data,  batch_size=args.batch_size, shuffle=False)

    # Model
    model = GCNPredictor(
        num_nodes=num_nodes,
        num_classes=len(classes),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        fc_hidden_dim=args.fc_hidden_dim,
        num_layers=args.num_layers,
        activation='relu',
        use_activation_final=False,
        dropout=args.dropout,
        edge_weighted=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Train
    best_state = None
    best_val_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # Test (best)
    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print("\n=== TEST RESULTS ===")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc : {test_acc*100:.2f}%")

if __name__ == "__main__":
    main()
