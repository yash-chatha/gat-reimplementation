"""
Training script for GAT on Cora / CiteSeer (transductive node classification).

Usage:
  python train.py --dataset Cora
  python train.py --dataset CiteSeer

Returns test accuracy after training with early stopping.
"""

import argparse
import copy
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from model import GAT

# --------------------------------------------------------------------------- #
# Hyperparameters (from Veličković et al., 2018)
# --------------------------------------------------------------------------- #
LR = 0.005
WEIGHT_DECAY = 5e-4
DROPOUT = 0.6
EPOCHS = 10_000
PATIENCE = 100  # early stopping patience on validation loss


def _default_planetoid_parent() -> Path:
    """Prefer mounted Google Drive in Colab; else ephemeral /tmp."""
    colab_repo = Path(
        "/content/drive/MyDrive/[Cornell] Spring Junior/CS 4782/gat-reimplementation"
    )
    if colab_repo.exists():
        return colab_repo / "gat_data"
    return Path("/tmp")


def train_epoch(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate_val_loss(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    return F.nll_loss(out[data.val_mask], data.y[data.val_mask]).item()


@torch.no_grad()
def test_accuracy(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    return (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()


def run(
    dataset_name: str,
    seed: int = 42,
    planetoid_parent: Optional[Path] = None,
) -> float:
    """Train GAT on `dataset_name`, return test accuracy."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    parent = planetoid_parent if planetoid_parent is not None else _default_planetoid_parent()
    parent.mkdir(parents=True, exist_ok=True)
    root = str(parent / dataset_name)

    dataset = Planetoid(
        root=root,
        name=dataset_name,
        transform=NormalizeFeatures(),
    )
    data = dataset[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    model = GAT(
        num_features=dataset.num_features,
        num_classes=dataset.num_classes,
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for _epoch in range(1, EPOCHS + 1):
        train_epoch(model, data, optimizer)
        val_loss = evaluate_val_loss(model, data)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    model.load_state_dict(best_state)
    return test_accuracy(model, data)


def main():
    parser = argparse.ArgumentParser(description="Train GAT on a Planetoid dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="Cora",
        choices=["Cora", "CiteSeer"],
        help="Dataset to use (default: Cora)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--planetoid-parent",
        type=Path,
        default=None,
        help=(
            "Directory under which Planetoid datasets are stored "
            "(default: Drive gat_data if mounted, else /tmp)."
        ),
    )
    args = parser.parse_args()

    test_acc = run(
        args.dataset,
        seed=args.seed,
        planetoid_parent=args.planetoid_parent,
    )
    print(f"[{args.dataset}] Test Accuracy: {test_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
