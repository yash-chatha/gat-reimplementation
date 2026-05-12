"""
Training script for GAT on Cora / CiteSeer (transductive node classification).

Usage:
  python train.py --dataset Cora
  python train.py --dataset CiteSeer
  python train.py --dataset PubMed

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

from model import GAT, TinyGAT, PubMedGAT
from model_repulsive import RepulsiveGAT

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


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    val_loss = F.nll_loss(out[data.val_mask], data.y[data.val_mask]).item()
    pred = out.argmax(dim=1)
    val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
    test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
    return val_loss, val_acc, test_acc


def load_data(
    dataset_name: str,
    device: torch.device,
    planetoid_parent: Optional[Path] = None,
):
    parent = (
        planetoid_parent if planetoid_parent is not None else _default_planetoid_parent()
    )
    parent.mkdir(parents=True, exist_ok=True)
    root = str(parent / dataset_name)
    dataset = Planetoid(
        root=root,
        name=dataset_name,
        transform=NormalizeFeatures(),
    )
    data = dataset[0].to(device)
    return dataset, data


def build_model(model_name: str, dataset, dropout: float, device: torch.device):
    if model_name == "gat":
        model = GAT(
            num_features=dataset.num_features,
            num_classes=dataset.num_classes,
            dropout=dropout,
        )
    elif model_name == "tinygat":
        model = TinyGAT(
            num_features=dataset.num_features,
            num_classes=dataset.num_classes,
            dropout=dropout,
        )
    elif model_name == "pubmedgat":
        model = PubMedGAT(
            num_features=dataset.num_features,
            num_classes=dataset.num_classes,
            dropout=dropout,
        )
    elif model_name == "repulsivegat":
        model = RepulsiveGAT(
            num_features=dataset.num_features,
            num_classes=dataset.num_classes,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")
    return model.to(device)


def run(
    dataset_name: str,
    seed: int = 42,
    model_name: str = "gat",
    train_mask_override: torch.Tensor | None = None,
    return_model_and_data: bool = False,
    planetoid_parent: Optional[Path] = None,
    ablate_features: bool = False,
    ablate_topology: bool = False,
) -> float | tuple[float, torch.nn.Module, object]:
    """Train model on `dataset_name`, return best test accuracy."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, data = load_data(
        dataset_name=dataset_name,
        device=device,
        planetoid_parent=planetoid_parent,
    )

    if ablate_features:
        # Replace sparse features with random normal features to ablate sparsity
        data.x = torch.randn_like(data.x)

    if ablate_topology:
        # Remove all edges, reducing GNN to MLP
        data.edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

    if train_mask_override is not None:
        data.train_mask = train_mask_override.to(device)

    if dataset_name == "PubMed" and model_name == "gat":
        model_name = "pubmedgat"

    model = build_model(model_name=model_name, dataset=dataset, dropout=DROPOUT, device=device)

    # Hyperparameters based on dataset
    lr = 0.01 if dataset_name == "PubMed" else LR
    weight_decay = 0.001 if dataset_name == "PubMed" else WEIGHT_DECAY
    epochs = 200 if dataset_name == "PubMed" else EPOCHS

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for _epoch in range(1, epochs + 1):
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
    best_test_acc = test_accuracy(model, data)

    if return_model_and_data:
        return best_test_acc, model, data
    return best_test_acc


def main():
    parser = argparse.ArgumentParser(description="Train GAT on a Planetoid dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="Cora",
        choices=["Cora", "CiteSeer", "PubMed"],
        help="Dataset to use (default: Cora)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--model",
        type=str,
        default="gat",
        choices=["gat", "tinygat", "repulsivegat"],
        help="Model architecture to use",
    )
    parser.add_argument(
        "--planetoid-parent",
        type=Path,
        default=None,
        help=(
            "Directory under which Planetoid datasets are stored "
            "(default: Drive gat_data if mounted, else /tmp)."
        ),
    )
    parser.add_argument(
        "--ablate-features",
        action="store_true",
        help="Ablate features by replacing with random noise",
    )
    parser.add_argument(
        "--ablate-topology",
        action="store_true",
        help="Ablate topology by removing all edges",
    )
    args = parser.parse_args()

    test_acc = run(
        args.dataset,
        seed=args.seed,
        model_name=args.model,
        planetoid_parent=args.planetoid_parent,
        ablate_features=args.ablate_features,
        ablate_topology=args.ablate_topology,
    )
    print(f"[{args.dataset} | {args.model}] Test Accuracy: {test_acc * 100:.2f}%")

if __name__ == "__main__":
    main()
