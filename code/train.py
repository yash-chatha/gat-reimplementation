"""
Training script for GAT on Cora / CiteSeer (transductive node classification).

Usage:
  python train.py --dataset Cora
  python train.py --dataset CiteSeer

Returns test accuracy after training with early stopping.
"""

import argparse
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
PATIENCE = 100        # early stopping patience on validation loss


def train_epoch(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    out = model(data.x, data.edge_index)

    val_loss = F.nll_loss(out[data.val_mask], data.y[data.val_mask]).item()

    pred = out.argmax(dim=1)
    val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
    test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()

    return val_loss, val_acc, test_acc


def run(dataset_name: str, seed: int = 42) -> float:
    """Train GAT on `dataset_name`, return test accuracy."""
    torch.manual_seed(seed)

    # Load dataset
    dataset = Planetoid(
        root=f"/tmp/{dataset_name}",
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
    best_test_acc = 0.0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        train_epoch(model, data, optimizer)
        val_loss, val_acc, test_acc = evaluate(model, data)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_test_acc = test_acc
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    return best_test_acc


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
    args = parser.parse_args()

    test_acc = run(args.dataset, seed=args.seed)
    print(f"[{args.dataset}] Test Accuracy: {test_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
