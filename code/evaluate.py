"""
Evaluation script: runs GAT training 100 times and reports mean ± std test accuracy.
Results are saved to results/<dataset>_results.csv.

Usage:
  python evaluate.py --dataset Cora
  python evaluate.py --dataset CiteSeer
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from train import run

NUM_RUNS = 100


def _default_results_dir() -> Path:
    """Prefer mounted Google Drive in Colab; fallback to local repo."""
    colab_repo = Path(
        "/content/drive/MyDrive/[Cornell] Spring Junior/CS 4782/gat-reimplementation"
    )
    if colab_repo.exists():
        return colab_repo / "results"
    return Path(__file__).resolve().parent.parent / "results"


def evaluate(dataset_name: str, results_dir: Path):
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"{dataset_name}_results.csv"

    accuracies = []

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "test_accuracy"])

        for i in range(1, NUM_RUNS + 1):
            acc = run(dataset_name, seed=i)
            accuracies.append(acc)
            writer.writerow([i, f"{acc:.6f}"])
            print(f"Run {i:3d}/{NUM_RUNS}  test_acc={acc * 100:.2f}%")

    mean_acc = np.mean(accuracies) * 100
    std_acc = np.std(accuracies) * 100

    print(f"\n{'='*45}")
    print(f"Dataset : {dataset_name}")
    print(f"Runs    : {NUM_RUNS}")
    print(f"Mean    : {mean_acc:.2f}%")
    print(f"Std     : {std_acc:.2f}%")
    print(f"Results saved to: {csv_path}")
    print(f"{'='*45}")

    return mean_acc, std_acc


def main():
    parser = argparse.ArgumentParser(
        description="Run GAT 100 times and report mean/std test accuracy."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Cora",
        choices=["Cora", "CiteSeer"],
        help="Dataset to evaluate (default: Cora)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=_default_results_dir(),
        help=(
            "Directory to save CSV results. Defaults to mounted Google Drive path "
            "in Colab if available, otherwise repo/results."
        ),
    )
    args = parser.parse_args()
    evaluate(args.dataset, args.results_dir)


if __name__ == "__main__":
    main()
