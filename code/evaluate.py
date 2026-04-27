"""
Evaluation script: runs GAT training 100 times and reports mean ± std test accuracy.
Results are saved to results/<dataset>_results.csv.

Usage:
  python evaluate.py --dataset Cora
  python evaluate.py --dataset CiteSeer
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np

from train import run

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
NUM_RUNS = 100


def evaluate(dataset_name: str):
    RESULTS_DIR.mkdir(exist_ok=True)
    csv_path = RESULTS_DIR / f"{dataset_name}_results.csv"

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
        choices=["Cora", "CiteSeer", "PubMed"],
        help="Dataset to evaluate (default: Cora)",
    )
    args = parser.parse_args()
    evaluate(args.dataset)


if __name__ == "__main__":
    main()
