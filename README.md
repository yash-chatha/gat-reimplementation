# Graph Attention Networks — Re-implementation

**CS 4782: Deep Learning | Cornell University**
Authors: Yash Chatha (yc2727), Jerry Ji (rj378), Leon Huang (lyh7)

Paper: [Graph Attention Networks, Veličković et al., ICLR 2018](https://arxiv.org/abs/1710.10903)

---

## Introduction

[TODO] Brief overview of Graph Attention Networks and why they improve on prior graph neural network methods (GCN, GraphSAGE) by using learned attention coefficients over neighbors.

## Chosen Result

[TODO] We target Table 2 from the paper: **83.0 ± 0.7%** accuracy on Cora and **72.5 ± 0.7%** accuracy on CiteSeer (transductive classification, averaged over 100 runs).

## GitHub Contents

```
gat-reimplementation/
├── code/
│   ├── model.py      # 2-layer GAT model (PyTorch Geometric)
│   ├── train.py      # Training loop with early stopping
│   └── evaluate.py   # 100-run evaluation, saves results to CSV
├── data/
│   └── README.md     # Dataset info (auto-downloaded via PyG)
├── results/          # CSV outputs from evaluate.py
├── poster/           # Final poster PDF
├── report/           # Final report PDF
└── README.md
```

## Re-implementation Details

[TODO] Describe architecture choices: 2-layer GAT, 8 heads × 8 features in layer 1, 1 head outputting num_classes in layer 2, ELU activation, dropout=0.6, Adam optimizer (lr=0.005, weight_decay=5e-4), early stopping patience=100.

## Reproduction Steps

**Install dependencies:**
```bash
pip install torch torch_geometric
```

**Run a single training trial:**
```bash
python code/train.py --dataset Cora
python code/train.py --dataset CiteSeer
```

**Run 100-trial evaluation (saves CSV to results/):**
```bash
python code/evaluate.py --dataset Cora
python code/evaluate.py --dataset CiteSeer
```

**Run low-label benchmark (few-day friendly defaults):**
```bash
python code/low_label_benchmark.py --dataset Cora --ratios 0.1 0.2 0.5 1.0 --runs 10
python code/low_label_benchmark.py --dataset CiteSeer --ratios 0.1 0.2 0.5 1.0 --runs 10
```

**Run TinyGAT distillation benchmark:**
```bash
python code/distill_tinygat.py --dataset Cora --runs 5
python code/distill_tinygat.py --dataset CiteSeer --runs 5
```

**Export failure cases with neighborhood + attention context:**
```bash
python code/failure_case_explorer.py --dataset Cora --seed 1 --max-cases 25 --top-k-edges 5
```

## Results / Insights

[TODO] Fill in after running evaluate.py. Report mean ± std accuracy for Cora and CiteSeer. Note any deviations from paper results and potential causes (random seed variance, implementation differences, etc.).

| Dataset  | Paper Result | Our Result |
|----------|-------------|------------|
| Cora     | 83.0 ± 0.7% | [TODO]     |
| CiteSeer | 72.5 ± 0.7% | [TODO]     |

### Additional Outputs (new scripts)

- `results/<dataset>_low_label_detail.csv`:
  per-run test accuracy for each label ratio.
- `results/<dataset>_low_label_summary.csv`:
  mean/std test accuracy for each label ratio.
- `results/<dataset>_tinygat_distill.csv`:
  teacher accuracy, TinyGAT supervised accuracy, TinyGAT distilled accuracy, and distillation gain per seed.
- `results/<dataset>_seed<seed>_failure_cases.csv` and `.json`:
  misclassified test nodes with confidence, neighborhood class histogram, and top incoming attention edges.

## Conclusion

[TODO] Summarize whether we successfully reproduced the paper's results, and any insights gained about attention mechanisms in graph neural networks.

## References

- Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). [Graph Attention Networks](https://arxiv.org/abs/1710.10903). *ICLR 2018*.
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)

## Acknowledgements

[TODO] Acknowledge CS 4782 course staff, any external code references, and compute resources used.
