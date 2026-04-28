# Graph Attention Networks — Re-implementation

**CS 4782: Deep Learning | Cornell University**
Authors: Yash Chatha (yc2727), Jerry Ji (rj378), Leon Huang (lyh7)

Paper: [Graph Attention Networks, Veličković et al., ICLR 2018](https://arxiv.org/abs/1710.10903)

---

## Introduction

This repository re-implements Graph Attention Networks (GATs), which learn node representations by attending over graph neighborhoods — assigning learned importance weights to each neighbor rather than treating them equally. GATs improve on prior graph convolution methods (GCN, GraphSAGE) by enabling adaptive, node-specific aggregation without requiring knowledge of the full graph structure upfront.

## Chosen Result

We reproduce **Table 2** from the paper: transductive node classification accuracy on Cora and CiteSeer, averaged over 100 runs. The paper reports **83.0 ± 0.7%** on Cora and **72.5 ± 0.7%** on CiteSeer using a 2-layer GAT with 8-head attention.

## GitHub Contents

```
gat-reimplementation/
├── code/
│   ├── model.py      # GAT + TinyGAT model definitions (PyTorch Geometric)
│   ├── train.py      # Training loop with early stopping
│   └── evaluate.py   # 100-run evaluation, saves results to CSV
├── data/
│   └── README.md     # Dataset info (auto-downloaded via PyG)
├── notebooks/
│   └── GAT_Colab.ipynb  # Full Colab notebook: baseline + 3 extensions
├── results/          # CSVs, JSONs, and figures from all experiments
├── poster/           # Final poster PDF
├── report/           # Final report PDF
└── README.md
```

## Re-implementation Details

We implement a 2-layer GAT using PyTorch Geometric's `GATConv`, matching the paper's transductive setup exactly:

- **Layer 1:** 8 attention heads × 8 features = 64-dim output, ELU activation, dropout 0.6
- **Layer 2:** 1 attention head → num_classes, log-softmax output, dropout 0.6
- **Optimizer:** Adam, lr=0.005, weight_decay=5e-4 (L2 λ=0.0005)
- **Early stopping:** patience=100 on validation loss
- **Datasets:** Cora and CiteSeer via PyG Planetoid loader with NormalizeFeatures; standard public splits (20 nodes/class train, 500 val, 1000 test)

We also implement three extensions (in `notebooks/GAT_Colab.ipynb`):
- **Low-Label Benchmark:** GAT accuracy vs. fraction of training labels (0.1–1.0)
- **TinyGAT Distillation:** knowledge distillation from full GAT to a compressed student (2 heads × 4 features)
- **Failure Case Explorer:** misclassified test nodes with confidence, neighbor class histogram, and top attention edges

### Why these extensions (motivation and goal)

The paper's main result asks whether GAT is accurate on standard transductive benchmarks under the canonical split and model setup. We keep that as the primary target in this repo.

The extensions are **post-reproduction stress tests** designed to answer practical questions that the original paper does not directly focus on:

- **Low-label benchmark (sample efficiency):** if labeled nodes are scarce, how quickly does performance and stability degrade?
- **TinyGAT distillation (efficiency):** can a much smaller student recover teacher performance, enabling cheaper/faster deployment?
- **Failure case explorer (diagnostics):** when GAT is wrong, is it usually because neighborhoods are class-mixed or attention is drawn to misleading neighbors?

To avoid conflicting with the paper, we treat these as additional analyses rather than new claims about the original Table 2 target. In other words, baseline reproduction evaluates alignment with Veličković et al. (2018), while extensions evaluate robustness, efficiency, and interpretability under constrained settings.

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

**Run 100-trial evaluation (saves CSV to `results/`):**
```bash
python code/evaluate.py --dataset Cora
python code/evaluate.py --dataset CiteSeer
```

**Run extensions (low-label, distillation, failure cases):**
Open `notebooks/GAT_Colab.ipynb` in Google Colab with GPU enabled and run cells top to bottom. Results and figures are saved to `RESULTS_DIR` (default `/content/gat_results/`; set to a mounted Drive path to persist across sessions).

> GPU recommended — 100 runs on CPU is slow. Enable via Runtime → Change runtime type → T4 GPU.

## Results / Insights

### Baseline (Table 2 reproduction)

| Dataset  | Paper Result | Our Result |
|----------|-------------|------------|
| Cora     | 83.0 ± 0.7% | **83.22 ± 0.41%** ✅ |
| CiteSeer | 72.5 ± 0.7% | **70.94 ± 0.52%** |

Cora matches and slightly exceeds the paper's target. CiteSeer falls 1.56% short, likely due to our early stopping monitoring only validation loss rather than both loss and accuracy as specified in the original implementation.

### Extension Results

These experiments are supplementary to (not replacements for) the paper-aligned baseline above.

**Low-Label Benchmark** — both datasets degrade sharply below 50% labels, but CiteSeer collapses far more severely due to its sparser features. At 10% labels, one CiteSeer run hits 8.3% — below random chance for 6 classes (16.7%) — while Cora's worst run stays above 50%:

| Label fraction | Cora labels | Cora mean | Cora std | CiteSeer labels | CiteSeer mean | CiteSeer std |
|---|---|---|---|---|---|---|
| 10% | 14 | 58.84% | ±5.90% | 12 | 30.16% | ±17.89% |
| 20% | 28 | 69.04% | ±4.60% | 24 | 41.64% | ±20.60% |
| 50% | 70 | 79.96% | ±1.26% | 60 | 65.88% | ±1.49% |
| 100% | 140 | 83.06% | ±0.48% | 120 | 71.00% | ±0.42% |

![Low-label curve Cora](results/Cora_low_label_curve.png)
![Low-label curve CiteSeer](results/CiteSeer_low_label_curve.png)

**TinyGAT Distillation** — mean distillation gain of −0.24% on Cora and −0.66% on CiteSeer over 5 seeds. TinyGAT with supervision alone nearly matches the full GAT teacher on both datasets, suggesting neither dataset is complex enough for model capacity to be the bottleneck.

![Distillation gain Cora](results/Cora_distillation_gain.png)
![Distillation gain CiteSeer](results/CiteSeer_distillation_gain.png)

**Failure Case Explorer** — 25 misclassified nodes on Cora, 24 on CiteSeer. Most failures occur on nodes with mixed-class neighborhoods. On Cora, node 1358 appears as a high-attention source in 6 of 25 failure cases, acting as a hub that pulls surrounding predictions toward class 2; two errors exceed 92% confidence. On CiteSeer, failures are more diffuse with no dominant hub, and the highest-confidence error (node 2365, 85.3% wrong) involves two of three neighbors from the incorrect class.

![Failure confidence Cora](results/Cora_failure_confidence.png)
![Failure confidence CiteSeer](results/CiteSeer_failure_confidence.png)

## Conclusion

We successfully reproduced the GAT paper's Cora result (83.22% vs. 83.0%) and came close on CiteSeer (70.94% vs. 72.5%). Our extensions show that GAT's performance degrades sharply with few labels — CiteSeer collapses to near-random at 10% labels while Cora degrades gracefully, revealing a dataset-level sensitivity to label scarcity driven by feature sparsity. A 6× smaller TinyGAT already performs near the teacher on both datasets, making distillation minimally beneficial. Most misclassifications occur on structurally ambiguous nodes with mixed-class neighborhoods where attention cannot disambiguate the true label.

## References

- Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). [Graph Attention Networks](https://arxiv.org/abs/1710.10903). *ICLR 2018*.
- Fey, M. & Lenssen, J.E. (2019). [Fast Graph Representation Learning with PyTorch Geometric](https://arxiv.org/abs/1903.02428). *ICLR Workshop*.
- Yang, Z., Cohen, W., & Salakhutdinov, R. (2016). [Revisiting Semi-Supervised Learning with Graph Embeddings](https://arxiv.org/abs/1603.08861). *ICML*.

## Acknowledgements

This project was completed as part of **CS 4782: Deep Learning** at Cornell University. We thank the course staff for guidance throughout the project. Datasets were accessed via [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/). Experiments were run on Google Colab with GPU acceleration.
