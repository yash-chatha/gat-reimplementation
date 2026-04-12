# Datasets

This project uses **Cora** and **CiteSeer** for transductive node classification.

Both datasets are **automatically downloaded** by PyTorch Geometric the first time
`train.py` or `evaluate.py` is run — no manual download is required.

```python
from torch_geometric.datasets import Planetoid
dataset = Planetoid(root="/tmp/Cora", name="Cora")
```

Downloaded files are cached to `/tmp/<DatasetName>/` by default.
Change the `root` argument in `train.py` if you prefer a different cache location.

## Dataset Stats

| Dataset  | Nodes | Edges  | Features | Classes |
|----------|-------|--------|----------|---------|
| Cora     | 2,708 | 5,429  | 1,433    | 7       |
| CiteSeer | 3,327 | 4,732  | 3,703    | 6       |
