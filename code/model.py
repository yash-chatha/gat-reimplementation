"""
GAT Model — Graph Attention Networks (Veličković et al., ICLR 2018)
2-layer implementation using PyTorch Geometric's GATConv.

Architecture:
  Layer 1: 8 attention heads, each outputting 8 features (64 total), ELU activation
  Layer 2: 1 attention head, outputting num_classes features, log_softmax output
  Dropout: 0.6 applied to inputs of both layers
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):
    def __init__(self, num_features: int, num_classes: int, dropout: float = 0.6):
        super().__init__()
        self.dropout = dropout

        # Layer 1: 8 heads × 8 features = 64-dim output
        self.conv1 = GATConv(
            in_channels=num_features,
            out_channels=8,
            heads=8,
            dropout=dropout,
            concat=True,
        )

        # Layer 2: 1 head → num_classes (concat=False averages heads)
        self.conv2 = GATConv(
            in_channels=8 * 8,  # 64 from layer 1
            out_channels=num_classes,
            heads=1,
            dropout=dropout,
            concat=False,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
