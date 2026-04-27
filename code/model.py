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


class ConfigurableGAT(torch.nn.Module):
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_channels: int,
        heads_first: int,
        heads_second: int = 1,
        dropout: float = 0.6,
    ):
        super().__init__()
        self.dropout = dropout

        self.conv1 = GATConv(
            in_channels=num_features,
            out_channels=hidden_channels,
            heads=heads_first,
            dropout=dropout,
            concat=True,
        )

        self.conv2 = GATConv(
            in_channels=hidden_channels * heads_first,
            out_channels=num_classes,
            heads=heads_second,
            dropout=dropout,
            concat=False,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

    def forward_with_attention(self, x: torch.Tensor, edge_index: torch.Tensor):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x, attn1 = self.conv1(x, edge_index, return_attention_weights=True)
        x = F.elu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x, attn2 = self.conv2(x, edge_index, return_attention_weights=True)

        return F.log_softmax(x, dim=1), attn1, attn2


class GAT(ConfigurableGAT):
    def __init__(self, num_features: int, num_classes: int, dropout: float = 0.6):
        super().__init__(
            num_features=num_features,
            num_classes=num_classes,
            hidden_channels=8,
            heads_first=8,
            heads_second=1,
            dropout=dropout,
        )


class TinyGAT(ConfigurableGAT):
    def __init__(self, num_features: int, num_classes: int, dropout: float = 0.6):
        super().__init__(
            num_features=num_features,
            num_classes=num_classes,
            hidden_channels=4,
            heads_first=2,
            heads_second=1,
            dropout=dropout,
        )
