"""
model.py
GraphSAGE model for node-level AML classification.
Designed to fit comfortably in 8 GB VRAM (RTX 5060).
"""

from unittest import skip

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, BatchNorm


class GraphSAGE_AML(nn.Module):
    """
    3-layer GraphSAGE with:
      • BatchNorm after each conv layer
      • Dropout for regularisation
      • Skip-connection from input → final hidden layer
      • Binary output (normal / laundering)

    Architecture:
        [N, 8]
          ↓ SAGEConv(8→64)  + BN + ReLU + Dropout
        [N, 64]
          ↓ SAGEConv(64→128) + BN + ReLU + Dropout
        [N, 128]           ←← skip: Linear(8→128) added
          ↓ SAGEConv(128→64) + BN + ReLU
        [N, 64]
          ↓ Linear(64→2)
        [N, 2]  → class logits
    """

    def __init__(
        self,
        in_channels:  int   = 8,
        hidden_dim:   int   = 128,
        out_channels: int   = 2,
        dropout:      float = 0.3,
    ):
        super().__init__()
        self.dropout = dropout

        # Graph convolutions
        self.conv1 = SAGEConv(in_channels, hidden_dim // 2)
        self.conv2 = SAGEConv(hidden_dim // 2, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim // 2)

        # Batch normalisation
        self.bn1 = BatchNorm(hidden_dim // 2)
        self.bn2 = BatchNorm(hidden_dim)
        self.bn3 = BatchNorm(hidden_dim // 2)

        # Skip connection: project raw input to hidden_dim
        self.skip = nn.Linear(in_channels, hidden_dim)

        # Final classifier
        self.classifier = nn.Sequential(
        nn.Linear(hidden_dim // 2, 64),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(64, 16),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(16, out_channels),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Layer 1
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Layer 2
        skip = self.skip(x)

        h = self.conv2(h, edge_index)
        h = self.bn2(h)

        # project to same dimension
        h = h + skip

        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Layer 3  (no dropout before skip add)
        h = self.conv3(h, edge_index)
        h = self.bn3(h)
        h = F.relu(h)

        # Classifier head
        return self.classifier(h)

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return node embeddings before the final classifier layer."""
        h = F.relu(self.bn1(self.conv1(x, edge_index)))
        h = F.relu(self.bn2(self.conv2(h, edge_index)))
        h = F.relu(self.bn3(self.conv3(h, edge_index)))
        return h

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Quick sanity check ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = GraphSAGE_AML()
    print(model)
    print(f"Trainable parameters: {model.count_parameters():,}")

    # Fake forward pass
    N = 1000
    E = 5000
    x          = torch.randn(N, 8)
    edge_index = torch.randint(0, N, (2, E))
    logits     = model(x, edge_index)
    print(f"Input  shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Predicted classes: {logits.argmax(dim=1).unique(return_counts=True)}")
