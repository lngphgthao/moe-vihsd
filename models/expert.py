"""Expert feed-forward networks used by the MoE layer."""

import torch.nn as nn


class Expert(nn.Module):
    """A two-layer feed-forward expert with residual-compatible dimensions."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, inputs):
        return self.network(inputs)
