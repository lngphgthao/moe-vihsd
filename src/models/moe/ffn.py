"""Expert feed-forward networks used by the MoE layer."""

import torch
import torch.nn as nn


class Expert(nn.Module):
    """A residual-compatible GELU or gated feed-forward expert."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        expert_type: str = "gelu",
    ) -> None:
        super().__init__()
        self.expert_type = expert_type.lower()
        if self.expert_type not in {"gelu", "geglu", "swiglu"}:
            raise ValueError("expert_type must be 'gelu', 'geglu', or 'swiglu'")

        if self.expert_type == "gelu":
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, input_dim),
            )
        else:
            self.value_projection = nn.Linear(input_dim, hidden_dim)
            self.gate_projection = nn.Linear(input_dim, hidden_dim)
            self.dropout = nn.Dropout(dropout)
            self.output_projection = nn.Linear(hidden_dim, input_dim)

    def upcycle_from_ffn(self, intermediate: nn.Linear, output: nn.Linear) -> None:
        """Copy a pretrained two-layer FFN into a compatible GELU expert."""
        if self.expert_type != "gelu":
            raise ValueError("Pretrained FFN upcycling is only supported for GELU experts")
        with torch.no_grad():
            self.network[0].weight.copy_(intermediate.weight)
            self.network[0].bias.copy_(intermediate.bias)
            self.network[3].weight.copy_(output.weight)
            self.network[3].bias.copy_(output.bias)

    def forward(self, inputs):
        if self.expert_type == "gelu":
            return self.network(inputs)
        gate = self.gate_projection(inputs)
        if self.expert_type == "geglu":
            gate = nn.functional.gelu(gate)
        else:
            gate = nn.functional.silu(gate)
        values = self.value_projection(inputs)
        return self.output_projection(self.dropout(values * gate))
