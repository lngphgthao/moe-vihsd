"""Reusable sparse Mixture of Experts block."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.expert import Expert
from models.router import TopKRouter


class SparseMoE(nn.Module):
    """Dispatch each token to selected experts and combine their outputs."""

    def __init__(self, model_dim: int, expert_dim: int, num_experts: int, top_k: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.router = TopKRouter(model_dim, num_experts, top_k)
        self.experts = nn.ModuleList(Expert(model_dim, expert_dim, dropout) for _ in range(num_experts))

    def forward(self, tokens):
        batch_size, sequence_length, model_dim = tokens.shape
        flat_tokens = tokens.reshape(-1, model_dim)
        weights, indices, probabilities = self.router(flat_tokens)
        output = torch.zeros_like(flat_tokens)
        for expert_id, expert in enumerate(self.experts):
            selected_tokens = indices == expert_id
            if not selected_tokens.any():
                continue
            token_positions, choice_positions = selected_tokens.nonzero(as_tuple=True)
            expert_output = expert(flat_tokens[token_positions])
            output.index_add_(0, token_positions, expert_output * weights[token_positions, choice_positions].unsqueeze(-1))
        return output.reshape(batch_size, sequence_length, model_dim), {
            "top_indices": indices,
            "probabilities": probabilities,
            "balance_loss": self.router.load_balance_loss(probabilities, indices),
        }
