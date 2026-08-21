"""Top-k router and auxiliary load-balancing loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKRouter(nn.Module):
    """Routes tokens to the top-k experts and exposes balancing statistics."""

    def __init__(self, input_dim: int, num_experts: int, top_k: int = 1) -> None:
        super().__init__()
        if top_k < 1 or top_k > num_experts:
            raise ValueError("top_k must be between 1 and num_experts")
        self.num_experts = num_experts
        self.top_k = top_k
        self.projection = nn.Linear(input_dim, num_experts)

    def forward(self, inputs):
        logits = self.projection(inputs)
        probabilities = F.softmax(logits, dim=-1)
        top_values, top_indices = torch.topk(probabilities, self.top_k, dim=-1)
        weights = top_values / top_values.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return weights, top_indices, probabilities

    def load_balance_loss(self, probabilities, top_indices):
        """Penalize unequal probability mass and hard token counts."""
        importance = probabilities.mean(dim=0)
        hard_counts = torch.bincount(top_indices.reshape(-1), minlength=self.num_experts).float()
        load = hard_counts / hard_counts.sum().clamp_min(1.0)
        target = torch.full_like(importance, 1.0 / self.num_experts)
        importance_loss = self.num_experts * torch.sum(importance * load)
        distribution_loss = F.mse_loss(importance, target) + F.mse_loss(load, target)
        return importance_loss + self.num_experts * distribution_loss
