"""Top-k router and auxiliary load-balancing loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKRouter(nn.Module):
    """Routes tokens to the top-k experts and exposes balancing statistics."""

    def __init__(self, input_dim: int, num_experts: int, top_k: int = 1) -> None:
        super().__init__()
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        if self.num_experts < 0:
            raise ValueError("num_experts must be non-negative")
        if self.num_experts == 0:
            self.projection = nn.Identity()
            return
        if self.top_k < 1 or self.top_k > self.num_experts:
            raise ValueError("top_k must be between 1 and num_experts")
        self.projection = nn.Linear(input_dim, num_experts)

    def forward(self, inputs):
        if self.num_experts == 0:
            empty_shape = inputs.shape[0], 0
            return (
                torch.empty(*empty_shape, device=inputs.device, dtype=inputs.dtype),
                torch.empty(*empty_shape, device=inputs.device, dtype=torch.long),
                torch.empty(*empty_shape, device=inputs.device, dtype=inputs.dtype),
            )
        logits = self.projection(inputs)
        probabilities = F.softmax(logits, dim=-1)
        top_values, top_indices = torch.topk(probabilities, self.top_k, dim=-1)
        weights = top_values / top_values.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return weights, top_indices, probabilities

    def load_balance_loss(self, probabilities, top_indices):
        """Penalize unequal probability mass and hard token counts."""
        if self.num_experts == 0:
            return torch.tensor(0.0, device=probabilities.device, dtype=probabilities.dtype)
        importance = probabilities.mean(dim=0)
        hard_counts = torch.bincount(top_indices.reshape(-1), minlength=self.num_experts).float()
        load = hard_counts / hard_counts.sum().clamp_min(1.0)
        target = torch.full_like(importance, 1.0 / self.num_experts)
        importance_loss = self.num_experts * torch.sum(importance * load)
        distribution_loss = F.mse_loss(importance, target) + F.mse_loss(load, target)
        return importance_loss + self.num_experts * distribution_loss
