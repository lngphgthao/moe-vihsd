"""Sequence-level dynamic-threshold MoE classifier inspired by NLIMoE."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbone import load_backbone


class DynamicMoEClassifier(nn.Module):
    """Mean-pool a shared encoder, then route each example by a learned threshold."""

    def __init__(self, vocab_size: int, num_labels: int, config: dict) -> None:
        super().__init__()
        self.model_name = config.get("pretrained_model_name")
        if not self.model_name:
            raise ValueError("model.pretrained_model_name is required for dynamic_moe")
        self.backbone = load_backbone(self.model_name)
        hidden_size = int(self.backbone.config.hidden_size)
        self.num_experts = int(config.get("dynamic_num_experts", 7))
        if self.num_experts < 1:
            raise ValueError("model.num_experts must be at least 1 for dynamic_moe")
        self.static_threshold = float(config.get("dynamic_static_threshold", 0.1))
        self.threshold_scale = float(config.get("dynamic_threshold_scale", 0.1))
        self.threshold_temperature = float(config.get("dynamic_threshold_temperature", 0.02))
        if (self.static_threshold < 0 or self.threshold_scale < 0
                or self.static_threshold + self.threshold_scale > 1):
            raise ValueError("dynamic routing threshold range must stay within [0, 1]")
        if self.threshold_temperature <= 0:
            raise ValueError("model.dynamic_threshold_temperature must be positive")

        self.router = nn.Linear(hidden_size, self.num_experts)
        self.complexity_gate = nn.Linear(hidden_size, 1)
        self.experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Dropout(float(config.get("dropout", 0.1))), nn.ReLU())
             for _ in range(self.num_experts)]
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(float(config.get("dropout", 0.1)))
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.last_routing_info: dict[str, torch.Tensor] = {}
        if bool(config.get("freeze_backbone", False)):
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        probabilities = F.softmax(self.router(pooled), dim=-1)
        complexity = torch.sigmoid(self.complexity_gate(pooled))
        threshold = self.static_threshold + self.threshold_scale * complexity
        selected = probabilities > threshold
        # Keep at least one expert active when every probability is below threshold.
        empty = ~selected.any(dim=-1)
        if empty.any():
            selected[empty, probabilities[empty].argmax(dim=-1)] = True

        temperature = float(self.threshold_temperature)
        soft_selected = torch.sigmoid((probabilities - threshold) / temperature)
        # Straight-through threshold mask: hard routing in the forward pass, while
        # allowing the complexity gate to learn from the effect of its threshold.
        selected_st = selected.to(probabilities.dtype) + soft_selected - soft_selected.detach()
        weights = probabilities * selected_st
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        routed = torch.zeros_like(pooled)
        for expert_id, expert in enumerate(self.experts):
            rows = selected[:, expert_id].nonzero(as_tuple=True)[0]
            if rows.numel():
                expert_output = expert(pooled[rows])
                routed.index_add_(0, rows, expert_output * weights[rows, expert_id].unsqueeze(-1))
        logits = self.classifier(self.dropout(self.layer_norm(routed)))

        entropy_loss = -(probabilities * probabilities.clamp_min(1e-9).log()).sum(dim=-1).mean() / self.num_experts
        load = selected.to(probabilities.dtype).mean(dim=0)
        importance = probabilities.mean(dim=0)
        balance_loss = self.num_experts * torch.sum(load * importance)
        self.last_routing_info = {
            "probabilities": probabilities.detach(),
            "selected": selected.detach(),
            "complexity": complexity.squeeze(-1).detach(),
            "threshold": threshold.squeeze(-1).detach(),
        }
        return logits, {"dynamic_loss": entropy_loss, "balance_loss": balance_loss}
