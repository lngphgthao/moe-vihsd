"""Reusable sequence-level dynamic/top-k MoE head."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicMoEHead(nn.Module):
    """Route a pooled representation through dynamic-threshold or top-k experts."""

    def __init__(self, input_dim: int, num_labels: int, config: dict) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_labels = int(num_labels)
        self.num_experts = int(config.get("num_experts", config.get("dynamic_num_experts", 4)))
        if self.num_experts < 1:
            raise ValueError("dynamic MoE head num_experts must be at least 1")
        self.routing_type = str(config.get("routing_type", "dynamic")).lower()
        if self.routing_type not in {"dynamic", "top_k"}:
            raise ValueError("dynamic MoE head routing_type must be 'dynamic' or 'top_k'")

        if self.routing_type == "top_k":
            self.top_k = int(config.get("top_k", config.get("dynamic_top_k", 2)))
            if self.top_k < 1 or self.top_k > self.num_experts:
                raise ValueError("dynamic MoE head top_k must be between 1 and num_experts")
        else:
            self.static_threshold = float(config.get("static_threshold", config.get("dynamic_static_threshold", 0.1)))
            self.threshold_scale = float(config.get("threshold_scale", config.get("dynamic_threshold_scale", 0.1)))
            self.threshold_temperature = float(config.get("threshold_temperature", config.get("dynamic_threshold_temperature", 0.02)))
            if self.static_threshold < 0 or self.threshold_scale < 0 or self.static_threshold + self.threshold_scale > 1:
                raise ValueError("dynamic MoE head threshold range must stay within [0, 1]")
            if self.threshold_temperature <= 0:
                raise ValueError("dynamic MoE head threshold_temperature must be positive")
            self.complexity_gate = nn.Linear(self.input_dim, 1)

        expert_hidden_dim = int(config.get("expert_hidden_dim", self.input_dim))
        expert_dropout = float(config.get("expert_dropout", config.get("dropout", 0.1)))
        classifier_dropout = float(config.get("classifier_dropout", config.get("dropout", 0.1)))
        self.router = nn.Linear(self.input_dim, self.num_experts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.input_dim, expert_hidden_dim),
                    nn.Dropout(expert_dropout),
                    nn.ReLU(),
                    nn.Linear(expert_hidden_dim, self.input_dim),
                )
                for _ in range(self.num_experts)
            ]
        )
        self.layer_norm = nn.LayerNorm(self.input_dim)
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.input_dim, self.num_labels)
        self.last_routing_info: dict[str, Any] = {}

    def _route(self, inputs: torch.Tensor):
        probabilities = F.softmax(self.router(inputs), dim=-1)
        if self.routing_type == "top_k":
            _, top_indices = torch.topk(probabilities, self.top_k, dim=-1)
            selected = torch.zeros_like(probabilities, dtype=torch.bool)
            selected.scatter_(1, top_indices, True)
            weights = probabilities * selected.to(probabilities.dtype)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            return weights, selected, probabilities, {"top_indices": top_indices.detach()}

        complexity = torch.sigmoid(self.complexity_gate(inputs))
        threshold = self.static_threshold + self.threshold_scale * complexity
        selected = probabilities > threshold
        empty = ~selected.any(dim=-1)
        if empty.any():
            selected[empty, probabilities[empty].argmax(dim=-1)] = True
        soft_selected = torch.sigmoid(
            (probabilities - threshold) / self.threshold_temperature
        )
        selected_st = selected.to(probabilities.dtype) + soft_selected - soft_selected.detach()
        weights = probabilities * selected_st
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return weights, selected, probabilities, {
            "complexity": complexity.squeeze(-1).detach(),
            "threshold": threshold.squeeze(-1).detach(),
        }

    def forward(self, inputs: torch.Tensor):
        weights, selected, probabilities, routing_info = self._route(inputs)
        routed = torch.zeros_like(inputs)
        for expert_id, expert in enumerate(self.experts):
            rows = selected[:, expert_id].nonzero(as_tuple=True)[0]
            if rows.numel():
                expert_output = expert(inputs[rows])
                routed.index_add_(0, rows, expert_output * weights[rows, expert_id].unsqueeze(-1))

        logits = self.classifier(self.dropout(self.layer_norm(routed)))
        dynamic_loss = (
            -(probabilities * probabilities.clamp_min(1e-9).log()).sum(dim=-1).mean()
            / self.num_experts
            if self.routing_type == "dynamic"
            else inputs.new_zeros(())
        )
        load = selected.to(probabilities.dtype).mean(dim=0)
        importance = probabilities.mean(dim=0)
        balance_loss = self.num_experts * torch.sum(load * importance)
        self.last_routing_info = {
            "probabilities": probabilities.detach(),
            "selected": selected.detach(),
            **routing_info,
        }
        return logits, {
            "dynamic_loss": dynamic_loss,
            "balance_loss": balance_loss,
            "probabilities": probabilities,
            "selected": selected,
            **routing_info,
        }
