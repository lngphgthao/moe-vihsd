"""Sequence-level dynamic-threshold MoE classifier inspired by NLIMoE.

Supports two routing strategies selectable via ``model.routing_type``:

* ``"dynamic"`` (default) – learned per-input dynamic-threshold routing as in
  NLIMoEDynamic.  A complexity gate adjusts the activation threshold per
  example; experts whose softmax probability exceeds the threshold are active.
  Reports both ``dynamic_loss`` (entropy regulariser) and ``balance_loss``
  (load-balance penalty) in the aux dict.

* ``"top_k"`` – selects exactly the top-``model.dynamic_top_k`` experts per
  example by softmax probability, as in NLIMoETopK.  The complexity gate and
  dynamic threshold are not used.  Reports only ``balance_loss`` in the aux
  dict (``dynamic_loss`` is 0.0).

All other components (shared encoder, square single-layer FFN experts, mean
pooling, LayerNorm, linear classifier) are identical across both routing types,
so experiments isolate the routing choice.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbone import load_backbone


class DynamicMoEClassifier(nn.Module):
    """Mean-pool a shared encoder, then route each example by a learned threshold or top-k."""

    def __init__(self, vocab_size: int, num_labels: int, config: dict) -> None:
        super().__init__()
        self.model_name = config.get("pretrained_model_name")
        if not self.model_name:
            raise ValueError("model.pretrained_model_name is required for dynamic_moe")
        self.backbone = load_backbone(self.model_name)
        hidden_size = int(self.backbone.config.hidden_size)
        self.num_experts = int(config.get("dynamic_num_experts", 7))
        if self.num_experts < 1:
            raise ValueError("model.dynamic_num_experts must be at least 1 for dynamic_moe")

        # --- routing type ---
        self.routing_type = str(config.get("routing_type", "dynamic")).lower()
        if self.routing_type not in {"dynamic", "top_k"}:
            raise ValueError(
                f"model.routing_type must be 'dynamic' or 'top_k', got '{self.routing_type}'"
            )

        if self.routing_type == "top_k":
            self.top_k = int(config.get("dynamic_top_k", 2))
            if self.top_k < 1 or self.top_k > self.num_experts:
                raise ValueError(
                    f"model.dynamic_top_k ({self.top_k}) must be between 1 and "
                    f"model.dynamic_num_experts ({self.num_experts})"
                )
        else:
            # dynamic-threshold parameters
            self.static_threshold = float(config.get("dynamic_static_threshold", 0.1))
            self.threshold_scale = float(config.get("dynamic_threshold_scale", 0.1))
            self.threshold_temperature = float(config.get("dynamic_threshold_temperature", 0.02))
            if (self.static_threshold < 0 or self.threshold_scale < 0
                    or self.static_threshold + self.threshold_scale > 1):
                raise ValueError("dynamic routing threshold range must stay within [0, 1]")
            if self.threshold_temperature <= 0:
                raise ValueError("model.dynamic_threshold_temperature must be positive")
            self.complexity_gate = nn.Linear(hidden_size, 1)

        self.router = nn.Linear(hidden_size, self.num_experts)
        self.experts = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Dropout(float(config.get("dropout", 0.1))),
                nn.ReLU(),
            ) for _ in range(self.num_experts)]
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(float(config.get("dropout", 0.1)))
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.last_routing_info: dict[str, torch.Tensor] = {}
        if bool(config.get("freeze_backbone", False)):
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    # ------------------------------------------------------------------
    # Routing helpers
    # ------------------------------------------------------------------

    def _route_dynamic(
        self, pooled: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Dynamic-threshold routing.  Returns (weights, selected mask, probabilities, info)."""
        probabilities = F.softmax(self.router(pooled), dim=-1)
        complexity = torch.sigmoid(self.complexity_gate(pooled))
        threshold = self.static_threshold + self.threshold_scale * complexity
        selected = probabilities > threshold
        # Guarantee at least one expert per example.
        empty = ~selected.any(dim=-1)
        if empty.any():
            selected[empty, probabilities[empty].argmax(dim=-1)] = True

        temperature = float(self.threshold_temperature)
        soft_selected = torch.sigmoid((probabilities - threshold) / temperature)
        # Straight-through: hard mask in forward, soft gradient for threshold learning.
        selected_st = selected.to(probabilities.dtype) + soft_selected - soft_selected.detach()
        weights = probabilities * selected_st
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        info = {
            "probabilities": probabilities.detach(),
            "selected": selected.detach(),
            "complexity": complexity.squeeze(-1).detach(),
            "threshold": threshold.squeeze(-1).detach(),
        }
        return weights, selected, probabilities, info

    def _route_top_k(
        self, pooled: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Top-k routing.  Returns (weights, selected mask, probabilities, info)."""
        probabilities = F.softmax(self.router(pooled), dim=-1)
        top_vals, top_idx = torch.topk(probabilities, self.top_k, dim=-1)
        selected = torch.zeros_like(probabilities, dtype=torch.bool)
        selected.scatter_(1, top_idx, True)
        # Normalise only over selected experts.
        masked_probs = probabilities * selected.to(probabilities.dtype)
        weights = masked_probs / masked_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        info = {
            "probabilities": probabilities.detach(),
            "selected": selected.detach(),
            "top_indices": top_idx.detach(),
        }
        return weights, selected, probabilities, info

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        if self.routing_type == "top_k":
            weights, selected, probabilities, routing_info = self._route_top_k(pooled)
        else:
            weights, selected, probabilities, routing_info = self._route_dynamic(pooled)

        routed = torch.zeros_like(pooled)
        for expert_id, expert in enumerate(self.experts):
            rows = selected[:, expert_id].nonzero(as_tuple=True)[0]
            if rows.numel():
                expert_output = expert(pooled[rows])
                routed.index_add_(0, rows, expert_output * weights[rows, expert_id].unsqueeze(-1))

        logits = self.classifier(self.dropout(self.layer_norm(routed)))

        # Auxiliary losses
        # Entropy regulariser (dynamic loss): encourages focused expert selection.
        # Skipped for top_k routing since selection is already deterministically sparse.
        if self.routing_type == "dynamic":
            dynamic_loss = (
                -(probabilities * probabilities.clamp_min(1e-9).log()).sum(dim=-1).mean()
                / self.num_experts
            )
        else:
            dynamic_loss = torch.zeros(1, device=pooled.device, dtype=pooled.dtype).squeeze()

        # Load-balance loss: penalises uneven expert utilisation across the batch.
        load = selected.to(probabilities.dtype).mean(dim=0)
        importance = probabilities.mean(dim=0)
        balance_loss = self.num_experts * torch.sum(load * importance)

        self.last_routing_info = routing_info
        return logits, {"dynamic_loss": dynamic_loss, "balance_loss": balance_loss}
