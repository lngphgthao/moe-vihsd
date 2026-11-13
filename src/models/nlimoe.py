"""Classifier MoE dynamic-threshold theo sequence, lấy cảm hứng từ NLIMoE.

Hỗ trợ hai routing strategy, chọn qua ``model.routing_type``:

* ``"dynamic"`` (mặc định): dynamic-threshold routing học theo input như
  NLIMoEDynamic. Complexity gate điều chỉnh activation threshold cho từng
  example; expert có softmax probability vượt threshold sẽ active. Ghi cả
  ``dynamic_loss`` (entropy regulariser) và ``balance_loss`` (load-balance
  penalty) vào aux dict.

* ``"top_k"``: chọn đúng top-``model.dynamic_top_k`` expert cho mỗi example
  theo softmax probability như NLIMoETopK. Không dùng complexity gate và
  dynamic threshold. Chỉ ghi ``balance_loss`` vào aux dict, còn
  ``dynamic_loss`` bằng 0.0.

Các component còn lại, gồm shared encoder, single-layer FFN expert dạng square,
mean pooling, LayerNorm và linear classifier, giống nhau ở cả hai routing type.
Nhờ đó experiment cô lập được ảnh hưởng của routing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbone import load_backbone


class NLIMoEClassifier(nn.Module):
    """Mean-pool shared encoder, rồi route từng example bằng threshold hoặc top-k."""

    def __init__(self, vocab_size: int, num_labels: int, config: dict) -> None:
        super().__init__()
        self.model_name = config.get("pretrained_model_name")
        if not self.model_name:
            raise ValueError("model.pretrained_model_name is required for nlimoe")
        self.backbone = load_backbone(
            self.model_name, revision=config.get("pretrained_model_revision")
        )
        hidden_size = int(self.backbone.config.hidden_size)
        self.num_experts = int(config.get("dynamic_num_experts", 7))
        if self.num_experts < 1:
            raise ValueError("model.dynamic_num_experts must be at least 1 for nlimoe")

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
            # Tham số dynamic-threshold.
            self.static_threshold = float(config.get("dynamic_static_threshold", 0.1))
            self.threshold_scale = float(config.get("dynamic_threshold_scale", 0.1))
            self.threshold_temperature = float(config.get("dynamic_threshold_temperature", 0.02))
            if (self.static_threshold < 0 or self.threshold_scale < 0
                    or self.static_threshold + self.threshold_scale > 1):
                raise ValueError("dynamic routing threshold range must stay within [0, 1]")
            if self.threshold_temperature <= 0:
                raise ValueError("model.dynamic_threshold_temperature must be positive")
            self.complexity_gate = nn.Linear(hidden_size, 1)

        nlimoe_expert_dropout = float(
            config.get("nlimoe_expert_dropout", config.get("dropout", 0.1))
        )
        self.router = nn.Linear(hidden_size, self.num_experts)
        self.experts = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Dropout(nlimoe_expert_dropout),
                nn.ReLU(),
            ) for _ in range(self.num_experts)]
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(float(config.get("classifier_dropout", nlimoe_expert_dropout)))
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.last_routing_info: dict[str, torch.Tensor] = {}
        if bool(config.get("freeze_backbone", False)):
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    # ------------------------------------------------------------------
    # Helper cho routing.
    # ------------------------------------------------------------------

    def _route_dynamic(
        self, pooled: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Dynamic-threshold routing, trả về weights, mask, probabilities và info."""
        probabilities = F.softmax(self.router(pooled), dim=-1)
        complexity = torch.sigmoid(self.complexity_gate(pooled))
        threshold = self.static_threshold + self.threshold_scale * complexity
        selected = probabilities > threshold
        # Bài không mô tả mask rỗng; giữ argmax fallback và cần đối chiếu author code.
        empty = ~selected.any(dim=-1)
        if empty.any():
            selected[empty, probabilities[empty].argmax(dim=-1)] = True

        temperature = float(self.threshold_temperature)
        soft_selected = torch.sigmoid((probabilities - threshold) / temperature)
        # Bài không mô tả gradient threshold; cần đối chiếu straight-through estimator với author code.
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
        """Top-k routing, trả về weights, mask, probabilities và info."""
        probabilities = F.softmax(self.router(pooled), dim=-1)
        top_vals, top_idx = torch.topk(probabilities, self.top_k, dim=-1)
        selected = torch.zeros_like(probabilities, dtype=torch.bool)
        selected.scatter_(1, top_idx, True)
        # Chỉ chuẩn hóa trên các expert được chọn.
        masked_probs = probabilities * selected.to(probabilities.dtype)
        weights = masked_probs / masked_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        info = {
            "probabilities": probabilities.detach(),
            "selected": selected.detach(),
            "top_indices": top_idx.detach(),
        }
        return weights, selected, probabilities, info

    # ------------------------------------------------------------------
    # Forward.
    # ------------------------------------------------------------------

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor | None = None):
        segment_inputs = {} if token_type_ids is None else {"token_type_ids": token_type_ids}
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask, **segment_inputs).last_hidden_state
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

        # Auxiliary loss.
        # Entropy regulariser (dynamic loss) khuyến khích chọn expert tập trung.
        # Bỏ qua với top_k vì selection đã sparse một cách deterministic.
        if self.routing_type == "dynamic":
            dynamic_loss = (
                -(probabilities * probabilities.clamp_min(1e-9).log()).sum(dim=-1).mean()
                / self.num_experts
            )
        else:
            dynamic_loss = torch.zeros(1, device=pooled.device, dtype=pooled.dtype).squeeze()

        # Load-balance loss phạt mức sử dụng expert không đều trong batch.
        load = selected.to(probabilities.dtype).mean(dim=0)
        importance = probabilities.mean(dim=0)
        balance_loss = self.num_experts * torch.sum(load * importance)

        self.last_routing_info = routing_info
        return logits, {"dynamic_loss": dynamic_loss, "balance_loss": balance_loss}
