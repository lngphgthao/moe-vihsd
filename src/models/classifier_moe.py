"""Module Mixture of Experts ở classifier level (Classifier-MoE).

Module này giữ Transformer encoder backbone ở dạng dense và thay classification
head bằng mạng Mixture of Experts (MoE) sparse.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbone import load_backbone
from src.models.moe.router import TopKRouter


class ClassifierMoEExpert(nn.Module):
    """Expert MLP hai layer: Linear -> activation -> dropout -> Linear."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation_name = activation.lower()

        if self.activation_name == "gelu":
            act_module = nn.GELU()
        elif self.activation_name == "relu":
            act_module = nn.ReLU()
        elif self.activation_name == "silu":
            act_module = nn.SiLU()
        else:
            raise ValueError(
                f"Unsupported activation '{activation}'. Supported activations: 'gelu', 'relu', 'silu'."
            )

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act_module,
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ClassifierMoEHead(nn.Module):
    """Routing head của Classifier-MoE.

    1. Router nhận representation h và tính routing score của expert.
    2. Dùng top-k routing để chọn k expert cho mỗi example.
    3. Phân phối token/example tới N expert MLP hai layer độc lập.
    4. Kết hợp output được chọn bằng routing weight đã chuẩn hóa.
    5. Có thể kết hợp thêm shared expert và/hoặc residual connection.
    6. Đưa representation kết hợp vào linear classification layer cuối.
    7. Tính auxiliary load-balancing loss.
    """

    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        num_experts: int = 4,
        top_k: int = 2,
        expert_hidden_dim: int = 512,
        dropout: float = 0.1,
        load_balance_loss_coef: float = 0.01,
        shared_expert: bool = False,
        use_residual: bool = False,
        routing_type: str = "top_k",
        activation: str = "gelu",
        classifier_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_labels = int(num_labels)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.expert_hidden_dim = int(expert_hidden_dim)
        self.dropout_rate = float(dropout)
        self.load_balance_loss_coef = float(load_balance_loss_coef)
        self.use_shared_expert = bool(shared_expert)
        self.use_residual = bool(use_residual)
        self.routing_type = str(routing_type).lower()
        self.activation = str(activation).lower()
        self.classifier_dropout = nn.Dropout(float(classifier_dropout))

        if self.num_experts < 1:
            raise ValueError("num_experts must be at least 1")
        if self.top_k < 1 or self.top_k > self.num_experts:
            raise ValueError(f"top_k ({self.top_k}) must be between 1 and num_experts ({self.num_experts})")
        if self.routing_type != "top_k":
            raise ValueError(f"Unsupported routing_type '{routing_type}'. Supported: 'top_k'.")

        self.router = TopKRouter(self.input_dim, self.num_experts, self.top_k)
        self.experts = nn.ModuleList(
            [
                ClassifierMoEExpert(
                    input_dim=self.input_dim,
                    hidden_dim=self.expert_hidden_dim,
                    dropout=self.dropout_rate,
                    activation=self.activation,
                )
                for _ in range(self.num_experts)
            ]
        )

        if self.use_shared_expert:
            self.shared_expert = ClassifierMoEExpert(
                input_dim=self.input_dim,
                hidden_dim=self.expert_hidden_dim,
                dropout=self.dropout_rate,
                activation=self.activation,
            )
        else:
            self.shared_expert = None

        self.classifier = nn.Linear(self.input_dim, self.num_labels)
        self.last_routing_info: dict[str, Any] = {}

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        """Thực hiện forward pass qua Classifier-MoE head.

        Args:
            h: Pooled representation tensor có shape (batch_size, input_dim).

        Returns:
            logits: Classification logits có shape (batch_size, num_labels).
            aux: Dictionary chứa auxiliary loss và routing metric.
        """
        weights, top_indices, probabilities = self.router(h)

        moe_output = torch.zeros_like(h)
        for expert_id, expert in enumerate(self.experts):
            expert_mask = (top_indices == expert_id)
            if not expert_mask.any():
                continue
            sample_indices, choice_indices = expert_mask.nonzero(as_tuple=True)
            expert_out = expert(h[sample_indices])
            expert_weight = weights[sample_indices, choice_indices].unsqueeze(-1)
            moe_output.index_add_(0, sample_indices, expert_out * expert_weight)

        if self.shared_expert is not None:
            shared_out = self.shared_expert(h)
            moe_output = moe_output + shared_out

        if self.use_residual:
            combined = h + moe_output
        else:
            combined = moe_output

        logits = self.classifier(self.classifier_dropout(combined))
        balance_loss = self.router.load_balance_loss(probabilities, top_indices)

        self.last_routing_info = {
            "balance_loss": balance_loss.detach(),
            "probabilities": probabilities.detach(),
            "top_indices": top_indices.detach(),
        }

        aux = {
            "balance_loss": balance_loss,
            "load_balance_loss": balance_loss,
            "load_balance_loss_coef": self.load_balance_loss_coef,
            "weighted_balance_loss": self.load_balance_loss_coef * balance_loss,
            "probabilities": probabilities,
            "top_indices": top_indices,
            "routing_weights": weights,
        }
        return logits, aux


class ClassifierMoEClassifier(nn.Module):
    """Dense Transformer encoder backbone với Classifier-MoE classification head."""

    def __init__(self, vocab_size: int, num_labels: int, config: dict) -> None:
        super().__init__()
        self.model_name = config.get("pretrained_model_name")
        if not self.model_name:
            raise ValueError("model.pretrained_model_name is required for classifier_moe")

        self.pooling = str(config.get("pooling", "cls")).lower()
        if self.pooling not in {"cls", "mean"}:
            raise ValueError("model.pooling must be 'cls' or 'mean'")

        self.freeze_backbone = bool(config.get("freeze_backbone", False))
        self.backbone = load_backbone(self.model_name)
        if self.freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        hidden_size = int(self.backbone.config.hidden_size)

        num_experts = int(config.get("num_experts", 4))
        top_k = int(config.get("top_k", 2))
        routing_type = str(config.get("routing_type", config.get("routing_method", "top_k"))).lower()
        expert_hidden_dim = int(config.get("expert_hidden_dim", config.get("expert_hidden_size", 512) or 512))
        dropout = float(
            config.get("classifier_moe_dropout", config.get("dropout", 0.1))
        )
        load_balance_loss_coef = float(config.get("load_balance_loss_coef", config.get("load_balance_loss_factor", 0.01)))
        shared_expert = bool(config.get("shared_expert", False))
        use_residual = bool(config.get("use_residual", False))
        activation = str(config.get("activation", config.get("expert_type", "gelu"))).lower()

        self.num_experts = num_experts
        self.top_k = top_k
        self.classifier_moe = ClassifierMoEHead(
            input_dim=hidden_size,
            num_labels=num_labels,
            num_experts=num_experts,
            top_k=top_k,
            expert_hidden_dim=expert_hidden_dim,
            dropout=dropout,
            load_balance_loss_coef=load_balance_loss_coef,
            shared_expert=shared_expert,
            use_residual=use_residual,
            routing_type=routing_type,
            activation=activation,
            classifier_dropout=0.0,
        )

    @property
    def last_routing_info(self) -> dict[str, Any]:
        return self.classifier_moe.last_routing_info

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
        segment_inputs = {} if token_type_ids is None else {"token_type_ids": token_type_ids}
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, **segment_inputs)
        hidden_states = outputs.last_hidden_state

        if self.pooling == "cls":
            pooled = hidden_states[:, 0]
        else:
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        return self.classifier_moe(pooled)
