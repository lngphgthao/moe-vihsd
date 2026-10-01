"""Hybrid Transformer-MoE encoder followed by a Classifier-MoE head."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.models.classifier_moe import ClassifierMoEHead
from src.models.moe.layer import TransformerMoEClassifier


class HybridMoEClassifier(nn.Module):
    """Route inside selected Transformer layers, then route the pooled representation."""

    def __init__(self, vocab_size: int, num_labels: int, config: dict) -> None:
        super().__init__()
        self.model_name = config.get("pretrained_model_name")
        if not self.model_name:
            raise ValueError("model.pretrained_model_name is required for hybrid_moe")

        transformer_config = dict(config)
        transformer_config.update(config.get("transformer_moe", {}))
        transformer_config["architecture"] = "transformer_moe"
        transformer_config["pretrained_model_name"] = self.model_name
        transformer_config["num_experts"] = int(
            config.get("transformer_moe", {}).get("num_experts", 6)
        )
        transformer_config["routing_method"] = str(
            config.get("transformer_moe", {}).get("routing_method", "dynamic_threshold")
        )
        transformer_branch = TransformerMoEClassifier(vocab_size, num_labels, transformer_config)
        self.backbone = transformer_branch.roberta
        self.moe_layers = transformer_branch.moe_layers
        self.moe_layer_indices = transformer_branch.moe_layer_indices
        self.pooling = transformer_branch.pooling
        self.num_transformer_experts = transformer_branch.num_experts
        self.transformer_top_k = transformer_branch.top_k
        self.transformer_routing_method = transformer_config["routing_method"]
        del transformer_branch

        head_config = dict(config.get("classifier_moe", {}))
        head_config.setdefault("num_experts", 4)
        head_config.setdefault("top_k", 2)
        head_config.setdefault("expert_hidden_dim", 512)
        head_config.setdefault("dropout", config.get("dropout", 0.1))
        head_config.setdefault("load_balance_loss_coef", 0.01)
        head_config.setdefault("shared_expert", False)
        head_config.setdefault("use_residual", False)
        head_config.setdefault("routing_type", "top_k")
        head_config.setdefault("activation", "gelu")
        self.classifier_moe = ClassifierMoEHead(
            input_dim=int(self.backbone.config.hidden_size),
            num_labels=num_labels,
            num_experts=int(head_config["num_experts"]),
            top_k=int(head_config["top_k"]),
            expert_hidden_dim=int(head_config["expert_hidden_dim"]),
            dropout=float(head_config["dropout"]),
            load_balance_loss_coef=float(head_config["load_balance_loss_coef"]),
            shared_expert=bool(head_config["shared_expert"]),
            use_residual=bool(head_config["use_residual"]),
            routing_type=str(head_config["routing_type"]),
            activation=str(head_config["activation"]),
        )
        self.num_classifier_experts = self.classifier_moe.num_experts
        self.classifier_top_k = self.classifier_moe.top_k

        self.freeze_attention = bool(config.get("freeze_attention", False))
        self.freeze_embeddings = bool(config.get("freeze_embeddings", False))
        self._apply_freezing()

    def _apply_freezing(self) -> None:
        if self.freeze_embeddings:
            for parameter in self.backbone.embeddings.parameters():
                parameter.requires_grad = False
        if self.freeze_attention:
            for layer in self.backbone.encoder.layer:
                for parameter in layer.attention.parameters():
                    parameter.requires_grad = False
                if layer not in self.moe_layers.values():
                    for parameter in layer.intermediate.parameters():
                        parameter.requires_grad = False
                    for parameter in layer.output.parameters():
                        parameter.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        if self.pooling == "cls":
            pooled = hidden_states[:, 0]
        else:
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        logits, classifier_aux = self.classifier_moe(pooled)
        transformer_balance_loss = pooled.new_zeros(())
        transformer_dynamic_loss = pooled.new_zeros(())
        transformer_routing: dict[str, dict[str, Any]] = {}
        for layer_name, layer in self.moe_layers.items():
            info = layer.last_routing_info
            if not info:
                continue
            transformer_balance_loss = transformer_balance_loss + info.get("balance_loss", 0.0)
            transformer_dynamic_loss = transformer_dynamic_loss + info.get("dynamic_loss", 0.0)
            transformer_routing[layer_name] = info

        classifier_balance_loss = classifier_aux["balance_loss"]
        aux = {
            "balance_loss": transformer_balance_loss + classifier_balance_loss,
            "dynamic_loss": transformer_dynamic_loss,
            "transformer_balance_loss": transformer_balance_loss,
            "classifier_balance_loss": classifier_balance_loss,
            "transformer_dynamic_loss": transformer_dynamic_loss,
            "layer_routing": {
                **transformer_routing,
                "classifier_moe": {
                    "probabilities": classifier_aux["probabilities"],
                    "top_indices": classifier_aux["top_indices"],
                    "balance_loss": classifier_balance_loss,
                },
            },
            "classifier_moe": classifier_aux,
        }
        return logits, aux
