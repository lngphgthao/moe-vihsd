"""PhoBERT-backed Mixture of Experts classifier for ViHSD."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel

from models.moe import SparseMoE


class PretrainedBackboneClassifier(nn.Module):
    """Use PhoBERT contextual features as input to a mandatory MoE block."""

    def __init__(self, vocab_size: int, num_labels: int, config: dict) -> None:
        super().__init__()
        self.model_name = config.get("pretrained_model_name", "vinai/phobert-base")
        self.freeze_backbone = bool(config.get("freeze_backbone", False))
        self.pooling = str(config.get("pooling", "mean")).lower()
        if self.pooling not in {"mean", "cls"}:
            raise ValueError("pooling must be either 'mean' or 'cls'")
        self.backbone = AutoModel.from_pretrained(self.model_name)

        if self.freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        hidden_size = self.backbone.config.hidden_size
        model_dim = int(config.get("model_dim", 128))
        self.dropout = nn.Dropout(float(config.get("dropout", 0.1)))
        self.projection = nn.Linear(hidden_size, model_dim)
        self.moe = SparseMoE(
            model_dim=model_dim,
            expert_dim=int(config.get("expert_hidden_dim", 512)),
            num_experts=int(config.get("num_experts", 4)),
            top_k=int(config.get("top_k", 1)),
            dropout=float(config.get("dropout", 0.1)),
        )
        self.norm = nn.LayerNorm(model_dim)
        self.classifier = nn.Linear(model_dim, num_labels)
        self.num_experts = self.moe.num_experts

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        if self.pooling == "cls":
            pooled = hidden_states[:, 0]
        else:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        moe_input = self.projection(self.dropout(pooled)).unsqueeze(1)
        moe_output, routing = self.moe(moe_input)
        combined = self.norm(moe_input.squeeze(1) + moe_output.squeeze(1))
        logits = self.classifier(combined)
        return logits, routing
