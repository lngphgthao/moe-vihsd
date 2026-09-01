"""Pretrained transformer backbone for ViHSD classification."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class PretrainedBackboneClassifier(nn.Module):
    """Use a Hugging Face transformer as the main encoder and classify on pooled output."""

    def __init__(self, vocab_size: int, num_labels: int, config: dict) -> None:
        super().__init__()
        self.model_name = config.get("pretrained_model_name", "vinai/phobert-base")
        self.freeze_backbone = bool(config.get("freeze_backbone", False))
        self.backbone = AutoModel.from_pretrained(self.model_name)

        if self.freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(float(config.get("dropout", 0.1)))
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        logits = self.classifier(self.dropout(pooled))
        return logits, {}
