"""Generic dense Hugging Face encoder classifier."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class DenseTransformerClassifier(nn.Module):
    """Fine-tune a pretrained encoder with a linear sequence classifier."""

    def __init__(self, vocab_size: int, num_labels: int, config: dict) -> None:
        super().__init__()
        self.model_name = config.get("pretrained_model_name")
        if not self.model_name:
            raise ValueError("model.pretrained_model_name is required for dense_transformer")
        self.pooling = str(config.get("pooling", "cls")).lower()
        if self.pooling not in {"cls", "mean"}:
            raise ValueError("model.pooling must be 'cls' or 'mean'")
        self.freeze_backbone = bool(config.get("freeze_backbone", False))
        self.backbone = AutoModel.from_pretrained(self.model_name)
        if self.freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        if self.pooling == "cls":
            pooled = hidden_states[:, 0]
        else:
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.classifier(pooled)