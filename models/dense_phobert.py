"""Dense PhoBERT baseline classifier for ViHSD."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class DensePhoBERTClassifier(nn.Module):
    """Fine-tuned PhoBERT with mean pooling and a single linear classifier."""

    def __init__(self, vocab_size: int, num_labels: int, config: dict) -> None:
        super().__init__()
        if num_labels != 3:
            raise ValueError(f"Dense PhoBERT baseline requires exactly 3 labels, got {num_labels}")
        self.model_name = config.get("pretrained_model_name", "vinai/phobert-base")
        self.freeze_backbone = bool(config.get("freeze_backbone", False))
        self.pooling = str(config.get("pooling", "mean")).lower()
        if self.pooling != "mean":
            raise ValueError("Dense PhoBERT uses mean pooling")

        self.backbone = AutoModel.from_pretrained(self.model_name)
        if self.freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.classifier(pooled)