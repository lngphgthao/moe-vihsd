"""Loss function implementations for ViHSD experiments."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_task_loss(logits: torch.Tensor, labels: torch.Tensor, config: dict) -> torch.Tensor:
    """Compute the configured classification loss (cross_entropy, weighted_cross_entropy, focal) for a batch."""
    training_config = config.get("training", {})
    loss_type = str(training_config.get("loss_type", "cross_entropy")).lower()
    class_weights = training_config.get("class_weights")
    aliases = {
        "ce": "cross_entropy",
        "weighted_ce": "weighted_cross_entropy",
        "class_weighted_ce": "weighted_cross_entropy",
    }
    loss_type = aliases.get(loss_type, loss_type)

    valid_loss_types = {"cross_entropy", "weighted_cross_entropy", "focal"}
    if loss_type not in valid_loss_types:
        raise ValueError(
            f"Unsupported training.loss_type={loss_type!r}; "
            f"choose one of {sorted(valid_loss_types)}."
        )

    weight_tensor = None
    if class_weights is not None:
        if len(class_weights) != logits.size(-1):
            raise ValueError(
                "training.class_weights must contain one weight per class "
                f"({logits.size(-1)} expected, got {len(class_weights)})."
            )
        weight_tensor = torch.as_tensor(
            class_weights,
            dtype=logits.dtype,
            device=logits.device,
        )

    if loss_type == "weighted_cross_entropy" and weight_tensor is None:
        raise ValueError(
            "training.class_weights is required when "
            "training.loss_type is 'weighted_cross_entropy'."
        )

    if loss_type in {"cross_entropy", "weighted_cross_entropy"}:
        return F.cross_entropy(logits, labels, weight=weight_tensor)

    gamma = float(training_config.get("focal_gamma", 2.0))
    if gamma < 0:
        raise ValueError("training.focal_gamma must be non-negative.")
    per_example_loss = F.cross_entropy(logits, labels, reduction="none")
    target_probs = logits.softmax(dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)
    focal_factor = (1.0 - target_probs).clamp_min(0.0).pow(gamma)
    if weight_tensor is not None:
        per_example_loss = per_example_loss * weight_tensor[labels]
    return (focal_factor * per_example_loss).mean()
