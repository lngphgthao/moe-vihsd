"""Helpers for loading compatible Hugging Face encoder backbones."""

from __future__ import annotations

from typing import Any

from transformers import AutoModel


def load_backbone(model_name: str) -> Any:
    """Load a Hugging Face encoder for a dense classifier."""
    return AutoModel.from_pretrained(model_name)


def load_moe_backbone(model_name: str) -> Any:
    """Load and validate a backbone whose internal FFNs will be replaced by MoE layers."""
    backbone = load_backbone(model_name)
    get_encoder_layers(backbone)
    return backbone


def get_encoder_layers(backbone: Any) -> Any:
    """Return encoder layers for BERT-, RoBERTa-, and XLM-R-style encoders."""
    encoder = getattr(backbone, "encoder", None)
    layers = getattr(encoder, "layer", None)
    if layers is None:
        model_type = getattr(getattr(backbone, "config", None), "model_type", "unknown")
        raise ValueError(
            "The selected backbone does not expose an encoder.layer stack required by "
            f"phobert_moe (model_type={model_type!r}). Use a BERT/RoBERTa-style encoder."
        )
    if len(layers) == 0:
        raise ValueError("The selected backbone has no encoder layers.")
    for index, layer in enumerate(layers):
        if not all(hasattr(layer, attribute) for attribute in ("attention", "intermediate", "output")):
            raise ValueError(
                f"Backbone encoder layer {index} is missing attention/intermediate/output "
                "modules required by phobert_moe."
            )
    return layers
