"""Helper load các encoder backbone Hugging Face tương thích."""

from __future__ import annotations

from typing import Any

from transformers import AutoModel


def load_backbone(model_name: str, *, revision: str | None = None) -> Any:
    """Load encoder Hugging Face cho dense classifier."""
    kwargs = {} if revision is None else {"revision": revision}
    return AutoModel.from_pretrained(model_name, **kwargs)


def load_moe_backbone(model_name: str) -> Any:
    """Load và kiểm tra backbone có FFN nội bộ được thay bằng MoE layer."""
    backbone = load_backbone(model_name)
    get_encoder_layers(backbone)
    return backbone


def get_encoder_layers(backbone: Any) -> Any:
    """Trả về encoder layer cho encoder kiểu BERT, RoBERTa và XLM-R."""
    encoder = getattr(backbone, "encoder", None)
    layers = getattr(encoder, "layer", None)
    if layers is None:
        model_type = getattr(getattr(backbone, "config", None), "model_type", "unknown")
        raise ValueError(
            "The selected backbone does not expose an encoder.layer stack required by "
            f"transformer_moe (model_type={model_type!r}). Use a BERT/RoBERTa-style encoder."
        )
    if len(layers) == 0:
        raise ValueError("The selected backbone has no encoder layers.")
    for index, layer in enumerate(layers):
        if not all(hasattr(layer, attribute) for attribute in ("attention", "intermediate", "output")):
            raise ValueError(
                f"Backbone encoder layer {index} is missing attention/intermediate/output "
                "modules required by transformer_moe."
            )
    return layers
