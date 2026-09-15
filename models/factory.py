"""Model factory for switching architecture variants while keeping one training workflow."""

from __future__ import annotations

from typing import Any

from models.phobert_moe import PhoBERTMoEClassifier
from models.pretrained_backbone import PretrainedBackboneClassifier
from models.dense_phobert import DensePhoBERTClassifier

MODEL_REGISTRY: dict[str, Any] = {
    "pretrained_backbone": PretrainedBackboneClassifier,
    "phobert_moe": PhoBERTMoEClassifier,
    "dense_phobert": DensePhoBERTClassifier,
}


def validate_model_config(config: dict) -> str:
    """Validate architecture-specific settings before loading model weights."""
    architecture = str(config.get("architecture", "phobert_moe"))
    if architecture not in MODEL_REGISTRY:
        supported = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Unknown model architecture '{architecture}'. Supported architectures: {supported}"
        )
    if architecture == "dense_phobert":
        pooling = str(config.get("pooling", "mean")).lower()
        if pooling != "mean":
            raise ValueError(
                "dense_phobert requires model.pooling=mean. "
                "Pass --set model.pooling=mean when using the shared config."
            )
    return architecture


def standardize_model_output(model_output):
    """Normalize model output to the shared contract: (logits, aux_dict).

    This helper keeps the shared tuple/dictionary contract across all architectures.
    """
    if not isinstance(model_output, tuple):
        return model_output, {}
    if len(model_output) == 2:
        logits, aux = model_output
        if isinstance(aux, dict):
            return logits, aux
        return logits, {}
    if len(model_output) == 1:
        return model_output[0], {}
    raise ValueError(
        "Model output must be a tensor or a (logits, aux_dict) tuple; "
        f"got tuple with {len(model_output)} items."
    )


def build_model(config: dict, vocab_size: int, num_labels: int):
    """Instantiate a model from a config-driven architecture name.

    The registry includes the supported PhoBERT and dense architectures.
    """
    architecture = validate_model_config(config)
    try:
        model_cls = MODEL_REGISTRY[architecture]
    except KeyError as exc:
        supported = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Unknown model architecture '{architecture}'. Supported architectures: {supported}"
        ) from exc
    return model_cls(vocab_size, num_labels, config)
