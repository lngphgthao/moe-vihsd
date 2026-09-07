"""Model factory for switching architecture variants while keeping one training workflow."""

from __future__ import annotations

from typing import Any

from models.moe import ViHSDMoEClassifier
from models.moe_v2 import StrongerViHSDMoEClassifier
from models.pretrained_backbone import PretrainedBackboneClassifier

MODEL_REGISTRY: dict[str, Any] = {
    "current_moe": ViHSDMoEClassifier,
    "stronger_moe": StrongerViHSDMoEClassifier,
    "pretrained_backbone": PretrainedBackboneClassifier,
}


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

    Every registered architecture includes an MoE block; the default remains the
    current MoE model so the existing workflow keeps its original semantics.
    """
    architecture = str(config.get("architecture", "current_moe"))
    try:
        model_cls = MODEL_REGISTRY[architecture]
    except KeyError as exc:
        supported = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Unknown model architecture '{architecture}'. Supported architectures: {supported}"
        ) from exc
    return model_cls(vocab_size, num_labels, config)
