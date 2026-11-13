"""Model factory để đổi architecture variant trong một training workflow."""

from __future__ import annotations

from typing import Any

from src.models.moe.layer import TransformerMoEClassifier
from src.models.dense import DenseTransformerClassifier
from src.models.nlimoe import NLIMoEClassifier
from src.models.classifier_moe import ClassifierMoEClassifier
from src.models.hybrid_moe import HybridMoEClassifier

MODEL_REGISTRY: dict[str, Any] = {
    "transformer_moe": TransformerMoEClassifier,
    # Compatibility alias cho config lịch sử và run đã lưu.
    "phobert_moe": TransformerMoEClassifier,
    "nlimoe": NLIMoEClassifier,
    "dense_transformer": DenseTransformerClassifier,
    # Compatibility alias cho config lịch sử và metadata experiment đã lưu.
    "dense_phobert": DenseTransformerClassifier,
    "classifier_moe": ClassifierMoEClassifier,
    "classifier_moe_phobert": ClassifierMoEClassifier,
    "hybrid_moe": HybridMoEClassifier,
}


def model_inputs_from_batch(batch, device) -> dict:
    """Chuyển các encoder input được hỗ trợ sang device của model."""
    return {
        key: batch[key].to(device)
        for key in ("input_ids", "attention_mask", "token_type_ids")
        if key in batch
    }


def validate_model_config(config: dict) -> str:
    """Kiểm tra thiết lập architecture trước khi load model weights."""
    architecture = str(config.get("architecture", "transformer_moe"))
    if architecture not in MODEL_REGISTRY:
        supported = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Unknown model architecture '{architecture}'. Supported architectures: {supported}"
        )
    return architecture


def standardize_model_output(model_output):
    """Chuẩn hóa model output về contract chung: (logits, aux_dict).

    Helper này giữ contract tuple/dictionary chung giữa các architecture.
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
    """Khởi tạo model từ tên architecture trong config.

    Registry bao gồm các PhoBERT và dense architecture được hỗ trợ.
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

