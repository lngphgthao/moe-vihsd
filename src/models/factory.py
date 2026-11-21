"""Factory cho encoder dense với head original hoặc moe."""
from src.models.classifier import HeadClassifier
from src.models.heads.original import OriginalClassifier


def validate_model_config(config):
    if "architecture" in config or not config.get("backbone"):
        raise ValueError("Use model.backbone and model.head.type; legacy architecture configs are unsupported")
    kind = config.get("head", {}).get("type")
    if kind not in {"original", "moe"}:
        raise ValueError("model.head.type must be original or moe")
    return kind


def build_model(config, num_labels, hf_config=None):
    kind = validate_model_config(config)
    cls = OriginalClassifier if kind == "original" else HeadClassifier
    return cls(config, num_labels, hf_config=hf_config)


def model_inputs_from_batch(batch, device):
    return {key: batch[key].to(device) for key in ("input_ids", "attention_mask", "token_type_ids") if key in batch}


def standardize_model_output(output):
    if not isinstance(output, tuple) or len(output) != 2 or not isinstance(output[1], dict):
        raise ValueError("Model must return (logits, aux_dict)")
    return output
