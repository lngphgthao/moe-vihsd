"""Tải encoder dense; không thay đổi layer nội bộ."""
from transformers import AutoModel, AutoConfig


def load_backbone(model_name, *, revision=None, hf_config=None):
    if hf_config is not None:
        metadata = dict(hf_config)
        return AutoModel.from_config(AutoConfig.for_model(metadata.pop("model_type"), **metadata))
    return AutoModel.from_pretrained(model_name, revision=revision)
