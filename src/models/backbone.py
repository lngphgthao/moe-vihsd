"""Tải encoder dense; không thay đổi layer nội bộ."""
import torch
from transformers import AutoModel, AutoConfig


def load_backbone(model_name, *, revision=None, hf_config=None):
    if hf_config is not None:
        metadata = dict(hf_config)
        model = AutoModel.from_config(AutoConfig.for_model(metadata.pop("model_type"), **metadata))
        return model.to(dtype=torch.float32)
    model = AutoModel.from_pretrained(model_name, revision=revision, torch_dtype=torch.float32)
    return model.to(dtype=torch.float32)
