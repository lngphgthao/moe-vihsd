"""Giữ head chuẩn của model phân lớp Hugging Face."""
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoConfig


class OriginalClassifier(nn.Module):
    def __init__(self, config, num_labels, hf_config=None):
        super().__init__()
        if hf_config is None:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                config["backbone"], revision=config.get("revision"), num_labels=num_labels)
        else:
            metadata = dict(hf_config)
            native_config = AutoConfig.for_model(metadata.pop("model_type"), **metadata)
            self.model = AutoModelForSequenceClassification.from_config(native_config)
        if config.get("freeze_backbone", False):
            for parameter in self.model.base_model.parameters():
                parameter.requires_grad = False

    @property
    def encoder(self):
        return self.model.base_model

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            inputs["token_type_ids"] = token_type_ids
        return self.model(**inputs).logits, {}
