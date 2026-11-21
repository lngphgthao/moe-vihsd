"""Ghép dense encoder với MoE head."""
from torch import nn
from src.models.backbone import load_backbone
from src.models.heads.moe import MoEHead


class HeadClassifier(nn.Module):
    def __init__(self, config, num_labels, hf_config=None):
        super().__init__()
        self.backbone = load_backbone(config["backbone"], revision=config.get("revision"), hf_config=hf_config)
        if config.get("freeze_backbone", False):
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
        head = config["head"]
        self.pooling = head.get("pooling", "cls")
        if self.pooling not in {"cls", "mean"}:
            raise ValueError("head.pooling must be cls or mean")
        router = head.get("router", {})
        if router.get("scoring", "linear") != "linear" or router.get("selection", "top_k") != "top_k":
            raise ValueError("Only linear scoring with top_k selection is implemented")
        self.head = MoEHead(
            input_dim=int(self.backbone.config.hidden_size), num_labels=num_labels,
            num_experts=int(head.get("num_experts", 4)), top_k=int(router.get("top_k", 2)),
            expert_hidden_dim=int(head.get("expert_hidden_dim", 512)),
            dropout=float(head.get("dropout", .1)), activation=head.get("activation", "gelu"),
            shared_expert=bool(head.get("shared_expert", False)),
            use_residual=bool(head.get("use_residual", False)),
            classifier_dropout=float(head.get("classifier_dropout", 0.0)))

    @property
    def encoder(self):
        return self.backbone

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            inputs["token_type_ids"] = token_type_ids
        hidden = self.backbone(**inputs).last_hidden_state
        if self.pooling == "cls":
            pooled = hidden[:, 0]
        else:
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        return self.head(pooled)
