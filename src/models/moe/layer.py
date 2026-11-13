"""Transformer Mixture of Experts token-level cho BERT-family backbone."""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.backbone import get_encoder_layers, load_moe_backbone
from src.models.moe.ffn import Expert
from src.models.moe.router import DynamicThresholdRouter, TopKRouter


class TransformerMoELayer(nn.Module):
    """Thay FFN của BERT-family layer bằng token-level MoE block."""

    def __init__(
        self,
        original_layer: nn.Module,
        num_experts: int = 4,
        top_k: int = 1,
        dropout: float = 0.1,
        upcycle: bool = True,
        expert_type: str = "gelu",
        expert_hidden_size: int | None = None,
        expert_init_noise: float = 0.0,
        learnable_residual_scale: bool = False,
        residual_scale_init: float = 1.0,
        use_shared_expert: bool = False,
        routing_method: str = "top_k",
        dynamic_static_threshold: float = 0.1,
        dynamic_threshold_scale: float = 0.1,
        dynamic_threshold_temperature: float = 0.02,
        expert_inner_dropout: float | None = None,
    ) -> None:
        super().__init__()
        # Giữ nguyên self-attention module ban đầu.
        self.attention = original_layer.attention

        hidden_size = original_layer.intermediate.dense.in_features
        intermediate_size = original_layer.intermediate.dense.out_features
        expert_hidden_size = expert_hidden_size or intermediate_size
        inner_dropout = dropout if expert_inner_dropout is None else float(expert_inner_dropout)
        if not 0.0 <= inner_dropout <= 1.0:
            raise ValueError("expert_inner_dropout must be within [0, 1] or null")
        if upcycle and expert_type != "gelu":
            raise ValueError("upcycle=true is only supported with expert_type='gelu'")
        if expert_init_noise < 0:
            raise ValueError("expert_init_noise must be non-negative")

        self.num_experts = int(num_experts)
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.expert_type = expert_type
        self.expert_hidden_size = expert_hidden_size
        self.use_shared_expert = bool(use_shared_expert)
        self.routing_method = str(routing_method).lower()
        if self.routing_method not in {"top_k", "dynamic_threshold"}:
            raise ValueError("routing_method must be 'top_k' or 'dynamic_threshold'")

        self.router = None
        self.experts = nn.ModuleList()
        if self.num_experts > 0:
            if self.routing_method == "dynamic_threshold":
                self.router = DynamicThresholdRouter(
                    hidden_size, self.num_experts, dynamic_static_threshold,
                    dynamic_threshold_scale, dynamic_threshold_temperature,
                )
            else:
                self.router = TopKRouter(hidden_size, self.num_experts, top_k)
            self.experts = nn.ModuleList(
                [Expert(hidden_size, expert_hidden_size, inner_dropout, expert_type) for _ in range(self.num_experts)]
            )

            if upcycle:
                # MoE Upcycling: clone weight pretrained của intermediate và output projection
                # vào toàn bộ expert.
                for expert in self.experts:
                    expert.upcycle_from_ffn(
                        original_layer.intermediate.dense,
                        original_layer.output.dense,
                    )

            if expert_init_noise > 0:
                with torch.no_grad():
                    for expert in self.experts:
                        for parameter in expert.parameters():
                            if parameter.dim() > 1:
                                parameter.add_(expert_init_noise * torch.randn_like(parameter))

        if self.use_shared_expert:
            self.shared_expert = Expert(hidden_size, expert_hidden_size, inner_dropout, "gelu")
            if upcycle:
                self.shared_expert.upcycle_from_ffn(
                    original_layer.intermediate.dense,
                    original_layer.output.dense,
                )
        else:
            self.shared_expert = None

        # Giữ nguyên output dropout và LayerNorm ban đầu.
        self.dropout = nn.Dropout(float(dropout))
        self.LayerNorm = copy.deepcopy(original_layer.output.LayerNorm)
        if learnable_residual_scale:
            self.moe_residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        else:
            self.register_buffer(
                "moe_residual_scale",
                torch.tensor(float(residual_scale_init)),
            )

        # Cache tên parameter mà attention.forward nhận để tương thích nhiều version.
        import inspect
        self._attn_params = set(inspect.signature(self.attention.forward).parameters.keys())

        # Lưu routing metric ở cấp layer từ forward pass.
        self.last_routing_info: dict[str, Any] = {}

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        past_key_value: tuple[tuple[torch.Tensor]] | None = None,
        output_attentions: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        # 1. Self-attention sub-layer, giữ contract RobertaAttention của HuggingFace.
        attn_kwargs = dict(kwargs)
        if head_mask is not None:
            attn_kwargs["head_mask"] = head_mask
        if encoder_hidden_states is not None:
            attn_kwargs["encoder_hidden_states"] = encoder_hidden_states
        if encoder_attention_mask is not None:
            attn_kwargs["encoder_attention_mask"] = encoder_attention_mask
        if output_attentions:
            attn_kwargs["output_attentions"] = output_attentions
        if past_key_value is not None:
            attn_kwargs["past_key_value"] = past_key_value

        filtered_kwargs = {k: v for k, v in attn_kwargs.items() if k in self._attn_params}
        attention_outputs = self.attention(hidden_states, attention_mask, **filtered_kwargs)
        attention_output = attention_outputs[0]

        # 2. Token-level MoE Feed-Forward Network.
        batch_size, seq_len, hidden_size = attention_output.shape
        flat_tokens = attention_output.reshape(-1, hidden_size)

        if self.num_experts > 0:
            dynamic_info = None
            if self.routing_method == "dynamic_threshold":
                weights, selected, probabilities, complexity, threshold = self.router(flat_tokens)
                indices = selected.nonzero(as_tuple=False)[:, 1].unsqueeze(-1)
                dynamic_info = {"selected_mask": selected, "complexity": complexity, "threshold": threshold}
            else:
                weights, indices, probabilities = self.router(flat_tokens)
                selected = None
            moe_output = torch.zeros_like(flat_tokens)

            for expert_id, expert in enumerate(self.experts):
                expert_selected = selected[:, expert_id] if selected is not None else indices == expert_id
                if not expert_selected.any():
                    continue
                if selected is not None:
                    token_positions = expert_selected.nonzero(as_tuple=True)[0]
                    choice_positions = torch.full_like(token_positions, expert_id)
                else:
                    token_positions, choice_positions = expert_selected.nonzero(as_tuple=True)
                expert_out = expert(flat_tokens[token_positions])
                expert_weight = weights[token_positions, choice_positions].unsqueeze(-1)
                moe_output.index_add_(0, token_positions, expert_out * expert_weight)
        else:
            moe_output = torch.zeros_like(flat_tokens)
            weights = None
            indices = None
            selected = None
            dynamic_info = None
            probabilities = torch.zeros((flat_tokens.shape[0], 0), device=flat_tokens.device, dtype=flat_tokens.dtype)

        if self.use_shared_expert:
            shared_out = self.shared_expert(flat_tokens)
            moe_output = moe_output + shared_out

        moe_output = moe_output.reshape(batch_size, seq_len, hidden_size)

        # Residual connection + Dropout + LayerNorm.
        layer_output = self.LayerNorm(
            self.moe_residual_scale * self.dropout(moe_output) + attention_output
        )

        # 3. Tính load-balancing loss và xử lý padding mask.
        valid_mask = None
        if attention_mask is not None:
            if attention_mask.dim() == 4:
                # Shape có thể là (batch, 1, 1, seq_len) hoặc (batch, 1, seq_len, seq_len).
                # [:, 0, 0, :] lấy key có shape (batch, seq_len).
                valid_mask = (attention_mask[:, 0, 0, :] > -1.0).reshape(-1)
            elif attention_mask.dim() == 2:
                valid_mask = (attention_mask == 1).reshape(-1)

            if valid_mask is not None and valid_mask.numel() != flat_tokens.shape[0]:
                valid_mask = None

        if self.router is not None:
            if valid_mask is not None and valid_mask.any():
                valid_probs = probabilities[valid_mask]
                valid_indices = selected[valid_mask] if selected is not None else indices[valid_mask]
                balance_loss = self.router.load_balance_loss(valid_probs, valid_indices)
                if self.routing_method == "dynamic_threshold":
                    dynamic_loss = self.router.dynamic_loss(valid_probs)
                else:
                    dynamic_loss = flat_tokens.new_zeros(())
            else:
                balance_loss = self.router.load_balance_loss(probabilities, selected if selected is not None else indices)
                dynamic_loss = self.router.dynamic_loss(probabilities) if self.routing_method == "dynamic_threshold" else flat_tokens.new_zeros(())
        else:
            balance_loss = torch.tensor(0.0, device=flat_tokens.device, dtype=flat_tokens.dtype)
            dynamic_loss = flat_tokens.new_zeros(())
            dynamic_info = None

        self.last_routing_info = {
            "balance_loss": balance_loss,
            "top_indices": indices,
            "probabilities": probabilities,
            "dynamic_loss": dynamic_loss,
            **(dynamic_info or {}),
        }

        # Transformers version mới truyền trực tiếp output của layer sang layer kế tiếp.
        return layer_output


class PhoBERTClassificationHead(nn.Module):
    """Sequence classification head chuẩn cho RoBERTa/PhoBERT."""

    def __init__(self, hidden_size: int, num_labels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)


class TransformerMoEClassifier(nn.Module):
    """MoE Transformer thay FFN nội bộ được chọn bằng Sparse MoE layer."""

    def __init__(self, vocab_size: int, num_labels: int, config: dict) -> None:
        super().__init__()
        self.model_name = config.get("pretrained_model_name", "vinai/phobert-base")
        self.roberta = load_moe_backbone(self.model_name)
        encoder_layers = get_encoder_layers(self.roberta)
        total_layers = len(encoder_layers)

        num_experts = int(config.get("num_experts", 4))
        top_k = int(config.get("top_k", 1))
        routing_method = str(config.get("routing_method", "top_k")).lower()
        self.num_experts = num_experts
        self.top_k = top_k
        dropout = float(
            config.get("transformer_moe_dropout", config.get("dropout", 0.1))
        )
        upcycle = bool(config.get("upcycle", True))
        use_shared_expert = bool(config.get("shared_expert", False))
        expert_type = str(config.get("expert_type", "gelu")).lower()
        expert_hidden_size = config.get("expert_hidden_size")
        if expert_hidden_size is not None:
            expert_hidden_size = int(expert_hidden_size)
        expert_init_noise = float(config.get("expert_init_noise", 0.0))
        learnable_residual_scale = bool(config.get("learnable_residual_scale", False))
        residual_scale_init = float(config.get("residual_scale_init", 1.0))
        self.pooling = str(config.get("pooling", "cls")).lower()
        if self.pooling not in {"cls", "mean"}:
            raise ValueError("pooling must be either 'cls' or 'mean'")

        # Parse các layer cần chuyển sang MoE.
        moe_layers_cfg = config.get("moe_layers", [8, 9, 10, 11])
        if isinstance(moe_layers_cfg, str):
            cfg_lower = moe_layers_cfg.lower()
            if cfg_lower == "all":
                target_indices = set(range(total_layers))
            elif cfg_lower in {"last_4", "top_4"}:
                target_indices = set(range(total_layers - 4, total_layers))
            elif cfg_lower == "alternating":
                target_indices = set(range(1, total_layers, 2))
            else:
                target_indices = set(range(total_layers - 4, total_layers))
        elif isinstance(moe_layers_cfg, (list, tuple)):
            target_indices = set(int(idx) for idx in moe_layers_cfg)
        else:
            target_indices = set(range(total_layers - 4, total_layers))

        self.moe_layer_indices = sorted(target_indices)
        self.moe_layers: dict[str, TransformerMoELayer] = {}

        for idx in self.moe_layer_indices:
            if not (0 <= idx < total_layers):
                raise ValueError(f"moe_layer index {idx} out of range [0, {total_layers - 1}]")
            orig_layer = encoder_layers[idx]
            moe_layer = TransformerMoELayer(
                original_layer=orig_layer,
                num_experts=num_experts,
                top_k=top_k,
                dropout=dropout,
                expert_inner_dropout=config.get("expert_inner_dropout"),
                upcycle=upcycle,
                expert_type=expert_type,
                expert_hidden_size=expert_hidden_size,
                expert_init_noise=expert_init_noise,
                learnable_residual_scale=learnable_residual_scale,
                residual_scale_init=residual_scale_init,
                use_shared_expert=use_shared_expert,
                routing_method=routing_method,
                dynamic_static_threshold=float(config.get("dynamic_static_threshold", 0.1)),
                dynamic_threshold_scale=float(config.get("dynamic_threshold_scale", 0.1)),
                dynamic_threshold_temperature=float(config.get("dynamic_threshold_temperature", 0.02)),
            )
            encoder_layers[idx] = moe_layer
            self.moe_layers[str(idx)] = moe_layer

        hidden_size = self.roberta.config.hidden_size
        self.classifier = PhoBERTClassificationHead(hidden_size, num_labels, dropout=dropout)

        # Xử lý tùy chọn freeze, mặc định là Full Fine-Tuning.
        self.freeze_attention = bool(config.get("freeze_attention", False))
        self.freeze_embeddings = bool(config.get("freeze_embeddings", False))
        self._apply_freezing()

    def _apply_freezing(self) -> None:
        """Áp dụng freeze parameter theo cấu hình của người dùng."""
        if self.freeze_embeddings:
            for param in self.roberta.embeddings.parameters():
                param.requires_grad = False

        if self.freeze_attention:
            # Freeze self-attention trong toàn bộ layer.
            encoder_layers = get_encoder_layers(self.roberta)
            for layer in encoder_layers:
                for param in layer.attention.parameters():
                    param.requires_grad = False

            # Freeze intermediate và output trong dense layer còn lại, không phải MoE.
            for idx, layer in enumerate(encoder_layers):
                if idx not in self.moe_layer_indices:
                    for param in layer.intermediate.parameters():
                        param.requires_grad = False
                    for param in layer.output.parameters():
                        param.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor | None = None):
        segment_inputs = {} if token_type_ids is None else {"token_type_ids": token_type_ids}
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, **segment_inputs)
        hidden_states = outputs.last_hidden_state

        if self.pooling == "cls":
            pooled = hidden_states[:, 0]
        else:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        logits = self.classifier(pooled)

        # Aggregate auxiliary balance loss trên toàn bộ MoE layer đang active.
        total_balance_loss = torch.tensor(0.0, device=input_ids.device)
        total_dynamic_loss = torch.tensor(0.0, device=input_ids.device)
        layer_routing = {}
        last_top_indices = None
        last_probabilities = None

        for layer_idx_str, moe_layer in self.moe_layers.items():
            info = moe_layer.last_routing_info
            if "balance_loss" in info:
                total_balance_loss = total_balance_loss + info["balance_loss"]
                total_dynamic_loss = total_dynamic_loss + info.get("dynamic_loss", 0.0)
                last_top_indices = info.get("top_indices")
                last_probabilities = info.get("probabilities")
                layer_routing[layer_idx_str] = info

        aux = {
            "balance_loss": total_balance_loss,
            "dynamic_loss": total_dynamic_loss,
            "top_indices": last_top_indices,
            "probabilities": last_probabilities,
            "layer_routing": layer_routing,
        }
        return logits, aux
