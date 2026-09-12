"""True token-level Mixture of Experts (MoE) Transformer using PhoBERT backbone."""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from models.expert import Expert
from models.router import TopKRouter


class PhoBERTMoELayer(nn.Module):
    """Replaces a standard RoBERTa/PhoBERT layer's FFN with a token-level MoE block."""

    def __init__(
        self,
        original_layer: nn.Module,
        num_experts: int = 4,
        top_k: int = 1,
        dropout: float = 0.1,
        upcycle: bool = True,
    ) -> None:
        super().__init__()
        # Preserve original self-attention module
        self.attention = original_layer.attention

        hidden_size = original_layer.intermediate.dense.in_features
        intermediate_size = original_layer.intermediate.dense.out_features

        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.router = TopKRouter(hidden_size, num_experts, top_k)
        self.experts = nn.ModuleList(
            [Expert(hidden_size, intermediate_size, dropout) for _ in range(num_experts)]
        )

        if upcycle:
            # MoE Upcycling: clone pretrained intermediate and output projection weights to all experts
            with torch.no_grad():
                for expert in self.experts:
                    expert.network[0].weight.copy_(original_layer.intermediate.dense.weight)
                    expert.network[0].bias.copy_(original_layer.intermediate.dense.bias)
                    expert.network[3].weight.copy_(original_layer.output.dense.weight)
                    expert.network[3].bias.copy_(original_layer.output.dense.bias)

        # Preserve original output dropout and LayerNorm
        self.dropout = nn.Dropout(float(dropout))
        self.LayerNorm = copy.deepcopy(original_layer.output.LayerNorm)

        # Cache parameter names accepted by attention.forward for cross-version compatibility
        import inspect
        self._attn_params = set(inspect.signature(self.attention.forward).parameters.keys())

        # Storage for layer-level routing metrics collected during forward pass
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
    ) -> tuple[torch.Tensor, ...]:
        # 1. Self-attention sub-layer (preserves HuggingFace RobertaAttention contract)
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
        extra_outputs = attention_outputs[1:]

        # 2. Token-level MoE Feed-Forward Network
        batch_size, seq_len, hidden_size = attention_output.shape
        flat_tokens = attention_output.reshape(-1, hidden_size)

        weights, indices, probabilities = self.router(flat_tokens)
        moe_output = torch.zeros_like(flat_tokens)

        for expert_id, expert in enumerate(self.experts):
            selected = indices == expert_id
            if not selected.any():
                continue
            token_positions, choice_positions = selected.nonzero(as_tuple=True)
            expert_out = expert(flat_tokens[token_positions])
            expert_weight = weights[token_positions, choice_positions].unsqueeze(-1)
            moe_output.index_add_(0, token_positions, expert_out * expert_weight)

        moe_output = moe_output.reshape(batch_size, seq_len, hidden_size)

        # Residual connection + Dropout + LayerNorm
        layer_output = self.LayerNorm(self.dropout(moe_output) + attention_output)

        # 3. Compute load-balancing loss with padding mask awareness
        valid_mask = None
        if attention_mask is not None:
            if attention_mask.dim() == 4:
                # Shape can be (batch, 1, 1, seq_len) or (batch, 1, seq_len, seq_len)
                # Slicing [:, 0, 0, :] extracts (batch, seq_len) keys
                valid_mask = (attention_mask[:, 0, 0, :] > -1.0).reshape(-1)
            elif attention_mask.dim() == 2:
                valid_mask = (attention_mask == 1).reshape(-1)

            if valid_mask is not None and valid_mask.numel() != flat_tokens.shape[0]:
                valid_mask = None

        if valid_mask is not None and valid_mask.any():
            valid_probs = probabilities[valid_mask]
            valid_indices = indices[valid_mask]
            balance_loss = self.router.load_balance_loss(valid_probs, valid_indices)
        else:
            balance_loss = self.router.load_balance_loss(probabilities, indices)

        self.last_routing_info = {
            "balance_loss": balance_loss,
            "top_indices": indices,
            "probabilities": probabilities,
        }

        return layer_output


class PhoBERTClassificationHead(nn.Module):
    """Standard sequence classification head for RoBERTa/PhoBERT."""

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


class PhoBERTMoEClassifier(nn.Module):
    """True MoE Transformer: PhoBERT with internal FFNs replaced by Sparse MoE layers."""

    def __init__(self, vocab_size: int, num_labels: int, config: dict) -> None:
        super().__init__()
        self.model_name = config.get("pretrained_model_name", "vinai/phobert-base")
        self.roberta = AutoModel.from_pretrained(self.model_name)
        total_layers = len(self.roberta.encoder.layer)

        num_experts = int(config.get("num_experts", 4))
        top_k = int(config.get("top_k", 1))
        self.num_experts = num_experts
        self.top_k = top_k
        dropout = float(config.get("dropout", 0.1))
        upcycle = bool(config.get("upcycle", True))
        self.pooling = str(config.get("pooling", "cls")).lower()
        if self.pooling not in {"cls", "mean"}:
            raise ValueError("pooling must be either 'cls' or 'mean'")

        # Parse which layers to convert to MoE
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
        self.moe_layers: dict[str, PhoBERTMoELayer] = {}

        for idx in self.moe_layer_indices:
            if not (0 <= idx < total_layers):
                raise ValueError(f"moe_layer index {idx} out of range [0, {total_layers - 1}]")
            orig_layer = self.roberta.encoder.layer[idx]
            moe_layer = PhoBERTMoELayer(
                original_layer=orig_layer,
                num_experts=num_experts,
                top_k=top_k,
                dropout=dropout,
                upcycle=upcycle,
            )
            self.roberta.encoder.layer[idx] = moe_layer
            self.moe_layers[str(idx)] = moe_layer

        hidden_size = self.roberta.config.hidden_size
        self.classifier = PhoBERTClassificationHead(hidden_size, num_labels, dropout=dropout)

        # Handle freezing options (Default: Full Fine-Tuning)
        self.freeze_attention = bool(config.get("freeze_attention", False))
        self.freeze_embeddings = bool(config.get("freeze_embeddings", False))
        self._apply_freezing()

    def _apply_freezing(self) -> None:
        """Apply parameter freezing based on user configuration."""
        if self.freeze_embeddings:
            for param in self.roberta.embeddings.parameters():
                param.requires_grad = False

        if self.freeze_attention:
            # Freeze self-attention in all layers
            for layer in self.roberta.encoder.layer:
                for param in layer.attention.parameters():
                    param.requires_grad = False

            # Freeze intermediate and output in any remaining dense (non-MoE) layers
            for idx, layer in enumerate(self.roberta.encoder.layer):
                if idx not in self.moe_layer_indices:
                    for param in layer.intermediate.parameters():
                        param.requires_grad = False
                    for param in layer.output.parameters():
                        param.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        if self.pooling == "cls":
            pooled = hidden_states[:, 0]
        else:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        logits = self.classifier(pooled)

        # Aggregate auxiliary balance loss across all active MoE layers
        total_balance_loss = torch.tensor(0.0, device=input_ids.device)
        layer_routing = {}
        last_top_indices = None
        last_probabilities = None

        for layer_idx_str, moe_layer in self.moe_layers.items():
            info = moe_layer.last_routing_info
            if "balance_loss" in info:
                total_balance_loss = total_balance_loss + info["balance_loss"]
                last_top_indices = info.get("top_indices")
                last_probabilities = info.get("probabilities")
                layer_routing[layer_idx_str] = info

        aux = {
            "balance_loss": total_balance_loss,
            "top_indices": last_top_indices,
            "probabilities": last_probabilities,
            "layer_routing": layer_routing,
        }
        return logits, aux
