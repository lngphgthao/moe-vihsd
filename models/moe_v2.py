"""A stronger MoE variant for the ViHSD classifier while keeping the same workflow."""

import math

import torch
import torch.nn as nn

from models.expert import Expert
from models.router import TopKRouter


class StrongerSparseMoE(nn.Module):
    """Two-stage sparse MoE block with top-k routing and weighted expert aggregation."""

    def __init__(
        self,
        model_dim: int,
        expert_dim: int,
        num_experts: int,
        top_k: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.router = TopKRouter(model_dim, num_experts, top_k)
        self.experts = nn.ModuleList(Expert(model_dim, expert_dim, dropout) for _ in range(num_experts))

    def forward(self, tokens):
        batch_size, sequence_length, model_dim = tokens.shape
        flat_tokens = tokens.reshape(-1, model_dim)
        weights, indices, probabilities = self.router(flat_tokens)
        output = torch.zeros_like(flat_tokens)
        for expert_id, expert in enumerate(self.experts):
            selected_tokens = indices == expert_id
            if not selected_tokens.any():
                continue
            token_positions, choice_positions = selected_tokens.nonzero(as_tuple=True)
            expert_output = expert(flat_tokens[token_positions])
            output.index_add_(
                0,
                token_positions,
                expert_output * weights[token_positions, choice_positions].unsqueeze(-1),
            )
        return output.reshape(batch_size, sequence_length, model_dim), {
            "top_indices": indices,
            "probabilities": probabilities,
            "balance_loss": self.router.load_balance_loss(probabilities, indices),
        }


class StrongerViHSDMoEClassifier(nn.Module):
    """A stronger MoE text classifier with deeper encoder and multi-expert routing."""

    def __init__(self, vocab_size: int, num_labels: int, config: dict) -> None:
        super().__init__()
        model_dim = int(config["model_dim"])
        self.num_experts = int(config.get("num_experts", 8))
        self.embedding = nn.Embedding(vocab_size, model_dim, padding_idx=int(config["pad_token_id"]))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=int(config.get("num_attention_heads", 4)),
            dim_feedforward=int(config.get("encoder_ff_dim", model_dim * 4)),
            dropout=float(config.get("dropout", 0.1)),
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(config.get("num_layers", 4)),
        )

        self.moe_1 = StrongerSparseMoE(
            model_dim=model_dim,
            expert_dim=int(config.get("expert_hidden_dim", 768)),
            num_experts=self.num_experts,
            top_k=int(config.get("top_k", 2)),
            dropout=float(config.get("dropout", 0.1)),
        )
        self.moe_2 = StrongerSparseMoE(
            model_dim=model_dim,
            expert_dim=int(config.get("expert_hidden_dim", 768)),
            num_experts=self.num_experts,
            top_k=int(config.get("top_k", 2)),
            dropout=float(config.get("dropout", 0.1)),
        )

        self.norm = nn.LayerNorm(model_dim)
        self.output_proj = nn.Linear(model_dim, model_dim)
        self.classifier = nn.Linear(model_dim, num_labels)

    def forward(self, input_ids, attention_mask):
        padding_mask = attention_mask == 0
        hidden = self.embedding(input_ids) * math.sqrt(self.embedding.embedding_dim)
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        moe_input = self.norm(pooled).unsqueeze(1)
        moe_output_1, routing_1 = self.moe_1(moe_input)
        moe_output_2, routing_2 = self.moe_2(moe_output_1)

        combined = self.norm(pooled + moe_output_2.squeeze(1))
        logits = self.classifier(self.output_proj(combined))

        aux = {
            "balance_loss": routing_1["balance_loss"] + routing_2["balance_loss"],
            "top_indices": routing_2["top_indices"],
            "probabilities": routing_2["probabilities"],
        }
        return logits, aux
