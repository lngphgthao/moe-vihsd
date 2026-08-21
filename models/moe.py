"""Full Transformer encoder with a top-k Mixture of Experts block."""

import math

import torch
import torch.nn as nn

from models.expert import Expert
from models.router import TopKRouter


class SparseMoE(nn.Module):
    """Dispatch each token to selected experts and combine their outputs."""

    def __init__(self, model_dim: int, expert_dim: int, num_experts: int, top_k: int = 1, dropout: float = 0.1) -> None:
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
            output.index_add_(0, token_positions, expert_output * weights[token_positions, choice_positions].unsqueeze(-1))
        return output.reshape(batch_size, sequence_length, model_dim), {
            "top_indices": indices,
            "probabilities": probabilities,
            "balance_loss": self.router.load_balance_loss(probabilities, indices),
        }


class ViHSDMoEClassifier(nn.Module):
    """Compact text classifier with trainable embeddings, attention, router, and experts."""

    def __init__(self, vocab_size: int, num_labels: int, config: dict) -> None:
        super().__init__()
        model_dim = int(config["model_dim"])
        self.num_experts = int(config["num_experts"])
        self.embedding = nn.Embedding(vocab_size, model_dim, padding_idx=int(config["pad_token_id"]))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=int(config["num_attention_heads"]),
            dim_feedforward=int(config["model_dim"]) * 4,
            dropout=float(config["dropout"]),
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(config["num_layers"]))
        self.moe = SparseMoE(
            model_dim=model_dim,
            expert_dim=int(config["expert_hidden_dim"]),
            num_experts=self.num_experts,
            top_k=int(config["top_k"]),
            dropout=float(config["dropout"]),
        )
        self.norm = nn.LayerNorm(model_dim)
        self.classifier = nn.Linear(model_dim, num_labels)

    def forward(self, input_ids, attention_mask):
        padding_mask = attention_mask == 0
        hidden = self.embedding(input_ids) * math.sqrt(self.embedding.embedding_dim)
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        moe_input = self.norm(pooled).unsqueeze(1)
        moe_output, routing = self.moe(moe_input)
        logits = self.classifier(self.norm(pooled + moe_output.squeeze(1)))
        return logits, routing
