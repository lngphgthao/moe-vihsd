"""MoE head theo example; giữ phép tính expert/top-k hiện hành."""
from typing import Any
import torch
from torch import nn
from src.models.heads.router import TopKRouter

class MoEExpert(nn.Module):
    """Expert MLP hai layer: Linear -> activation -> dropout -> Linear."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation_name = activation.lower()

        if self.activation_name == "gelu":
            act_module = nn.GELU()
        elif self.activation_name == "relu":
            act_module = nn.ReLU()
        elif self.activation_name == "silu":
            act_module = nn.SiLU()
        else:
            raise ValueError(
                f"Unsupported activation '{activation}'. Supported activations: 'gelu', 'relu', 'silu'."
            )

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act_module,
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MoEHead(nn.Module):
    """Routing head của Classifier-MoE.

    1. Router nhận representation h và tính routing score của expert.
    2. Dùng top-k routing để chọn k expert cho mỗi example.
    3. Phân phối example tới N expert MLP hai layer độc lập.
    4. Kết hợp output được chọn bằng routing weight đã chuẩn hóa.
    5. Có thể kết hợp thêm shared expert và/hoặc residual connection.
    6. Đưa representation kết hợp vào linear classification layer cuối.
    7. Tính auxiliary load-balancing loss.
    """

    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        num_experts: int = 4,
        top_k: int = 2,
        expert_hidden_dim: int = 512,
        dropout: float = 0.1,
        shared_expert: bool = False,
        use_residual: bool = False,
        activation: str = "gelu",
        classifier_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_labels = int(num_labels)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.expert_hidden_dim = int(expert_hidden_dim)
        self.dropout_rate = float(dropout)
        self.use_shared_expert = bool(shared_expert)
        self.use_residual = bool(use_residual)
        self.activation = str(activation).lower()
        self.classifier_dropout = nn.Dropout(float(classifier_dropout))

        if self.num_experts < 1:
            raise ValueError("num_experts must be at least 1")
        if self.top_k < 1 or self.top_k > self.num_experts:
            raise ValueError(f"top_k ({self.top_k}) must be between 1 and num_experts ({self.num_experts})")

        self.router = TopKRouter(self.input_dim, self.num_experts, self.top_k)
        self.experts = nn.ModuleList(
            [
                MoEExpert(
                    input_dim=self.input_dim,
                    hidden_dim=self.expert_hidden_dim,
                    dropout=self.dropout_rate,
                    activation=self.activation,
                )
                for _ in range(self.num_experts)
            ]
        )

        if self.use_shared_expert:
            self.shared_expert = MoEExpert(
                input_dim=self.input_dim,
                hidden_dim=self.expert_hidden_dim,
                dropout=self.dropout_rate,
                activation=self.activation,
            )
        else:
            self.shared_expert = None

        self.classifier = nn.Linear(self.input_dim, self.num_labels)
        self.last_routing_info: dict[str, Any] = {}

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        """Thực hiện forward pass qua Classifier-MoE head.

        Args:
            h: Pooled representation tensor có shape (batch_size, input_dim).

        Returns:
            logits: Classification logits có shape (batch_size, num_labels).
            aux: Dictionary chứa auxiliary loss và routing metric.
        """
        weights, top_indices, probabilities = self.router(h)

        moe_output = torch.zeros_like(h)
        for expert_id, expert in enumerate(self.experts):
            expert_mask = (top_indices == expert_id)
            if not expert_mask.any():
                continue
            sample_indices, choice_indices = expert_mask.nonzero(as_tuple=True)
            expert_out = expert(h[sample_indices])
            expert_weight = weights[sample_indices, choice_indices].unsqueeze(-1)
            moe_output.index_add_(0, sample_indices, expert_out * expert_weight)

        if self.shared_expert is not None:
            shared_out = self.shared_expert(h)
            moe_output = moe_output + shared_out

        if self.use_residual:
            combined = h + moe_output
        else:
            combined = moe_output

        logits = self.classifier(self.classifier_dropout(combined))
        balance_loss = self.router.load_balance_loss(probabilities, top_indices)

        self.last_routing_info = {
            "balance_loss": balance_loss.detach(),
            "probabilities": probabilities.detach(),
            "top_indices": top_indices.detach(),
        }

        aux = {
            "balance_loss": balance_loss,
            "probabilities": probabilities,
            "top_indices": top_indices,
            "routing_weights": weights,
        }
        return logits, aux
