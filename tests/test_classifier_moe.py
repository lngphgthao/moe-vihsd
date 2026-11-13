"""Unit test và smoke test cho module và architecture Classifier-MoE."""

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.classifier_moe import (
    ClassifierMoEExpert,
    ClassifierMoEHead,
    ClassifierMoEClassifier,
)
from src.models.factory import build_model, standardize_model_output
from src.utils.config import set_seed


class DummyConfig:
    def __init__(self, hidden_size: int = 64):
        self.hidden_size = hidden_size


class DummyEncoderOutput:
    def __init__(self, last_hidden_state: torch.Tensor):
        self.last_hidden_state = last_hidden_state


class DummyBackbone(nn.Module):
    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.config = DummyConfig(hidden_size=hidden_size)
        self.embedding = nn.Embedding(100, hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        h = self.embedding(input_ids)
        return DummyEncoderOutput(last_hidden_state=h)


class TestClassifierMoE(unittest.TestCase):
    """Test shape tensor, routing, tính loss và forward/backward pass."""

    def setUp(self):
        set_seed(42)
        self.batch_size = 4
        self.input_dim = 64
        self.num_labels = 3
        self.num_experts = 4
        self.top_k = 2
        self.expert_hidden_dim = 128
        self.dropout = 0.1
        self.load_balance_loss_coef = 0.01

    def test_expert_architecture(self):
        """Kiểm tra ClassifierMoEExpert là MLP hai layer: Linear -> activation -> dropout -> Linear."""
        expert = ClassifierMoEExpert(
            input_dim=self.input_dim,
            hidden_dim=self.expert_hidden_dim,
            dropout=self.dropout,
            activation="gelu",
        )
        # Kiểm tra cấu trúc network.
        self.assertEqual(len(expert.net), 4)
        self.assertIsInstance(expert.net[0], nn.Linear)
        self.assertEqual(expert.net[0].in_features, self.input_dim)
        self.assertEqual(expert.net[0].out_features, self.expert_hidden_dim)

        self.assertIsInstance(expert.net[1], nn.GELU)

        self.assertIsInstance(expert.net[2], nn.Dropout)
        self.assertAlmostEqual(expert.net[2].p, self.dropout)

        self.assertIsInstance(expert.net[3], nn.Linear)
        self.assertEqual(expert.net[3].in_features, self.expert_hidden_dim)
        self.assertEqual(expert.net[3].out_features, self.input_dim)

        # Kiểm tra shape của forward pass.
        x = torch.randn(self.batch_size, self.input_dim)
        out = expert(x)
        self.assertEqual(out.shape, (self.batch_size, self.input_dim))

    def test_head_tensor_shapes_and_routing(self):
        """Kiểm tra shape output, top-k routing weight chuẩn hóa và auxiliary loss."""
        head = ClassifierMoEHead(
            input_dim=self.input_dim,
            num_labels=self.num_labels,
            num_experts=self.num_experts,
            top_k=self.top_k,
            expert_hidden_dim=self.expert_hidden_dim,
            dropout=self.dropout,
            load_balance_loss_coef=self.load_balance_loss_coef,
            shared_expert=False,
            use_residual=False,
        )

        h = torch.randn(self.batch_size, self.input_dim)
        logits, aux = head(h)

        # Kiểm tra shape.
        self.assertEqual(logits.shape, (self.batch_size, self.num_labels))
        self.assertIn("balance_loss", aux)
        self.assertIn("top_indices", aux)
        self.assertIn("routing_weights", aux)
        self.assertIn("probabilities", aux)

        top_indices = aux["top_indices"]
        weights = aux["routing_weights"]
        probabilities = aux["probabilities"]

        self.assertEqual(top_indices.shape, (self.batch_size, self.top_k))
        self.assertEqual(weights.shape, (self.batch_size, self.top_k))
        self.assertEqual(probabilities.shape, (self.batch_size, self.num_experts))

        # Kiểm tra expert index hợp lệ.
        self.assertTrue((top_indices >= 0).all())
        self.assertTrue((top_indices < self.num_experts).all())

        # Kiểm tra weight dương và tổng trên top-k bằng 1 cho mỗi example.
        weights_sum = weights.sum(dim=-1)
        torch.testing.assert_close(weights_sum, torch.ones_like(weights_sum), rtol=1e-5, atol=1e-5)

        # Kiểm tra balance loss là scalar không âm.
        balance_loss = aux["balance_loss"]
        self.assertEqual(balance_loss.dim(), 0)
        self.assertTrue(balance_loss.item() >= 0.0)

    def test_head_forward_backward(self):
        """Kiểm tra gradient forward và backward đi qua expert, router và classifier."""
        head = ClassifierMoEHead(
            input_dim=self.input_dim,
            num_labels=self.num_labels,
            num_experts=self.num_experts,
            top_k=self.top_k,
            expert_hidden_dim=self.expert_hidden_dim,
            dropout=self.dropout,
            load_balance_loss_coef=self.load_balance_loss_coef,
            shared_expert=False,
            use_residual=False,
        )

        h = torch.randn(self.batch_size, self.input_dim, requires_grad=True)
        targets = torch.tensor([0, 1, 2, 0], dtype=torch.long)

        logits, aux = head(h)
        task_loss = F.cross_entropy(logits, targets)
        total_loss = task_loss + self.load_balance_loss_coef * aux["balance_loss"]

        total_loss.backward()

        # Input phải có gradient.
        self.assertIsNotNone(h.grad)
        self.assertFalse(torch.isnan(h.grad).any())

        # Router projection phải có gradient.
        self.assertIsNotNone(head.router.projection.weight.grad)
        self.assertFalse(torch.isnan(head.router.projection.weight.grad).any())

        # Classifier cuối phải có gradient.
        self.assertIsNotNone(head.classifier.weight.grad)
        self.assertFalse(torch.isnan(head.classifier.weight.grad).any())

        # Ít nhất các expert được chọn phải nhận gradient.
        expert_has_grad = any(
            expert.net[0].weight.grad is not None and not torch.isnan(expert.net[0].weight.grad).any()
            for expert in head.experts
        )
        self.assertTrue(expert_has_grad)

    def test_configurable_options(self):
        """Kiểm tra parameter shared_expert, use_residual và routing."""
        # 1. shared_expert=True.
        head_shared = ClassifierMoEHead(
            input_dim=self.input_dim,
            num_labels=self.num_labels,
            num_experts=self.num_experts,
            top_k=self.top_k,
            expert_hidden_dim=self.expert_hidden_dim,
            shared_expert=True,
            use_residual=False,
        )
        self.assertIsNotNone(head_shared.shared_expert)
        h = torch.randn(self.batch_size, self.input_dim)
        logits, aux = head_shared(h)
        self.assertEqual(logits.shape, (self.batch_size, self.num_labels))
        logits.sum().backward()
        self.assertIsNotNone(head_shared.shared_expert.net[0].weight.grad)

        # 2. use_residual=True.
        head_res = ClassifierMoEHead(
            input_dim=self.input_dim,
            num_labels=self.num_labels,
            num_experts=self.num_experts,
            top_k=self.top_k,
            expert_hidden_dim=self.expert_hidden_dim,
            shared_expert=False,
            use_residual=True,
        )
        logits, aux = head_res(h)
        self.assertEqual(logits.shape, (self.batch_size, self.num_labels))

        # 3. top_k = 1.
        head_top1 = ClassifierMoEHead(
            input_dim=self.input_dim,
            num_labels=self.num_labels,
            num_experts=self.num_experts,
            top_k=1,
        )
        logits, aux = head_top1(h)
        self.assertEqual(aux["top_indices"].shape, (self.batch_size, 1))

        # 4. top_k > num_experts không hợp lệ và phải raise ValueError.
        with self.assertRaises(ValueError):
            ClassifierMoEHead(
                input_dim=self.input_dim,
                num_labels=self.num_labels,
                num_experts=2,
                top_k=3,
            )

        # 5. num_experts < 1 không hợp lệ và phải raise ValueError.
        with self.assertRaises(ValueError):
            ClassifierMoEHead(
                input_dim=self.input_dim,
                num_labels=self.num_labels,
                num_experts=0,
                top_k=1,
            )

    def test_full_model_with_dummy_backbone(self):
        """Smoke test forward, pooling và backward step của ClassifierMoEClassifier."""
        config = {
            "architecture": "classifier_moe",
            "pretrained_model_name": "dummy",
            "pooling": "cls",
            "num_experts": 4,
            "top_k": 2,
            "expert_hidden_dim": 64,
            "dropout": 0.1,
            "load_balance_loss_coef": 0.01,
            "shared_expert": False,
            "use_residual": False,
        }

        # Monkey-patch load_backbone để trả về DummyBackbone.
        import src.models.classifier_moe as cm_module
        orig_load = cm_module.load_backbone
        try:
            cm_module.load_backbone = lambda name: DummyBackbone(hidden_size=64)

            # Kiểm tra CLS pooling.
            model = ClassifierMoEClassifier(vocab_size=100, num_labels=3, config=config)
            input_ids = torch.randint(0, 100, (4, 16))
            attention_mask = torch.ones(4, 16, dtype=torch.long)

            logits, aux = standardize_model_output(model(input_ids, attention_mask))
            self.assertEqual(logits.shape, (4, 3))
            self.assertIn("balance_loss", aux)

            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            loss = F.cross_entropy(logits, torch.tensor([0, 1, 2, 1])) + 0.01 * aux["balance_loss"]
            loss.backward()
            optimizer.step()

            # Kiểm tra mean pooling.
            config_mean = {**config, "pooling": "mean"}
            model_mean = ClassifierMoEClassifier(vocab_size=100, num_labels=3, config=config_mean)
            logits_mean, aux_mean = standardize_model_output(model_mean(input_ids, attention_mask))
            self.assertEqual(logits_mean.shape, (4, 3))

            # Kiểm tra factory build_model.
            from src.models.factory import build_model
            factory_model = build_model(config, vocab_size=100, num_labels=3)
            self.assertIsInstance(factory_model, ClassifierMoEClassifier)
            self.assertTrue(hasattr(factory_model, "classifier_moe"))
            self.assertFalse(hasattr(factory_model, "moe_layers"))

        finally:
            cm_module.load_backbone = orig_load

    def test_reproducibility(self):
        """Kiểm tra seed giống nhau tạo initialization và output giống nhau."""
        set_seed(123)
        head1 = ClassifierMoEHead(
            input_dim=self.input_dim,
            num_labels=self.num_labels,
            num_experts=self.num_experts,
            top_k=self.top_k,
            dropout=0.0,
        )
        x = torch.randn(self.batch_size, self.input_dim)
        logits1, _ = head1(x)

        set_seed(123)
        head2 = ClassifierMoEHead(
            input_dim=self.input_dim,
            num_labels=self.num_labels,
            num_experts=self.num_experts,
            top_k=self.top_k,
            dropout=0.0,
        )
        logits2, _ = head2(x)

        torch.testing.assert_close(logits1, logits2)


if __name__ == "__main__":
    unittest.main()

