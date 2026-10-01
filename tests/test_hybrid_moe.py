"""Tests for the Hybrid-MoE composition and routing contracts."""

import unittest

import torch
import torch.nn as nn

from src.models.factory import build_model


class DummyConfig:
    hidden_size = 16


class DummyOutput:
    def __init__(self, hidden_states):
        self.last_hidden_state = hidden_states


class DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = DummyConfig()
        self.embeddings = nn.Embedding(100, 16)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList()

    def forward(self, input_ids, attention_mask=None):
        return DummyOutput(self.embeddings(input_ids))


class DummyMoELayer:
    last_routing_info = {}


class FakeTransformerMoE(nn.Module):
    def __init__(self, vocab_size, num_labels, config):
        super().__init__()
        self.roberta = DummyBackbone()
        self.moe_layers = {str(index): DummyMoELayer() for index in range(6, 12)}
        self.moe_layer_indices = list(range(6, 12))
        self.pooling = config.get("pooling", "cls")
        self.num_experts = 6
        self.top_k = 1
        self.routing_method = "dynamic_threshold"


class TestHybridMoE(unittest.TestCase):
    def test_factory_registers_hybrid(self):
        import src.models.hybrid_moe as hybrid_module

        original = hybrid_module.TransformerMoEClassifier
        hybrid_module.TransformerMoEClassifier = FakeTransformerMoE
        try:
            model = build_model(
                {
                    "architecture": "hybrid_moe",
                    "pretrained_model_name": "dummy",
                    "classifier_moe": {"num_experts": 4, "top_k": 2},
                },
                vocab_size=100,
                num_labels=3,
            )
            self.assertEqual(model.num_transformer_experts, 6)
            self.assertEqual(model.num_classifier_experts, 4)
            self.assertEqual(model.classifier_top_k, 2)
        finally:
            hybrid_module.TransformerMoEClassifier = original

    def test_forward_returns_both_routing_branches(self):
        import src.models.hybrid_moe as hybrid_module

        original = hybrid_module.TransformerMoEClassifier
        hybrid_module.TransformerMoEClassifier = FakeTransformerMoE
        try:
            model = build_model(
                {
                    "architecture": "hybrid_moe",
                    "pretrained_model_name": "dummy",
                    "classifier_moe": {"num_experts": 4, "top_k": 2},
                },
                vocab_size=100,
                num_labels=3,
            )
            logits, aux = model(
                torch.randint(0, 100, (4, 8)),
                torch.ones(4, 8, dtype=torch.long),
            )
            self.assertEqual(logits.shape, (4, 3))
            self.assertIn("classifier_moe", aux)
            self.assertIn("layer_routing", aux)
            self.assertIn("balance_loss", aux)
        finally:
            hybrid_module.TransformerMoEClassifier = original

    def test_dynamic_head_variant(self):
        import src.models.hybrid_moe as hybrid_module

        original = hybrid_module.TransformerMoEClassifier
        hybrid_module.TransformerMoEClassifier = FakeTransformerMoE
        try:
            model = build_model(
                {
                    "architecture": "hybrid_moe",
                    "pretrained_model_name": "dummy",
                    "hybrid_head_type": "dynamic_moe",
                    "dynamic_moe": {
                        "num_experts": 4,
                        "routing_type": "top_k",
                        "top_k": 2,
                    },
                },
                vocab_size=100,
                num_labels=3,
            )
            self.assertTrue(hasattr(model, "dynamic_moe"))
            self.assertEqual(model.num_classifier_experts, 4)
            self.assertEqual(model.classifier_top_k, 2)
        finally:
            hybrid_module.TransformerMoEClassifier = original


if __name__ == "__main__":
    unittest.main()
