"""Test tập trung cho token routing trong Transformer-MoE layer."""

import unittest

import torch
import torch.nn as nn

from src.models.moe.layer import TransformerMoELayer


class IdentityAttention(nn.Module):
    def forward(self, hidden_states, attention_mask=None):
        return (hidden_states,)


def make_original_layer(hidden_size: int = 2) -> nn.Module:
    return nn.ModuleDict(
        {
            "attention": IdentityAttention(),
            "intermediate": nn.ModuleDict({"dense": nn.Linear(hidden_size, hidden_size)}),
            "output": nn.ModuleDict(
                {
                    "dense": nn.Linear(hidden_size, hidden_size),
                    "LayerNorm": nn.LayerNorm(hidden_size),
                }
            ),
        }
    )


class TestTransformerMoELayer(unittest.TestCase):
    def test_upcycle_noise_controls_expert_symmetry(self):
        with torch.random.fork_rng():
            torch.manual_seed(42)
            original = make_original_layer()
            for noise in [0.0, 0.001, 0.01]:
                with self.subTest(noise=noise):
                    layer = TransformerMoELayer(original, num_experts=2, upcycle=True, expert_init_noise=noise)
                    first, second = layer.experts
                    for left, right in zip(first.parameters(), second.parameters()):
                        if left.dim() > 1:
                            self.assertEqual(torch.equal(left, right), noise == 0.0)
                        else:
                            torch.testing.assert_close(left, right, rtol=0, atol=0)
                    if noise == 0.0:
                        for expert in layer.experts:
                            torch.testing.assert_close(expert.network[0].weight, original.intermediate.dense.weight, rtol=0, atol=0)
                            torch.testing.assert_close(expert.network[3].weight, original.output.dense.weight, rtol=0, atol=0)

    def test_inner_dropout_keeps_legacy_default_and_eval_output(self):
        original = make_original_layer()
        legacy = TransformerMoELayer(original, num_experts=2, dropout=0.2, use_shared_expert=True)
        inputs = torch.randn(2, 3, 2)
        for option, expected in [(None, 0.2), (0.0, 0.0)]:
            with self.subTest(expert_inner_dropout=option):
                layer = TransformerMoELayer(
                    original, num_experts=2, dropout=0.2, use_shared_expert=True,
                    expert_inner_dropout=option,
                )
                layer.load_state_dict(legacy.state_dict(), strict=True)
                self.assertEqual(layer.dropout.p, 0.2)
                for expert in [*layer.experts, layer.shared_expert]:
                    self.assertEqual(expert.network[2].p, expected)
                legacy.eval()
                layer.eval()
                torch.testing.assert_close(layer(inputs), legacy(inputs), rtol=0, atol=0)
                for expert, reference in zip(layer.experts, legacy.experts):
                    torch.testing.assert_close(expert(inputs), reference(inputs), rtol=0, atol=0)

    def test_invalid_inner_dropout_is_rejected(self):
        for value in [-0.1, 1.1, float("nan")]:
            with self.assertRaises(ValueError):
                TransformerMoELayer(make_original_layer(), expert_inner_dropout=value)

    def test_forward_backward_and_padding_aware_balance_loss(self):
        layer = TransformerMoELayer(
            make_original_layer(),
            num_experts=2,
            top_k=1,
            dropout=0.0,
            upcycle=False,
            expert_hidden_size=2,
        )
        with torch.no_grad():
            layer.router.projection.weight.copy_(torch.eye(2))
            layer.router.projection.bias.zero_()

        hidden_states = torch.tensor(
            [[[3.0, 0.0], [0.0, 3.0], [3.0, 0.0], [3.0, 0.0]]],
            requires_grad=True,
        )
        attention_mask = torch.tensor([[1, 1, 0, 0]])
        outputs = layer(hidden_states, attention_mask)

        self.assertEqual(outputs.shape, hidden_states.shape)
        routing = layer.last_routing_info
        valid_balance_loss = layer.router.load_balance_loss(
            routing["probabilities"][:2], routing["top_indices"][:2]
        )
        all_token_balance_loss = layer.router.load_balance_loss(
            routing["probabilities"], routing["top_indices"]
        )
        torch.testing.assert_close(routing["balance_loss"], valid_balance_loss)
        self.assertFalse(torch.isclose(valid_balance_loss, all_token_balance_loss))

        (outputs.square().mean() + routing["balance_loss"]).backward()
        self.assertIsNotNone(hidden_states.grad)
        self.assertIsNotNone(layer.router.projection.weight.grad)
        self.assertTrue(any(parameter.grad is not None for parameter in layer.experts.parameters()))


if __name__ == "__main__":
    unittest.main()
