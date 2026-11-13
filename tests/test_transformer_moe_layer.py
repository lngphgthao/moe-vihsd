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
