"""Integration test offline dùng model random kích thước nhỏ."""

from unittest.mock import patch
import unittest

import torch
from transformers import BertConfig, BertModel, RobertaConfig, RobertaModel, XLMRobertaConfig, XLMRobertaModel

from src.models.factory import build_model, model_inputs_from_batch, standardize_model_output


class TestEncoderIntegration(unittest.TestCase):
    def test_real_encoders_keep_batch_shape_and_receive_segment_ids(self):
        for config_class, encoder_class in [
            (BertConfig, BertModel),
            (RobertaConfig, RobertaModel),
            (XLMRobertaConfig, XLMRobertaModel),
        ]:
            for architecture in ["dense_transformer", "transformer_moe", "classifier_moe", "nlimoe", "hybrid_moe"]:
                with self.subTest(encoder=encoder_class.__name__, architecture=architecture):
                    encoder = encoder_class(config_class(
                        vocab_size=30, hidden_size=8, intermediate_size=16,
                        num_hidden_layers=2, num_attention_heads=2, pad_token_id=0,
                    ))
                    model_config = {
                        "architecture": architecture, "pretrained_model_name": "offline",
                        "moe_layers": [1], "num_experts": 2, "top_k": 1,
                        "dynamic_num_experts": 2, "dynamic_top_k": 1, "routing_type": "top_k",
                        "expert_hidden_dim": 8, "dropout": 0.0,
                        "transformer_moe": {"num_experts": 2, "routing_method": "top_k"},
                        "classifier_moe": {"num_experts": 2, "top_k": 1, "expert_hidden_dim": 8},
                    }
                    with patch("src.models.dense.load_backbone", return_value=encoder), \
                         patch("src.models.classifier_moe.load_backbone", return_value=encoder), \
                         patch("src.models.nlimoe.load_backbone", return_value=encoder), \
                         patch("src.models.moe.layer.load_moe_backbone", return_value=encoder):
                        model = build_model(model_config, 30, 3)
                    batch = {
                        "input_ids": torch.tensor([[1, 2, 3, 0], [1, 4, 5, 0]]),
                        "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]]),
                        "token_type_ids": torch.tensor([[0, 0, 1, 0], [0, 0, 1, 0]]),
                        "labels": torch.tensor([0, 1]),
                    }
                    with patch.object(encoder, "forward", wraps=encoder.forward) as forward:
                        logits, aux = standardize_model_output(model(**model_inputs_from_batch(batch, "cpu")))
                    torch.testing.assert_close(forward.call_args.kwargs["token_type_ids"], batch["token_type_ids"])
                    self.assertEqual(tuple(logits.shape), (2, 3))
                    loss = torch.nn.functional.cross_entropy(logits, batch["labels"]) + 0.01 * aux.get("balance_loss", 0.0)
                    loss.backward()
                    self.assertTrue(torch.isfinite(logits).all())
                    self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))
                    batch.pop("token_type_ids")
                    logits, _ = standardize_model_output(model(**model_inputs_from_batch(batch, "cpu")))
                    self.assertEqual(tuple(logits.shape), (2, 3))
