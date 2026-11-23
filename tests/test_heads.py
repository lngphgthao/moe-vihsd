"""Kiểm tra offline head chuẩn và MoE trên encoder nhỏ."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file
from transformers import (BertConfig, BertModel, BertForSequenceClassification,
                          RobertaConfig, RobertaModel, RobertaForSequenceClassification,
                          XLMRobertaConfig, XLMRobertaModel, XLMRobertaForSequenceClassification,
                          DebertaV2Config, DebertaV2ForSequenceClassification)

from src.models.factory import build_model


class TestHeads(unittest.TestCase):
    def test_original_matches_native_head_and_checkpoint(self):
        for config_cls, model_cls in [(BertConfig, BertForSequenceClassification),
                                      (RobertaConfig, RobertaForSequenceClassification),
                                      (XLMRobertaConfig, XLMRobertaForSequenceClassification),
                                      (DebertaV2Config, DebertaV2ForSequenceClassification)]:
            with self.subTest(backbone=config_cls.__name__):
                cfg = config_cls(vocab_size=30, hidden_size=8, intermediate_size=12,
                                 num_hidden_layers=1, num_attention_heads=2, num_labels=3,
                                 pad_token_id=0, max_position_embeddings=32)
                if config_cls is DebertaV2Config:
                    cfg.relative_attention = True
                    cfg.position_buckets = 8
                    cfg.max_relative_positions = 32
                    cfg.pos_att_type = ["p2c", "c2p"]
                    cfg.share_att_key = True
                    cfg.position_biased_input = False
                    cfg.norm_rel_ebd = "layer_norm"
                    cfg.type_vocab_size = 0
                native = model_cls(cfg).eval()
                config = {"backbone": "offline", "revision": None, "head": {"type": "original"}}
                with patch("src.models.heads.original.AutoModelForSequenceClassification.from_pretrained",
                           return_value=copy.deepcopy(native)):
                    model = build_model(config, num_labels=3).eval()
                inputs = {"input_ids": torch.tensor([[2, 3, 4, 0], [2, 5, 0, 0]]),
                          "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])}
                if config_cls in (BertConfig, DebertaV2Config):
                    inputs["token_type_ids"] = torch.tensor([[0, 0, 1, 0], [0, 1, 0, 0]])
                logits, aux = model(**inputs)
                torch.testing.assert_close(logits, native(**inputs).logits)
                self.assertEqual(aux, {})
                logits.sum().backward()
                self.assertTrue(any(p.grad is not None for p in model.parameters()))
                with tempfile.TemporaryDirectory() as tmp:
                    checkpoint = str(Path(tmp) / "best.safetensors")
                    save_file(model.state_dict(), checkpoint)
                    restored = build_model(config, num_labels=3, hf_config=native.config.to_dict()).eval()
                    restored.load_state_dict(load_file(checkpoint))
                    torch.testing.assert_close(restored(**inputs)[0], logits)

    def test_moe_masked_pooling_backward_and_frozen_encoder(self):
        for config_cls, encoder_cls in [(BertConfig, BertModel), (RobertaConfig, RobertaModel),
                                        (XLMRobertaConfig, XLMRobertaModel)]:
            with self.subTest(backbone=config_cls.__name__):
                encoder = encoder_cls(config_cls(vocab_size=30, hidden_size=8, intermediate_size=12,
                                      num_hidden_layers=1, num_attention_heads=2, pad_token_id=0,
                                      max_position_embeddings=32))
                config = {"backbone": "offline", "freeze_backbone": True,
                          "head": {"type": "moe", "pooling": "mean", "num_experts": 3,
                                   "expert_hidden_dim": 12, "dropout": 0.0,
                                   "router": {"scoring": "linear", "selection": "top_k", "top_k": 2}}}
                with patch("src.models.classifier.load_backbone", return_value=encoder):
                    model = build_model(config, num_labels=3).eval()
                ids = torch.tensor([[2, 3, 4, 0], [2, 5, 0, 0]])
                mask = ids.ne(0).long()
                logits, aux = model(ids, mask)
                padded_logits, _ = model(torch.nn.functional.pad(ids, (0, 2)),
                                         torch.nn.functional.pad(mask, (0, 2)))
                torch.testing.assert_close(logits, padded_logits)
                self.assertEqual(tuple(logits.shape), (2, 3))
                torch.testing.assert_close(aux["routing_weights"].sum(-1), torch.ones(2))
                (logits.square().mean() + .01 * aux["balance_loss"]).backward()
                self.assertIsNotNone(model.head.router.projection.weight.grad)
                self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))

    def test_moe_preserves_weighted_expert_equation(self):
        from src.models.heads.moe import MoEHead
        for shared, residual in [(False, False), (True, True)]:
            head = MoEHead(input_dim=4, num_labels=3, num_experts=3, top_k=2,
                           expert_hidden_dim=6, dropout=0, shared_expert=shared,
                           use_residual=residual)
            h = torch.randn(5, 4)
            logits, aux = head(h)
            weights, indices, probs = head.router(h)
            expected = torch.zeros_like(h)
            for row in range(len(h)):
                for choice in range(2):
                    expected[row] = expected[row] + weights[row, choice] * head.experts[indices[row, choice]](h[row])
            if shared:
                expected = expected + head.shared_expert(h)
            if residual:
                expected = expected + h
            torch.testing.assert_close(logits, head.classifier(expected))
            torch.testing.assert_close(aux["balance_loss"], head.router.load_balance_loss(probs, indices))

    def test_invalid_model_config_fails_before_download(self):
        for config in [{"architecture": "nlimoe"}, {"backbone": "offline", "head": {"type": "lstm"}}]:
            with self.assertRaises(ValueError):
                build_model(config, num_labels=3)


if __name__ == "__main__":
    unittest.main()
