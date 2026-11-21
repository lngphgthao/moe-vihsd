"""Kiểm tra diagnostics giữ logits/class distribution và routing theo example."""
import unittest
import torch
from src.training.diagnostics import collect_model_diagnostics

class FixedModel(torch.nn.Module):
    def __init__(self, routing):
        super().__init__()
        self.routing = routing
    def forward(self, **inputs):
        logits = torch.tensor([[2.0, 0.0], [1.0, 0.0]])
        aux = {"probabilities": torch.tensor([[.8, .2], [.3, .7]]),
               "top_indices": torch.tensor([[0], [1]])} if self.routing else {}
        return logits, aux

class TestDiagnostics(unittest.TestCase):
    def test_logits_class_distribution_and_example_assignment_counts(self):
        batch = {"input_ids": torch.tensor([[1, 0], [1, 2]]),
                 "attention_mask": torch.tensor([[1, 0], [1, 1]]), "labels": torch.tensor([0, 1])}
        for routing in [False, True]:
            with self.subTest(routing=routing):
                result = collect_model_diagnostics(FixedModel(routing), [batch], "cpu", ["a", "b"])
                self.assertEqual(result["logits_by_true_class"]["a"], [2.0, 0.0])
                self.assertEqual(result["prediction_diagnostics"]["actual_class_distribution"], {"a": 1, "b": 1})
                self.assertEqual(result["prediction_diagnostics"]["predicted_class_distribution"], {"a": 2, "b": 0})
                self.assertAlmostEqual(sum(result["probabilities_by_true_class"]["a"]), 1.0, places=6)
                if routing:
                    self.assertEqual(result["by_class"]["b"]["example_count"], 1)
                    self.assertEqual(result["by_class"]["b"]["assignment_counts"], [0, 1])
                else:
                    self.assertEqual(result["by_class"], {})
