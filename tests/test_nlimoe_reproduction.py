"""Regression check offline cho entry point tái triển khai NLIMoE."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from datasets import Dataset, DatasetDict
from safetensors.torch import load_file
from torch import nn
from torch.utils.data import DataLoader

import train
from src.data.loader import _load_raw, prepare_data
from src.models.backbone import load_backbone
from src.models.nlimoe import NLIMoEClassifier
from src.utils import load_config


ROOT = Path(__file__).resolve().parents[1]


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.embedding = nn.Embedding(12, 8)
        self.training_flags = []

    def forward(self, input_ids, attention_mask, **kwargs):
        self.training_flags.append(self.training)
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class TestNLIMoEReproduction(unittest.TestCase):
    def test_training_entry_point_evaluates_global_steps_and_restores_best_checkpoint(self):
        cases = [(2, [2, 4, 6]), (4, [4, 6]), (400, [6]), (None, [3, 6])]
        for routing in ["dynamic", "topk"]:
            for frequency, expected_steps in cases:
                with self.subTest(routing=routing, frequency=frequency), tempfile.TemporaryDirectory() as directory:
                    config = load_config(ROOT / f"configs/architectures/vianli_nlimoe_{routing}.yaml")
                    self.assertEqual(config["training"]["eval_frequency"], 400)
                    self.assertEqual(config["model"]["pretrained_model_name"], "xlm-roberta-base")
                    config["training"].update(epochs=2, eval_frequency=frequency)
                    config["paths"] = {
                        "checkpoint_dir": str(Path(directory) / "checkpoints"),
                        "results_dir": str(Path(directory) / "results"),
                    }
                    dataset = Dataset.from_dict({
                        "input_ids": [[1, 2, 0], [1, 3, 4]] * 3,
                        "attention_mask": [[1, 1, 0], [1, 1, 1]] * 3,
                        "labels": [0, 1, 2, 0, 1, 2],
                    }).with_format("torch")
                    loader = DataLoader(dataset, batch_size=2)
                    bundle = SimpleNamespace(
                        tokenizer=SimpleNamespace(vocab_size=12, pad_token_id=0),
                        loaders={"train": loader, "validation": loader, "test": object()},
                        label_names=["entailment", "contradiction", "neutral"],
                        label_mapping={"entailment": 0, "contradiction": 1, "neutral": 2},
                        num_labels=3,
                    )
                    encoder = TinyEncoder()
                    with patch("src.models.nlimoe.load_backbone", return_value=encoder) as backbone:
                        model = NLIMoEClassifier(12, 3, config["model"])
                    backbone.assert_called_once_with(
                        "xlm-roberta-base", revision=config["model"]["pretrained_model_revision"]
                    )
                    self.assertEqual(model.dropout.p, 0.0)
                    self.assertTrue(all(expert[1].p == 0.4 for expert in model.experts))
                    self.assertTrue(all(len(expert) == 3 for expert in model.experts))
                    first_checkpoint = {}
                    validation_calls = []
                    real_evaluate = train.evaluate

                    def evaluate_at_step(model, loader, device, label_names):
                        validation_calls.append(True)
                        if len(validation_calls) == 1:
                            first_checkpoint.update({key: value.detach().clone() for key, value in model.state_dict().items()})
                        metrics = real_evaluate(model, loader, device, label_names)
                        metrics["macro_f1"] = 1.0 / len(validation_calls)
                        return metrics

                    wandb_run = Mock()
                    with contextlib.ExitStack() as stack:
                        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                        stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
                        stack.enter_context(patch("sys.argv", ["train.py", "--run-id", "offline"]))
                        stack.enter_context(patch.dict("os.environ", {
                            "CHECKPOINT_DIR": config["paths"]["checkpoint_dir"],
                            "RESULTS_DIR": config["paths"]["results_dir"],
                        }))
                        stack.enter_context(patch.object(train, "load_config", return_value=config))
                        prepare = stack.enter_context(patch.object(train, "prepare_data", return_value=bundle))
                        stack.enter_context(patch.object(train, "build_model", return_value=model))
                        stack.enter_context(patch.object(train.torch.cuda, "is_available", return_value=False))
                        stack.enter_context(patch.object(train, "maybe_start_wandb", return_value=wandb_run))
                        stack.enter_context(patch.object(train, "evaluate", side_effect=evaluate_at_step))
                        train.main()
                    self.assertEqual(prepare.call_args.args[0]["tokenizer_revision"], config["model"]["pretrained_model_revision"])
                    results = Path(directory) / "results/offline"
                    history = json.loads((results / "training_history.json").read_text())
                    self.assertEqual([record["global_step"] for record in history], expected_steps)
                    summary = json.loads((results / "run_metrics.json").read_text())
                    self.assertEqual(summary["best_global_step"], expected_steps[0])
                    checkpoint_dir = Path(directory) / "checkpoints/offline"
                    checkpoint = load_file(str(checkpoint_dir / "nlimoe_best.safetensors"))
                    for key, expected in first_checkpoint.items():
                        torch.testing.assert_close(checkpoint[key], expected)
                        torch.testing.assert_close(model.state_dict()[key], expected)
                    self.assertEqual(sum(encoder.training_flags), 6)
                    for record in history:
                        self.assertAlmostEqual(
                            record["train_loss"],
                            record["train_classification_loss"] + record["train_weighted_auxiliary_loss"],
                            places=5,
                        )
                    self.assertEqual([call.kwargs["step"] for call in wandb_run.log.call_args_list[:-1]], expected_steps)
                    wandb_run.finish.assert_called_once()
                    if routing == "topk":
                        self.assertNotIn("mean_complexity", summary["routing_diagnostics"]["nlimoe"])
                    else:
                        self.assertIn("mean_complexity", summary["routing_diagnostics"]["nlimoe"])

    def test_invalid_evaluation_frequencies_fail_before_loading(self):
        for frequency in [0, -1, 2.5, "400", True]:
            with self.subTest(frequency=frequency):
                with self.assertRaisesRegex(ValueError, "optimizer steps"):
                    train.apply_training_profile({"training": {"eval_frequency": frequency}}, None)

    def test_official_splits_and_revisions_are_checked(self):
        config = {
            "name": "offline", "revision": "data-sha", "tokenizer": "offline",
            "tokenizer_revision": "model-sha", "input_mode": "nli",
            "premise_column": "premise", "hypothesis_column": "hypothesis",
            "label_column": "label", "label_names": ["a", "b"], "max_length": 4,
            "train_split": "train", "validation_split": "validation", "test_split": "test",
            "expected_split_sizes": {"train": 2, "validation": 2, "test": 2},
            "batch_size": 2, "num_workers": 0,
        }
        with patch("src.data.loader.load_dataset") as download:
            _load_raw(config)
        download.assert_called_once_with("offline", revision="data-sha")
        with patch("src.models.backbone.AutoModel.from_pretrained") as download:
            load_backbone("offline", revision="model-sha")
        download.assert_called_once_with("offline", revision="model-sha")
        examples = {"premise": ["first", "second"], "hypothesis": ["pair", "pair"], "label": ["a", "b"]}
        raw = DatasetDict({name: Dataset.from_dict(examples) for name in ["train", "validation", "test"]})
        tokenizer = Mock(return_value={"input_ids": [[1, 2, 0, 0]] * 2, "attention_mask": [[1, 1, 0, 0]] * 2})
        with patch("src.data.loader._load_raw", return_value=raw):
            with patch("src.data.loader.AutoTokenizer.from_pretrained", return_value=tokenizer) as download:
                bundle = prepare_data(config)
        download.assert_called_once_with("offline", revision="model-sha")
        self.assertEqual(len(bundle.loaders["train"].dataset), 2)
        missing_dev = DatasetDict({"train": raw["train"], "test": raw["test"]})
        with patch("src.data.loader._load_raw", return_value=missing_dev):
            with self.assertRaisesRegex(ValueError, "official split"):
                prepare_data(config)
        raw["train"] = Dataset.from_dict({**examples, "premise": [None, "second"]})
        with patch("src.data.loader._load_raw", return_value=raw):
            with self.assertRaisesRegex(ValueError, "after filtering, got 1"):
                prepare_data(config)

    def test_dynamic_fallback_losses_and_complexity_gradient(self):
        config = load_config(ROOT / "configs/architectures/vianli_nlimoe_dynamic.yaml")["model"]
        with patch("src.models.nlimoe.load_backbone", return_value=TinyEncoder()):
            model = NLIMoEClassifier(12, 3, config)
        model.eval()
        with torch.no_grad():
            model.router.weight.zero_()
            model.router.bias.zero_()
            model.complexity_gate.weight.zero_()
            model.complexity_gate.bias.zero_()
        weights, selected, _, _ = model._route_dynamic(torch.zeros(2, 8))
        self.assertEqual(selected.sum(-1).tolist(), [1, 1])
        torch.testing.assert_close(weights.sum(-1), torch.ones(2))
        with torch.no_grad():
            model.router.bias.copy_(torch.tensor([1.2, 0.9, 0.5, 0.0, -0.4, -0.8, -1.2]))
        logits, aux = model(torch.tensor([[1, 2], [3, 4]]), torch.ones(2, 2, dtype=torch.long))
        info = model.last_routing_info
        probabilities = info["probabilities"]
        torch.testing.assert_close(aux["dynamic_loss"], -(probabilities * probabilities.log()).mean())
        torch.testing.assert_close(aux["balance_loss"], 7 * (info["selected"].float().mean(0) * probabilities.mean(0)).sum())
        nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
        gradient = model.complexity_gate.bias.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0)
