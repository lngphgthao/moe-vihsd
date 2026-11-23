"""Kiểm tra CLI train, artifact, W&B lỗi, checkpoint và final-evaluation offline."""
import copy
import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict
from transformers import BertConfig
import yaml

import train
import evaluate
from src.data.loader import DatasetBundle, _ensure_splits, _label_info, _load_raw
from src.models.factory import build_model as real_build_model
from src.training import engine
from src.utils import load_config
from src.utils.artifacts import RunArtifacts, reserve_run, config_fingerprint, sanitized

ROOT = Path(__file__).resolve().parents[1]


def tiny_bundle():
    data = Dataset.from_dict({"input_ids": [[2, 3, 0], [2, 4, 5]] * 3,
                              "attention_mask": [[1, 1, 0], [1, 1, 1]] * 3,
                              "labels": [0, 1, 2, 0, 1, 2]}).with_format("torch")
    loaders = {name: DataLoader(data, batch_size=2) for name in ("train", "validation", "test")}
    return DatasetBundle(SimpleNamespace(), loaders, ["a", "b", "c"], {"a": 0, "b": 1, "c": 2}, 3,
                         {"splits": {"train": {"fingerprint": "offline"}}})


class TestRunPipeline(unittest.TestCase):
    def setUp(self):
        self.run_dir_patcher = patch.dict(os.environ)
        self.run_dir_patcher.start()
        os.environ.pop("RUNS_DIR", None)

    def tearDown(self):
        self.run_dir_patcher.stop()

    def config(self, directory, kind):
        cfg = load_config(ROOT / f"configs/experiments/vianli/exp-{'003' if kind == 'original' else '004'}.yaml")
        cfg["model"]["backbone"] = "offline"
        cfg["model"]["revision"] = None
        if kind == "moe":
            cfg["model"]["head"].update(expert_hidden_dim=12, dropout=0.0)
        cfg["training"].update(epochs=2, batch_size=2)
        cfg["logging"]["use_wandb"] = False
        cfg["logging"]["use_tracker"] = True
        cfg["diagnostics"]["collect_model_diagnostics"] = True
        cfg["paths"]["runs_dir"] = str(Path(directory) / "runs")
        return cfg

    def test_train_cli_restores_best_checkpoint_and_final_test_is_locked(self):
        cases = [("original", None, [3, 6], True), ("moe", 2, [2, 4, 6], True),
                 ("moe", 4, [4, 6], True), ("original", 2, [2, 4, 6], False)]
        for kind, frequency, expected, use_tracker in cases:
            with self.subTest(kind=kind, frequency=frequency, use_tracker=use_tracker), tempfile.TemporaryDirectory() as directory:
                config = self.config(directory, kind)
                config["training"]["eval_frequency"] = frequency
                config["logging"]["use_tracker"] = use_tracker
                source = Path(directory) / "config.yaml"
                source.write_text(yaml.safe_dump(config), encoding="utf-8")
                tracker = Path(directory) / "tracker.csv"
                bundle = tiny_bundle()
                hf = BertConfig(vocab_size=12, hidden_size=8, intermediate_size=12,
                                num_hidden_layers=1, num_attention_heads=2, num_labels=3).to_dict()
                built, best_weights, seen = [], {}, []
                def build(cfg, num_labels):
                    model = real_build_model(cfg, num_labels, hf_config=hf)
                    built.append(model)
                    return model
                real_eval = engine.evaluate
                def validation(model, loader, device, names):
                    metrics = real_eval(model, loader, device, names)
                    seen.append(True)
                    if len(seen) == 1:
                        best_weights.update({k: v.detach().clone() for k, v in model.state_dict().items()})
                    metrics["macro_f1"] = 1 / len(seen)
                    return metrics
                with patch("src.utils.artifacts.TRACKER", tracker), patch("src.models.factory.build_model", side_effect=build), \
                     patch.object(engine, "prepare_run_data", return_value=bundle), patch.object(engine, "evaluate", side_effect=validation):
                    path = train.main(["--config", str(source), "--run-id", "run-010", "--no-smoke-test"])
                manifest = json.loads((path / "manifest.json").read_text())
                self.assertEqual(manifest["status"], "completed")
                self.assertEqual(manifest["wandb_sync"], "disabled")
                for name, value in built[0].state_dict().items():
                    torch.testing.assert_close(value, best_weights[name])
                records = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]
                self.assertEqual([r["global_step"] for r in records if r["split"] == "validation"], expected)
                for record in records:
                    if record["split"] == "train":
                        self.assertAlmostEqual(record["loss"], record["classification_loss"] + .01 * record["balance_loss"], places=6)
                self.assertFalse((path / "evaluation/test.json").exists())
                self.assertEqual(json.loads((path / "diagnostics/validation.json").read_text())["scope"], "example")
                if use_tracker:
                    row = list(csv.DictReader(tracker.read_text(encoding="utf-8").splitlines()))[0]
                    self.assertEqual(row["val_macro_f1"], "1.0")
                    self.assertEqual(row["test_macro_f1"], "")
                else:
                    self.assertFalse(tracker.exists())
                    self.assertFalse(tracker.with_suffix(".lock").exists())
                finalist = copy.deepcopy(config)
                finalist["finalization"] = {"locked": True, "locked_at": "2026-11-22", "selection_reason": "offline test only", "seeds": [42]}
                lockfile = Path(directory) / "finalist.yaml"
                lockfile.write_text(yaml.safe_dump(finalist), encoding="utf-8")
                bad = copy.deepcopy(finalist)
                bad["model"]["freeze_backbone"] = True
                lockfile.write_text(yaml.safe_dump(bad), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Finalist"):
                    evaluate.evaluate_run(path, lockfile, tracker=tracker)
                lockfile.write_text(yaml.safe_dump(finalist), encoding="utf-8")
                with patch.object(evaluate, "prepare_run_data", return_value=bundle):
                    result = evaluate.evaluate_run(path, lockfile, tracker=tracker)
                self.assertEqual(result["split"], "test")
                self.assertTrue((path / "evaluation/predictions.json").exists())
                with self.assertRaises(FileExistsError):
                    evaluate.evaluate_run(path, lockfile, tracker=tracker)
                if use_tracker:
                    with self.assertRaises(ValueError):
                        reserve_run(config, "run-010", tracker=tracker)
                else:
                    self.assertFalse(tracker.exists())
                    self.assertFalse(tracker.with_suffix(".lock").exists())

    def test_failed_data_load_preserves_manifest_console_and_no_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self.config(directory, "original")
            cfg["profile"] = "full"
            tracker = Path(directory) / "tracker.csv"
            with patch.object(engine, "prepare_run_data", side_effect=RuntimeError("offline data failure")):
                with self.assertRaisesRegex(RuntimeError, "offline data failure"):
                    engine.run_training(cfg, "offline.yaml", [], "run-011", tracker=tracker)
            path = Path(directory) / "runs/run-011"
            manifest = json.loads((path / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertIn("offline data failure", (path / "console.log").read_text())
            row = list(csv.DictReader(tracker.read_text(encoding="utf-8").splitlines()))[0]
            self.assertEqual(row["status"], "failed")
            self.assertEqual(row["val_macro_f1"], "")

    def test_wandb_failure_keeps_local_completed_run(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory, "moe")
            config["profile"] = "full"
            config["logging"]["use_wandb"] = True
            config["training"]["epochs"] = 1
            hf = BertConfig(vocab_size=12, hidden_size=8, intermediate_size=12, num_hidden_layers=1,
                            num_attention_heads=2, num_labels=3).to_dict()
            model = real_build_model(config["model"], 3, hf_config=hf)
            fake_wandb = SimpleNamespace(init=Mock(side_effect=ConnectionError("offline W&B")))
            with patch.dict("sys.modules", {"wandb": fake_wandb}), \
                 patch.object(engine, "prepare_run_data", return_value=tiny_bundle()), \
                 patch("src.models.factory.build_model", return_value=model):
                path = engine.run_training(config, "offline.yaml", [], "run-012", tracker=Path(directory) / "tracker.csv")
            manifest = json.loads((path / "manifest.json").read_text())
            self.assertEqual((manifest["status"], manifest["wandb_sync"]), ("completed", "pending"))

    def test_duplicate_directory_and_console_secrets(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self.config(directory, "original")
            cfg["profile"] = "smoke"
            tracker = Path(directory) / "tracker.csv"
            artifacts = RunArtifacts(cfg, "run-013", "offline.yaml", ["logging.api_key=private-literal"], tracker=tracker)
            self.assertNotIn("private-literal", (artifacts.path / "manifest.json").read_text())
            secret = "example-private-secret-12345"
            with patch.dict("os.environ", {"HF_TOKEN": secret}), artifacts.console():
                print(secret)
            self.assertNotIn(secret, (artifacts.path / "console.log").read_text())
            before = (artifacts.path / "manifest.json").read_bytes()
            with self.assertRaises(ValueError):
                RunArtifacts(cfg, "run-013", "offline.yaml", ["logging.api_key=private-literal"], tracker=tracker)
            self.assertEqual(before, (artifacts.path / "manifest.json").read_bytes())

    def test_integer_display_labels_and_explicit_missing_path(self):
        data = Dataset.from_dict({"label_id": [0, 1, 2]})
        names, mapping = _label_info(data, "label_id", ["CLEAN", "OFFENSIVE", "HATE"])
        self.assertEqual(mapping, {0: 0, 1: 1, 2: 2})
        with patch("src.data.loader.load_dataset") as download:
            with self.assertRaises(FileNotFoundError):
                _load_raw({"name": "data/processed/not-present"})
            download.assert_not_called()
        raw = DatasetDict(train=data, validation=data)
        with self.assertRaisesRegex(ValueError, "test split"):
            _ensure_splits(raw, {"train_split": "train", "validation_split": "validation", "test_split": "test"})

    def test_config_fingerprint_excludes_seed_and_redacts_credentials(self):
        config = {key: {} for key in ("dataset", "model", "training", "routing")}
        first = config_fingerprint({**config, "seed": 42})
        self.assertEqual(first, config_fingerprint({**config, "seed": 43}))
        self.assertEqual(sanitized({"api_key": "private", "tokenizer": "public"}), {"api_key": "[REDACTED]", "tokenizer": "public"})


if __name__ == "__main__":
    unittest.main()
