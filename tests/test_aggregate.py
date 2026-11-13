"""Regression check cho discovery và reporting của result đã lưu."""

import json
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.analysis.aggregate import find_runs, format_latex_table, format_markdown_table, main, parse_run


class TestAggregate(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write_json(self, run, name, data):
        run.mkdir(parents=True, exist_ok=True)
        path = run / name
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_training_and_test_artifacts_are_combined(self):
        run = self.root / "run"
        path = self.write_json(run, "run_metrics.json", {"validation": {"macro_f1": 0.8}})
        self.write_json(run, "vihsd_predictions.json", {"macro_f1": 0.5})
        row = parse_run(run, path, self.root)
        self.assertEqual(row["val_macro_f1"], 80.0)
        self.assertEqual(row["test_macro_f1"], 50.0)
        self.assertIn(r"\textbf{50.00}", format_latex_table([row]))

    def test_missing_test_score_stays_missing(self):
        run = self.root / "run"
        path = self.write_json(run, "run_metrics.json", {"validation": {"macro_f1": 0.8}})
        row = parse_run(run, path, self.root)
        self.assertIsNone(row["test_macro_f1"])
        self.assertNotIn(r"\textbf{80.00}", format_latex_table([row]))
        self.assertNotIn("**80.00%**", format_markdown_table([row]))

    def test_zero_scores_and_settings_are_preserved(self):
        run = self.root / "moe"
        path = self.write_json(run, "run_metrics.json", {
            "validation": {"macro_f1": 0.8}, "test": {"macro_f1": 0.0},
        })
        self.write_json(run, "hyperparameters.json", {"flat_hyperparameters": {
            "model.num_experts": 0, "model.top_k": 0, "model.freeze_attention": False,
            "training.learning_rate": 0.0,
        }})
        row = parse_run(run, path, self.root)
        self.assertEqual(row["experts"], 0)
        self.assertEqual(row["top_k"], 0)
        self.assertIs(row["freeze_attention"], False)
        self.assertEqual(row["macro_f1"], 0.0)
        self.assertIn("**0.00%**", format_markdown_table([row]))
        self.assertIn(r"\textbf{0.00}", format_latex_table([row]))

    def test_missing_settings_are_not_guessed_from_name(self):
        run = self.root / "moe-top2"
        path = self.write_json(run, "run_metrics.json", {})
        row = parse_run(run, path, self.root)
        for key in ["experts", "top_k", "freeze_attention", "learning_rate", "pooling", "loss_type"]:
            self.assertEqual(row[key], "-")

    def test_nested_mixed_formats_are_all_found_once(self):
        new = self.root / "group" / "new"
        old = self.root / "group" / "old"
        self.write_json(new, "run_metrics.json", {})
        self.write_json(new, "vihsd_predictions.json", {})
        self.write_json(old, "vihsd_predictions.json", {})
        runs = find_runs(self.root)
        self.assertEqual({run for run, _ in runs}, {new, old})
        self.assertEqual(len(runs), 2)

    def test_resolved_config_fills_missing_metadata_and_preserves_zero(self):
        run = self.root / "run"
        path = self.write_json(run, "run_metrics.json", {})
        self.write_json(run, "hyperparameters.json", {"profile": "full"})
        (run / "resolved_config.yaml").write_text(
            "model:\n  architecture: transformer_moe\n  num_experts: 0\n  top_k: 1\n", encoding="utf-8",
        )
        row = parse_run(run, path, self.root)
        self.assertEqual(row["experts"], 0)
        self.assertEqual(row["architecture"], "transformer_moe")
        self.assertEqual(row["profile"], "full")

    def test_cli_exports_blank_missing_values_and_keeps_zero(self):
        self.write_json(self.root / "scored", "run_metrics.json", {"test": {"macro_f1": 0.0}})
        self.write_json(self.root / "missing", "run_metrics.json", {})
        with patch("sys.argv", ["aggregate_results.py", "--results-dir", str(self.root)]):
            main()
        with (self.root / "summary_table.csv").open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual([row["run_id"] for row in rows], ["scored", "missing"])
        self.assertEqual(rows[0]["test_macro_f1"], "0.0")
        self.assertEqual(rows[1]["test_macro_f1"], "")
