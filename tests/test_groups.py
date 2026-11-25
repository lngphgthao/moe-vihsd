"""Group đăng ký tất cả run trước khi bắt đầu training."""
import csv
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import yaml
from scripts.run_group import main, run_group
from src.utils import load_config
from src.utils.artifacts import reserve_run

ROOT = Path(__file__).resolve().parents[1]

class TestGroups(unittest.TestCase):
    def test_cli_set_override_is_applied_and_forwarded_to_every_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tracker = root / "tracker.csv"
            run_root = root / "runs"
            calls = []

            def invoke_group(group_path, smoke_test, *, set_overrides):
                return run_group(group_path, smoke_test, tracker=tracker, set_overrides=set_overrides)

            group = load_config(ROOT / "configs/groups/vianli_screening.yaml")
            for entry in group["runs"]:
                entry["config"] = str((ROOT / "configs/groups" / entry["config"]).resolve())
            group_path = root / "group.yaml"
            group_path.write_text(yaml.safe_dump(group), encoding="utf-8")

            with patch.dict(os.environ, {"RUNS_DIR": str(run_root)}):
                with patch("scripts.run_group.run_group", side_effect=invoke_group):
                    with patch("train.main", side_effect=lambda argv: calls.append(argv)):
                        with patch(
                            "scripts.run_group.reserve_run",
                            wraps=reserve_run,
                        ) as reserve:
                            main([
                                "--group-config", str(group_path),
                                "--set", "logging.use_wandb=false",
                            ])

            self.assertEqual(len(calls), 2)
            for argv in calls:
                self.assertTrue(any(
                    argv[index:index + 2] == ["--set", "logging.use_wandb=false"]
                    for index in range(len(argv) - 1)
                ))
            self.assertTrue(all(
                call.args[0]["logging"]["use_wandb"] is False
                for call in reserve.call_args_list
            ))
            rows = list(csv.DictReader(tracker.read_text(encoding="utf-8").splitlines()))
            self.assertEqual(len(rows), 2)

    def test_auto_group_skips_existing_artifacts_and_allocates_new_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_root = root / "runs"
            (run_root / "run-010").mkdir(parents=True)
            tracker = root / "tracker.csv"
            calls = []
            def train(argv):
                calls.append(argv)
            with patch.dict(os.environ, {"RUNS_DIR": str(run_root)}):
                with patch("train.main", side_effect=train):
                    group = load_config(ROOT / "configs/groups/vianli_multilingual_screening.yaml")
                    for entry in group['runs']:
                        entry['run_id'] = 'auto'
                        entry['config'] = str((ROOT / 'configs/groups' / entry['config']).resolve())
                    group_path = root / 'auto_group.yaml'
                    group_path.write_text(yaml.safe_dump(group), encoding='utf-8')
                    run_group(group_path, True, tracker=tracker)
            run_ids = [argv[argv.index("--run-id") + 1] for argv in calls]
            self.assertEqual(run_ids, ["run-011", "run-012"])

    def test_explicit_group_rejects_existing_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_root = root / "runs"
            (run_root / "run-010").mkdir(parents=True)
            tracker = root / "tracker.csv"
            group = load_config(ROOT / "configs/groups/vianli_screening.yaml")
            group["runs"][0]["run_id"] = "run-010"
            group["overrides"] = {"logging.use_tracker": True}
            for entry in group["runs"]:
                entry["config"] = str((ROOT / "configs/groups" / entry["config"]).resolve())
            group_path = root / "explicit_group.yaml"
            group_path.write_text(yaml.safe_dump(group), encoding="utf-8")
            with patch.dict(os.environ, {"RUNS_DIR": str(run_root)}):
                with self.assertRaisesRegex(ValueError, "run-010.*Không rerun ID cũ"):
                    run_group(group_path, True, tracker=tracker)

    def test_group_paths_and_seeds_are_explicit_and_reserved(self):
        with tempfile.TemporaryDirectory() as directory:
            tracker = Path(directory) / "tracker.csv"
            run_root = Path(directory) / "runs"
            calls = []
            def train(argv):
                rows = list(csv.DictReader(tracker.read_text(encoding="utf-8").splitlines()))
                self.assertEqual(len(rows), 2)
                self.assertTrue(all(row["status"] == "planned" for row in rows))
                calls.append(argv)
            group = load_config(ROOT / "configs/groups/vianli_screening.yaml")
            group["overrides"] = {"logging.use_tracker": True}
            for entry in group["runs"]:
                entry["config"] = str((ROOT / "configs/groups" / entry["config"]).resolve())
            group_path = Path(directory) / "tracked_group.yaml"
            group_path.write_text(yaml.safe_dump(group), encoding="utf-8")
            with patch.dict(os.environ, {"RUNS_DIR": str(run_root)}):
                with patch("train.main", side_effect=train):
                    run_group(group_path, True, tracker=tracker)
                self.assertEqual(len(calls), 2)
                self.assertIn("--smoke-test", calls[0])
                self.assertIn("seed=42", calls[0])
                cfg = load_config(ROOT / "configs/experiments/vianli/exp-003.yaml")
                self.assertEqual(reserve_run(cfg, tracker=tracker), "run-005")
