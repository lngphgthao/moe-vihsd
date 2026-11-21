"""Kiểm tra export W&B offline qua CLI và khi history tải lỗi."""

import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts.sync_wandb import export_run, main, path_component


def fake_run(run_id="abc123"):
    return SimpleNamespace(
        id=run_id, name="Thử nghiệm", path=["team", "project", run_id],
        url=f"https://wandb.ai/team/project/runs/{run_id}", state="finished",
        created_at="2026-11-13T00:00:00Z", tags=["colab"], notes=None,
        config={"seed": 42}, summary={"validation_macro_f1": 0.8},
        scan_history=Mock(return_value=iter([
            {"_step": 0, "loss": 0.0},
            {"_step": 2, "validation_macro_f1": 0.8, "loss": float("nan")},
        ])),
    )


class TestSyncWandb(unittest.TestCase):
    def test_cli_exports_all_runs_and_full_sparse_history(self):
        run = fake_run()
        api = Mock()
        api.runs.return_value = [run]
        wandb = SimpleNamespace(Api=Mock(return_value=api))
        with TemporaryDirectory() as directory:
            with patch.dict("sys.modules", {"wandb": wandb}), contextlib.redirect_stdout(io.StringIO()):
                status = main(["team/project", "--output-dir", directory])
            snapshot = json.loads((Path(directory) / "team/project/abc123.json").read_text(encoding="utf-8"))
        self.assertEqual(status, 0)
        api.runs.assert_called_once_with("team/project")
        run.scan_history.assert_called_once_with()
        self.assertEqual(snapshot["config"], {"seed": 42})
        self.assertEqual(snapshot["summary"], run.summary)
        self.assertEqual(snapshot["metadata"]["name"], "Thử nghiệm")
        self.assertEqual(snapshot["metadata"]["verification"], "unverified")
        self.assertEqual(snapshot["history"][0]["loss"], 0)
        self.assertEqual(snapshot["history"][1]["_step"], 2)
        self.assertIsNone(snapshot["history"][1]["loss"])
        self.assertIn("export_completed_at", snapshot)

    def test_failed_download_preserves_previous_snapshot_and_cleans_temp(self):
        def interrupted_history():
            yield {"_step": 0}
            raise ConnectionError("disconnected")

        with TemporaryDirectory() as directory:
            root = Path(directory)
            export_run(fake_run(), root)
            previous = (root / "abc123.json").read_bytes()
            run = fake_run()
            run.scan_history.return_value = interrupted_history()
            with self.assertRaises(ConnectionError):
                export_run(run, root)
            self.assertEqual((root / "abc123.json").read_bytes(), previous)
            self.assertEqual(list(root.glob("*.tmp")), [])
            export_run(fake_run(), root)
            self.assertEqual(len(list(root.glob("*.json"))), 1)

    def test_selected_runs_deduplicated_and_empty_history_supported(self):
        run = fake_run()
        run.scan_history.return_value = iter([])
        api = Mock()
        api.run.return_value = run
        with TemporaryDirectory() as directory:
            with patch.dict("sys.modules", {"wandb": SimpleNamespace(Api=Mock(return_value=api))}):
                with contextlib.redirect_stdout(io.StringIO()):
                    status = main(["team/project", "--run-id", "abc123", "--run-id", "abc123", "--output-dir", directory])
            snapshot = json.loads((Path(directory) / "team/project/abc123.json").read_text())
        self.assertEqual(status, 0)
        api.run.assert_called_once_with("team/project/abc123")
        api.runs.assert_not_called()
        self.assertEqual(snapshot["history"], [])

    def test_failure_returns_nonzero_and_other_runs_export(self):
        failed = fake_run("failed")
        failed.scan_history.side_effect = ConnectionError("disconnected")
        api = Mock()
        api.runs.return_value = [failed, fake_run()]
        with TemporaryDirectory() as directory:
            with patch.dict("sys.modules", {"wandb": SimpleNamespace(Api=Mock(return_value=api))}):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    status = main(["team/project", "--output-dir", directory])
            self.assertTrue((Path(directory) / "team/project/abc123.json").exists())
            self.assertFalse((Path(directory) / "team/project/failed.json").exists())
        self.assertEqual(status, 1)

    def test_unsafe_paths_rejected(self):
        for value in ("..", "../outside", "a/b", "a\\b", "C:outside", "CON", "trailing."):
            with self.subTest(value=value), self.assertRaises(ValueError):
                path_component(value)

    def test_credentials_redacted_and_missing_selected_run_does_not_stop_sync(self):
        run = fake_run()
        run.config = {"auth": {"WANDB_API_KEY": "hidden", "hf_token": "hidden"}, "max_tokens": 128}
        api = Mock()
        api.run.side_effect = [ValueError("Run not found"), run]
        with TemporaryDirectory() as directory:
            with patch.dict("sys.modules", {"wandb": SimpleNamespace(Api=Mock(return_value=api))}):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    status = main(["team/project", "--run-id", "missing", "--run-id", "abc123", "--output-dir", directory])
            snapshot = json.loads((Path(directory) / "team/project/abc123.json").read_text())
        self.assertEqual(status, 1)
        self.assertEqual(snapshot["config"]["auth"], {"WANDB_API_KEY": "[REDACTED]", "hf_token": "[REDACTED]"})
        self.assertEqual(snapshot["config"]["max_tokens"], 128)


if __name__ == "__main__":
    unittest.main()
