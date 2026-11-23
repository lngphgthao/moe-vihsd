"""Regression giao diện stdout/stderr khi console được ghi thêm ra file."""
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from tqdm import tqdm

# Test the stream wrapper without importing the training stack.
SPEC = importlib.util.spec_from_file_location(
    "console_artifacts", Path(__file__).resolve().parents[1] / "src/utils/artifacts.py")
artifacts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(artifacts)


class TestConsoleStream(unittest.TestCase):
    def test_terminal_queries_and_text_metadata_follow_original_stream(self):
        for terminal in (False, True):
            with self.subTest(terminal=terminal), tempfile.TemporaryFile(
                    mode="w+", encoding="utf-8", errors="replace") as original:
                with patch.object(original, "isatty", return_value=terminal):
                    tee = artifacts.Tee(original, io.StringIO())
                    self.assertEqual(tee.isatty(), terminal)
                    self.assertEqual(tee.encoding, "utf-8")
                    self.assertEqual(tee.errors, "replace")
                    self.assertEqual(tee.fileno(), original.fileno())
                    self.assertTrue(tee.writable())

    def test_console_progress_redaction_and_restoration_on_error(self):
        with tempfile.TemporaryDirectory() as directory:
            run = artifacts.RunArtifacts.__new__(artifacts.RunArtifacts)
            run.path = Path(directory)
            stdout, stderr = io.StringIO(), io.StringIO()
            secret = "console-regression-secret-12345"
            with patch.object(sys, "stdout", stdout), patch.object(sys, "stderr", stderr), \
                    patch.dict("os.environ", {"HF_TOKEN": secret}):
                with self.assertRaisesRegex(RuntimeError, "expected failure"):
                    with run.console():
                        self.assertFalse(sys.stdout.isatty())
                        self.assertFalse(sys.stderr.isatty())
                        print(secret)
                        for _ in tqdm(range(2), file=sys.stderr, disable=False, desc="progress"):
                            pass
                        sys.stderr.flush()
                        raise RuntimeError("expected failure")
                self.assertIs(sys.stdout, stdout)
                self.assertIs(sys.stderr, stderr)
            console = (run.path / "console.log").read_text(encoding="utf-8")
            self.assertIn("progress", console)
            self.assertIn("[REDACTED]", console)
            self.assertNotIn(secret, console)
            self.assertNotIn(secret, stdout.getvalue())
