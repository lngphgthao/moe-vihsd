"""Regression cho nhiều process cập nhật cùng CSV và projection manifest."""
import csv
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from filelock import FileLock
from src.utils import load_config
from src.utils.artifacts import RunArtifacts, reserve_run, tracker_rows

ROOT = Path(__file__).resolve().parents[1]

# Import module directly so these CSV-only subprocesses do not import torch.
WORKER = r'''
import importlib.util
from pathlib import Path
import sys
import time
spec = importlib.util.spec_from_file_location("artifacts", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
tracker, run_id = Path(sys.argv[2]), sys.argv[3]
Path(sys.argv[4]).write_text("ready")
while not Path(sys.argv[5]).exists():
    time.sleep(.02)
for step in range(25):
    module.update_tracker(run_id, {"val_accuracy": step / 100, "notes": run_id}, tracker=tracker)
module.update_tracker(run_id, {"status": "completed", "val_macro_f1": 0.5}, tracker=tracker)
'''

class TestTrackerConcurrency(unittest.TestCase):
    def test_two_processes_wait_for_lock_and_keep_both_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            tracker = folder / 'tracker.csv'
            config = load_config(ROOT / 'configs/experiments/vianli/exp-003.yaml')
            for run_id in ['run-030', 'run-031']:
                reserve_run(config, run_id, tracker=tracker)
            processes = []
            try:
                for run_id in ['run-030', 'run-031']:
                    processes.append(subprocess.Popen(
                        [sys.executable, '-c', WORKER, str(ROOT / 'src/utils/artifacts.py'),
                         str(tracker), run_id, str(folder / f'{run_id}.ready'), str(folder / 'go')],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
                deadline = time.monotonic() + 90
                while not all((folder / f'{rid}.ready').exists() for rid in ['run-030', 'run-031']):
                    for process in processes:
                        if process.poll() is not None:
                            stdout, stderr = process.communicate()
                            self.fail('CSV worker exited before readiness: ' + stdout + stderr)
                    if time.monotonic() > deadline:
                        self.fail('CSV workers did not become ready')
                    time.sleep(.02)
                with FileLock(str(tracker.with_suffix('.lock'))):
                    (folder / 'go').write_text('go')
                    time.sleep(.1)
                    self.assertTrue(all(process.poll() is None for process in processes),
                                    'Writers must wait while another process holds the lock')
                for process in processes:
                    stdout, stderr = process.communicate(timeout=30)
                    self.assertEqual(process.returncode, 0, stdout + stderr)
                rows = list(csv.DictReader(tracker.read_text(encoding='utf-8').splitlines()))
                self.assertEqual(len(rows), 2)
                self.assertEqual({row['run_id'] for row in rows}, {'run-030', 'run-031'})
                for row in rows:
                    self.assertEqual(row['status'], 'completed')
                    self.assertEqual(row['notes'], row['run_id'])
                    self.assertEqual(float(row['val_accuracy']), .24)
                    self.assertEqual(float(row['val_macro_f1']), .5)
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                        process.communicate()

    def test_timeout_preserves_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'tracker.csv'
            path.write_bytes(b'original bytes')
            with FileLock(str(path.with_suffix('.lock'))):
                with self.assertRaises(TimeoutError):
                    with tracker_rows(path, timeout=.05):
                        self.fail('Lock must not be acquired')
            self.assertEqual(path.read_bytes(), b'original bytes')

    def test_manifest_keeps_scalar_dataset_and_model_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            config = load_config(ROOT / 'configs/experiments/vianli/exp-003.yaml')
            config['logging']['use_tracker'] = True
            config['profile'] = 'smoke'
            config['paths']['runs_dir'] = str(folder / 'runs')
            tracker = folder / 'tracker.csv'
            artifacts = RunArtifacts(config, 'run-032', 'offline.yaml', [], tracker=tracker)
            artifacts.save_manifest()
            row = list(csv.DictReader(tracker.read_text(encoding='utf-8').splitlines()))[0]
            self.assertEqual(row['dataset'], 'vianli')
            self.assertEqual(row['backbone'], 'xlm-roberta-base')
            self.assertEqual(row['architecture'], 'dense_encoder')
            self.assertEqual(row['variant'], 'original')
