"""Tắt tracker không được đọc/ghi CSV, kể cả khi hai run khởi tạo đồng thời."""
import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest
import yaml
from unittest.mock import patch

from scripts.run_group import run_group
from src.utils import load_config
from src.utils.artifacts import RunArtifacts, reserve_run, update_tracker

ROOT = Path(__file__).resolve().parents[1]

class TestTrackerDisabled(unittest.TestCase):
    def test_concurrent_artifacts_leave_tracker_and_lock_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            tracker = folder / 'tracker.csv'
            tracker.write_bytes(b'not-a-tracker-do-not-read-or-write')
            lock = tracker.with_suffix('.lock')
            lock.write_bytes(b'existing-lock-do-not-touch')
            config = load_config(ROOT / 'configs/experiments/vianli/exp-003.yaml')
            config['logging']['use_tracker'] = False
            config['profile'] = 'smoke'
            config['paths']['runs_dir'] = str(folder / 'runs')
            def create(run_id):
                artifacts = RunArtifacts(copy.deepcopy(config), run_id, 'offline.yaml', [], tracker=tracker)
                artifacts.manifest['status'] = 'completed'
                artifacts.save_manifest()
                update_tracker(run_id, {'status': 'completed'}, tracker=artifacts.tracker)
                return artifacts
            with ThreadPoolExecutor(max_workers=2) as pool:
                outputs = list(pool.map(create, ['run-030', 'run-031']))
            self.assertTrue(all(artifact.tracker is None for artifact in outputs))
            self.assertTrue(all((artifact.path / 'manifest.json').exists() for artifact in outputs))
            self.assertEqual(tracker.read_bytes(), b'not-a-tracker-do-not-read-or-write')
            self.assertEqual(lock.read_bytes(), b'existing-lock-do-not-touch')
            with self.assertRaises(FileExistsError):
                create('run-030')

    def test_group_explicit_opt_out_does_not_register_or_update_tracker(self):
        with tempfile.TemporaryDirectory() as directory:
            tracker = Path(directory) / 'tracker.csv'
            tracker.write_bytes(b'keep-existing-csv')
            group = load_config(ROOT / 'configs/groups/vianli_multilingual_smoke.yaml')
            group['overrides']['logging.use_tracker'] = False
            for entry in group['runs']:
                entry['config'] = str((ROOT / 'configs/groups' / entry['config']).resolve())
            path = Path(directory) / 'group.yaml'
            path.write_text(yaml.safe_dump(group), encoding='utf-8')
            with patch('train.main') as train, patch('scripts.run_group.reserve_run', wraps=reserve_run) as reserve:
                run_group(path, True, tracker=tracker)
            self.assertEqual(train.call_count, 2)
            self.assertTrue(all(call.kwargs['tracker'] is None for call in reserve.call_args_list))
            self.assertEqual(tracker.read_bytes(), b'keep-existing-csv')

    def test_no_tracker_requires_explicit_safe_run_id(self):
        config = load_config(ROOT / 'configs/experiments/vianli/exp-003.yaml')
        self.assertEqual(reserve_run(config, 'run-030', tracker=None), 'run-030')
        for run_id in [None, '../outside', 'run-001']:
            with self.subTest(run_id=run_id), self.assertRaises(ValueError):
                reserve_run(config, run_id, tracker=None)
