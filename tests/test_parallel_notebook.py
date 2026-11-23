"""Fixture kiểm tra logic notebook; không phải bằng chứng chạy GPU thật."""
import ast
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from src.utils import load_config

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebooks/parallel_kaggle_smoke.ipynb'
MAIN_NOTEBOOKS = [
    ROOT / 'notebooks/main_kaggle.ipynb',
    ROOT / 'notebooks/main_colab.ipynb',
]


def notebook_functions():
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    definitions = []
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            tree = ast.parse(''.join(cell['source']))
            definitions.extend(node for node in tree.body if isinstance(node, ast.FunctionDef))
    return compile(ast.Module(body=definitions, type_ignores=[]), str(NOTEBOOK), 'exec')


class TestParallelNotebook(unittest.TestCase):
    def test_main_notebooks_leave_offline_suite_to_ci(self):
        for notebook in MAIN_NOTEBOOKS:
            data = json.loads(notebook.read_text(encoding='utf-8'))
            source = '\n'.join(
                ''.join(cell['source']) for cell in data['cells']
            )
            self.assertNotIn('unittest discover', source)
            self.assertIn('GitHub Actions', source)
    def fixtures(self, directory):
        folder = Path(directory)
        jobs = [dict(gpu='0', run_id='run-030', config='a.yaml', experiment_id='exp-003', backbone='offline-a'),
                dict(gpu='1', run_id='run-031', config='b.yaml', experiment_id='exp-007', backbone='offline-b')]
        rows = []
        for index, job in enumerate(jobs):
            output = folder / 'runs' / job['run_id']
            (output / 'checkpoints').mkdir(parents=True)
            (output / 'checkpoints/best.safetensors').write_bytes(b'fixture')
            manifest = dict(run_id=job['run_id'], experiment_id=job['experiment_id'], model={'backbone': job['backbone']},
                status='completed', profile='smoke', csv_tracker_enabled=True, hardware={'device': 'cuda', 'visible_gpu_count': 1, 'cuda_visible_devices': job['gpu']},
                training_started_at=f'2026-11-23T00:00:0{index * 2}+00:00', training_finished_at=f'2026-11-23T00:00:1{index * 2}+00:00',
                wandb_sync='synced', wandb_url='https://example.invalid/fixture')
            (output / 'manifest.json').write_text(json.dumps(manifest))
            (output / 'summary.json').write_text(json.dumps({'checkpoint': 'checkpoints/best.safetensors', 'validation': {'macro_f1': .5}, 'best_epoch': 1}))
            (output / 'metrics.jsonl').write_text('\n'.join(json.dumps({'split': split, 'loss': 1.0}) for split in ['train', 'validation']))
            rows.append(dict(run_id=job['run_id'], status='completed', dataset='vianli', backbone=job['backbone'], val_macro_f1=.5))
        tracker = folder / 'tracker.csv'
        with tracker.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        namespace = dict(Path=Path, csv=csv, json=json, math=math, datetime=datetime)
        exec(notebook_functions(), namespace)
        return jobs, tracker, folder / 'runs', namespace

    def test_cells_compile_and_diagnostics_are_smoke_with_tracker(self):
        data = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
        for cell in data['cells']:
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), str(NOTEBOOK), 'exec')
                self.assertIsNone(cell['execution_count'])
                self.assertEqual(cell['outputs'], [])
        setup_source = '\n'.join(
            ''.join(cell['source']) for cell in data['cells']
        )
        self.assertIn('HF_HUB_DISABLE_XET', setup_source)
        self.assertIn('test_console_stream.py', setup_source)
        self.assertNotIn('subprocess', setup_source)
        self.assertNotIn('Popen', setup_source)
        self.assertIn('get_ipython().system(command + " &")', setup_source)
        self.assertIn('.exit', setup_source)
        for name in ['vianli_xlmr_parallel_smoke.yaml', 'vianli_mdeberta_parallel_smoke.yaml']:
            config = load_config(ROOT / 'configs/diagnostics' / name)
            self.assertTrue(config['logging']['use_tracker'])
            self.assertTrue(config['training']['smoke_test'])
            self.assertEqual(config['training']['batch_size'], 8)
            self.assertEqual(config['training']['smoke'], {'epochs': 2, 'max_train_samples': 256, 'max_eval_samples': 64})

    def test_confirmation_requires_gpu_masks_overlap_and_own_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            jobs, tracker, runs, namespace = self.fixtures(directory)
            verify = namespace['verify_runs']
            codes = {job['run_id']: 0 for job in jobs}
            report = verify(jobs, codes, tracker, runs, True)
            self.assertTrue(report['passed'])
            self.assertEqual(report['training_overlap_seconds'], 8.0)
            path = runs / 'run-031/manifest.json'
            manifest = json.loads(path.read_text())
            manifest['hardware']['cuda_visible_devices'] = '0'
            path.write_text(json.dumps(manifest))
            self.assertFalse(verify(jobs, codes, tracker, runs, True)['passed'])
            manifest['hardware']['cuda_visible_devices'] = '1'
            manifest['training_started_at'] = '2026-11-23T00:00:11+00:00'
            path.write_text(json.dumps(manifest))
            self.assertFalse(verify(jobs, codes, tracker, runs, True)['passed'])
            manifest['training_started_at'] = '2026-11-23T00:00:02+00:00'
            path.write_text(json.dumps(manifest))
            summary = runs / 'run-031/summary.json'
            record = json.loads(summary.read_text())
            record['checkpoint'] = '../run-030/checkpoints/best.safetensors'
            summary.write_text(json.dumps(record))
            self.assertFalse(verify(jobs, codes, tracker, runs, True)['passed'])

    def test_confirmation_reports_missing_artifact_without_success(self):
        with tempfile.TemporaryDirectory() as directory:
            jobs, tracker, runs, namespace = self.fixtures(directory)
            (runs / 'run-031/summary.json').unlink()
            report = namespace['verify_runs'](jobs, {'run-030': 0, 'run-031': 1}, tracker, runs, False)
            self.assertFalse(report['passed'])
            self.assertIn('summary.json', report['runs']['run-031']['missing'])

