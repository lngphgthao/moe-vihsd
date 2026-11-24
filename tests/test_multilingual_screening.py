"""Contract của screening multilingual: raw data, revision đúng và run ID riêng."""
from pathlib import Path
import unittest

from src.utils import load_config
from src.training.engine import prepare_run_data
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]

class TestMultilingualScreening(unittest.TestCase):
    def test_original_heads_revisions_and_raw_dataset_sources(self):
        cases = [('vianli', 'exp-003', 'xlm-roberta-base'),
                 ('vianli', 'exp-007', 'microsoft/mdeberta-v3-base'),
                 ('vihsd', 'exp-008', 'xlm-roberta-base'),
                 ('vihsd', 'exp-009', 'microsoft/mdeberta-v3-base')]
        budgets = {}
        for dataset, experiment, backbone in cases:
            with self.subTest(experiment=experiment):
                config = load_config(ROOT / f'configs/experiments/{dataset}/{experiment}.yaml')
                self.assertEqual(config['experiment_id'], experiment)
                self.assertEqual(config['model']['backbone'], backbone)
                self.assertEqual(config['model']['head']['type'], 'original')
                self.assertFalse(config['model']['freeze_backbone'])
                self.assertEqual(len(config['model']['revision']), 40)
                self.assertEqual(config['dataset']['tokenizer_revision'], config['model']['revision'])
                self.assertEqual(config['dataset']['name'], 'uitnlp/ViANLI' if dataset == 'vianli' else 'uitnlp/vihsd')
                self.assertIsNotNone(config['dataset']['revision'])
                if dataset == 'vihsd':
                    self.assertEqual(config['dataset']['expected_split_sizes']['train'], 24048)
                    self.assertEqual(config['dataset']['filtered_split_sizes']['train'], 24046)
                self.assertEqual(config['training']['loss_type'], 'cross_entropy')
                current = (config['training']['epochs'], config['training']['batch_size'],
                           config['training']['learning_rate'], config['dataset']['max_length'])
                self.assertEqual(current, budgets.setdefault(dataset, current))
                with patch('src.data.loader.prepare_data') as prepare:
                    prepare_run_data(config)
                self.assertEqual(prepare.call_args.args[0]['tokenizer'], backbone)
                self.assertEqual(prepare.call_args.args[0]['tokenizer_revision'], config['model']['revision'])

    def test_screening_groups_have_distinct_ids_with_csv_tracking(self):
        ids = set()
        for dataset in ['vianli', 'vihsd']:
            group = load_config(ROOT / f'configs/groups/{dataset}_multilingual_screening.yaml')
            self.assertEqual(len(group['runs']), 2)
            for entry in group['runs']:
                self.assertNotIn(entry['run_id'], ids)
                ids.add(entry['run_id'])
                config = load_config(ROOT / 'configs/groups' / entry['config'])
                self.assertTrue(config['logging']['use_tracker'])
                self.assertEqual(entry['seed'], 42)
