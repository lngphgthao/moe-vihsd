"""Contract config giai đoạn 0–1: cùng budget, profile riêng và capacity control."""
from copy import deepcopy
from pathlib import Path
import unittest

import torch

from src.models.factory import validate_model_config
from src.models.heads.moe import MoEHead
from src.training.engine import apply_training_profile
from src.utils import load_config

ROOT = Path(__file__).resolve().parents[1]
CASES = {'vianli': ('010', '012', '013', '014'),
         'vihsd': ('011', '015', '016', '017')}


def experiment(dataset, number):
    return load_config(ROOT / f'configs/experiments/{dataset}/exp-{number}.yaml')


class TestPhoBERTStages(unittest.TestCase):
    def test_controls_keep_original_data_budget_and_pinned_encoder(self):
        for dataset, numbers in CASES.items():
            original = experiment(dataset, numbers[0])
            for number in numbers:
                with self.subTest(dataset=dataset, experiment=number):
                    config = experiment(dataset, number)
                    validate_model_config(config['model'])
                    self.assertEqual(config['experiment_id'], f'exp-{number}')
                    self.assertEqual(config['model']['backbone'], 'vinai/phobert-base')
                    self.assertEqual(config['model']['revision'],
                                     '01daacda68afe13d83023d16ec647239e344a1e6')
                    self.assertEqual(config['dataset']['tokenizer_revision'], config['model']['revision'])
                    self.assertEqual(config['dataset'], original['dataset'])
                    self.assertEqual(config['training'], original['training'])
                    self.assertEqual(config['diagnostics'], original['diagnostics'])
                    self.assertFalse(config['model']['freeze_backbone'])
            for number in numbers[1:3]:
                config = experiment(dataset, number)
                self.assertEqual(config['model']['head']['num_experts'], 1)
                self.assertEqual(config['model']['head']['router']['top_k'], 1)
                self.assertEqual(config['routing']['balance_loss_coef'], 0)

    def test_groups_have_disjoint_ids_seeds_and_diagnostic_limits(self):
        ids = set()
        for dataset, numbers in CASES.items():
            for stage in ('stage0_baseline', 'stage1_screening'):
                group_path = ROOT / f'configs/groups/{dataset}_phobert_{stage}.yaml'
                group = load_config(group_path)
                expected = ([f'exp-{numbers[0]}'] * len(group['runs']) if stage == 'stage0_baseline' else
                            [f'exp-{number}' for number in numbers[1:]])
                actual, seeds = [], []
                for entry in group['runs']:
                    with self.subTest(dataset=dataset, stage=stage, run_id=entry['run_id']):
                        self.assertNotIn(entry['run_id'], ids)
                        ids.add(entry['run_id'])
                        config = load_config(group_path.parent / entry['config'])
                        actual.append(config['experiment_id'])
                        seeds.append(entry['seed'])
                        self.assertTrue(config['logging']['use_tracker'])
                        for smoke in (True, False):
                            profiled = deepcopy(config)
                            self.assertEqual(apply_training_profile(profiled, smoke), smoke)
                            training = profiled['training']
                            self.assertEqual((training['epochs'], training['max_train_samples'],
                                              training['max_eval_samples']),
                                             (1, 64, 32) if smoke else (7, None, None))
                self.assertEqual(actual, expected)
                if stage == 'stage0_baseline':
                    self.assertTrue(seeds)
                    self.assertTrue(set(seeds).issubset({42, 123, 2026}))
                    self.assertEqual(len(seeds), len(set(seeds)))
                else:
                    self.assertEqual(seeds, [42] * len(expected))

    def test_reference_and_capacity_controls_forward_backward(self):
        for dataset, numbers in CASES.items():
            counts = {}
            for role, number in zip(('C1', 'C2', 'M0'), numbers[1:]):
                config = experiment(dataset, number)
                head = deepcopy(config['model']['head'])
                head.pop('type')
                self.assertEqual(head.pop('pooling'), 'cls')
                router = head.pop('router')
                self.assertEqual((router['scoring'], router['selection']), ('linear', 'top_k'))
                model = MoEHead(768, 3, top_k=router['top_k'], **head)
                counts[role] = sum(parameter.numel() for parameter in model.parameters())
                logits, aux = model(torch.randn(2, 768, requires_grad=True))
                self.assertEqual(tuple(logits.shape), (2, 3))
                loss = logits.square().mean() + config['routing']['balance_loss_coef'] * aux['balance_loss']
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(any(parameter.grad is not None for expert in model.experts
                                    for parameter in expert.parameters()))
            self.assertEqual(counts, {'C1': 790788, 'C2': 3156231, 'M0': 3156231})
