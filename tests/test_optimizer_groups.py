"""Kiểm tra nhóm optimizer, tham số dùng chung và tương thích AdamW cũ."""

import copy
import unittest

import torch
from torch import nn

from train import create_optimizer_param_groups
from src.utils import apply_overrides, load_config
from pathlib import Path


def make_model():
    model = nn.ModuleDict({
        "backbone": nn.ModuleDict({"dense": nn.Linear(2, 2), "router": nn.Linear(2, 2)}),
        "classifier_moe": nn.ModuleDict({"expert": nn.Linear(2, 2), "router": nn.Linear(2, 2)}),
        "nlimoe": nn.ModuleDict({"expert": nn.Linear(2, 2), "router": nn.Linear(2, 2)}),
        "classifier": nn.Linear(2, 3),
        "router": nn.Linear(2, 2),
    })
    model["backbone"]["dense"].bias.requires_grad_(False)
    model["shared_alias"] = model["classifier"]
    return model


class TestOptimizerGroups(unittest.TestCase):
    def test_groups_cover_trainable_parameters_once_with_router_priority(self):
        model = make_model()
        for options in [{}, {"head_learning_rate": 0.002}, {"router_learning_rate": 0.003},
                        {"head_learning_rate": 0.0, "router_learning_rate": 0.003}]:
            with self.subTest(options=options):
                groups = create_optimizer_param_groups(model, {"learning_rate": 1e-5, **options})
                parameters = [p for group in groups for p in group["params"]]
                self.assertEqual(len(parameters), len({id(p) for p in parameters}))
                self.assertEqual({id(p) for p in parameters}, {id(p) for p in model.parameters() if p.requires_grad})
                self.assertEqual(sum(p.numel() for p in parameters), sum(p.numel() for p in model.parameters() if p.requires_grad))
                by_parameter = {id(p): group for group in groups for p in group["params"]}
                for name, parameter in model.named_parameters():
                    if not parameter.requires_grad:
                        continue
                    if "router" in name.split(".")[:-1]:
                        expected = "router"
                    elif name.startswith(("classifier_moe.", "nlimoe.", "classifier.")):
                        expected = "head"
                    else:
                        expected = "backbone"
                    self.assertEqual(by_parameter[id(parameter)]["name"], expected)
                    self.assertEqual(by_parameter[id(parameter)]["lr"], options.get(f"{expected}_learning_rate", 1e-5))

    def test_default_and_null_match_legacy_adamw_updates(self):
        for options in [{}, {"head_learning_rate": None, "router_learning_rate": None}]:
            original = make_model()
            grouped = copy.deepcopy(original)
            old_optimizer = torch.optim.AdamW([p for p in original.parameters() if p.requires_grad], lr=1e-5, weight_decay=0.01)
            groups = create_optimizer_param_groups(grouped, {"learning_rate": 1e-5, **options})
            self.assertEqual({group["lr"] for group in groups}, {1e-5})
            optimizer = torch.optim.AdamW(groups, lr=1e-5, weight_decay=0.01)
            for _ in range(2):
                for first, second in zip(original.parameters(), grouped.parameters()):
                    if first.requires_grad:
                        first.grad = torch.ones_like(first)
                        second.grad = first.grad.clone()
                old_optimizer.step()
                optimizer.step()
            for first, second in zip(original.parameters(), grouped.parameters()):
                torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_base_configs_allow_cli_overrides(self):
        root = Path(__file__).resolve().parents[1]
        for dataset in ["vianli", "vihsd"]:
            config = load_config(root / f"configs/{dataset}.yaml")
            self.assertIsNone(config["training"]["head_learning_rate"])
            self.assertIsNone(config["training"]["router_learning_rate"])
            apply_overrides(config, ["training.head_learning_rate=0.001", "training.router_learning_rate=0.002"])
            self.assertEqual(config["training"]["head_learning_rate"], 0.001)

    def test_invalid_learning_rates_are_rejected(self):
        for value in [-0.1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                create_optimizer_param_groups(make_model(), {"learning_rate": 1e-5, "head_learning_rate": value})
