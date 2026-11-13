"""Kiểm tra tính nhất quán cấu hình nghiên cứu, không dùng kết quả thực nghiệm."""

from pathlib import Path
import unittest

from src.utils import load_config


ROOT = Path(__file__).resolve().parents[1]


class TestResearchConfigs(unittest.TestCase):
    def test_noise_ablations_each_change_only_initialization_noise(self):
        for filename, noise in [("vianli_moe_noise_0_001.yaml", 0.001), ("vianli_moe_noise_0_01.yaml", 0.01)]:
            with self.subTest(config=filename):
                expected = load_config(ROOT / "configs/finalists/vianli_best.yaml")
                self.assertEqual(expected["model"]["expert_init_noise"], 0.0)
                self.assertTrue(expected["model"]["upcycle"])
                expected["model"]["expert_init_noise"] = noise
                actual = load_config(ROOT / "configs/ablations" / filename)
                self.assertEqual(actual, expected)

    def test_no_inner_dropout_ablation_changes_only_one_variable(self):
        expected = load_config(ROOT / "configs/finalists/vianli_best.yaml")
        self.assertIsNone(expected["model"]["expert_inner_dropout"])
        expected["model"]["expert_inner_dropout"] = 0.0
        actual = load_config(ROOT / "configs/ablations/vianli_moe_no_inner_dropout.yaml")
        self.assertEqual(actual, expected)

    def test_vianli_finalist_name_matches_current_expert_count(self):
        config = load_config(ROOT / "configs/finalists/vianli_best.yaml")
        self.assertIn(f"-e{config['model']['num_experts']}-", config["logging"]["run_name"])
