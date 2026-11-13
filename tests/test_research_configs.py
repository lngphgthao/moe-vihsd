"""Kiểm tra tính nhất quán cấu hình nghiên cứu, không dùng kết quả thực nghiệm."""

from pathlib import Path
import unittest

from src.utils import load_config


ROOT = Path(__file__).resolve().parents[1]


class TestResearchConfigs(unittest.TestCase):
    def test_vianli_finalist_name_matches_current_expert_count(self):
        config = load_config(ROOT / "configs/finalists/vianli_best.yaml")
        self.assertIn(f"-e{config['model']['num_experts']}-", config["logging"]["run_name"])
