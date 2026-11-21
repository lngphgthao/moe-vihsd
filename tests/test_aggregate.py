"""Không trộn seed/config/profile và không biến missing thành zero."""
import unittest
from src.analysis.aggregate import aggregate

class TestAggregate(unittest.TestCase):
    def row(self, seed, score, **overrides):
        return {"experiment_id": "exp-003", "config_fingerprint": "same", "seed": seed,
                "status": "completed", "profile": "full", "validation_macro_f1": score,
                "test_macro_f1": None, **overrides}

    def test_missing_zero_sample_sd_and_profiles(self):
        rows = [self.row(42, 0.0), self.row(43, 1.0), self.row(44, .9, profile="smoke")]
        record = aggregate(rows)[0]
        self.assertEqual(record["validation_n"], 2)
        self.assertEqual(record["validation_mean_macro_f1"], .5)
        self.assertAlmostEqual(record["validation_sample_sd"], 2**-.5)
        self.assertEqual(record["test_n"], 0)
        self.assertIsNone(record["test_mean_macro_f1"])

    def test_distinct_configs_not_combined_and_duplicate_seed_rejected(self):
        self.assertEqual(len(aggregate([self.row(42, .1), self.row(43, .2, config_fingerprint="different")])), 2)
        with self.assertRaises(ValueError):
            aggregate([self.row(42, .1), self.row(42, .2)])
