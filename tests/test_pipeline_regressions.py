"""Regression check cho group config, metric và lỗi preprocessing."""

from pathlib import Path
from unittest.mock import Mock, patch
import unittest

from scripts import preprocess_vihsd, run_group, segment_vianli
from src.evaluation.metrics import compute_classification_metrics
from src.data.loader import prepare_data
from src.utils import apply_overrides, load_config


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestPipelineRegressions(unittest.TestCase):
    def test_data_loader_keeps_optional_segment_ids(self):
        from datasets import Dataset, DatasetDict

        examples = {"premise": ["first", "second"], "hypothesis": ["pair", "pair"], "label": ["a", "b"]}
        raw = DatasetDict({name: Dataset.from_dict(examples) for name in ["train", "validation", "test"]})
        config = {
            "input_mode": "nli", "name": "offline", "tokenizer": "offline",
            "premise_column": "premise", "hypothesis_column": "hypothesis",
            "label_column": "label", "label_names": ["a", "b"], "max_length": 4,
            "train_split": "train", "validation_split": "validation", "test_split": "test",
            "batch_size": 2, "num_workers": 0,
        }
        for with_segments in [True, False]:
            with self.subTest(with_segments=with_segments):
                encoded = {"input_ids": [[1, 2, 3, 0]] * 2, "attention_mask": [[1, 1, 1, 0]] * 2}
                if with_segments:
                    encoded["token_type_ids"] = [[0, 0, 1, 0]] * 2
                tokenizer = Mock(return_value=encoded)
                with patch("src.data.loader._load_raw", return_value=raw):
                    with patch("src.data.loader.AutoTokenizer.from_pretrained", return_value=tokenizer):
                        bundle = prepare_data(config)
                for loader in bundle.loaders.values():
                    batch = next(iter(loader))
                    self.assertEqual("token_type_ids" in batch, with_segments)
                    if with_segments:
                        self.assertEqual(batch["token_type_ids"].tolist(), [[0, 0, 1, 0]] * 2)

    def test_group_runs_use_the_declared_dense_baseline(self):
        for filename in ["vianli_dense_backbones.yaml", "vianli_stage1_dense_backbones.yaml"]:
            with self.subTest(group=filename):
                group_path = REPO_ROOT / "configs/groups" / filename
                with patch("sys.argv", ["run_group.py", "--group-config", str(group_path)]):
                    with patch.object(run_group, "_run_entry") as run:
                        run_group.main()
                self.assertEqual(run.call_count, 9)
                seeds_by_backbone = {}
                for call in run.call_args_list:
                    config_path, run_id, overrides, _ = call.args
                    self.assertEqual(Path(config_path), REPO_ROOT / "configs/baselines/vianli_dense.yaml")
                    resolved = apply_overrides(load_config(config_path), run_group._override_args(overrides)[1::2])
                    self.assertEqual(resolved["logging"]["project"], "dense-vianli")
                    self.assertEqual(resolved["model"]["architecture"], "dense_transformer")
                    seeds_by_backbone.setdefault(resolved["model"]["pretrained_model_name"], []).append(resolved["seed"])
                self.assertEqual(len(seeds_by_backbone), 3)
                for seeds in seeds_by_backbone.values():
                    self.assertEqual(seeds, [42, 43, 44])

    def test_macro_f1_includes_all_configured_classes(self):
        metrics = compute_classification_metrics([0, 0], [0, 0], ["a", "b", "c"])
        self.assertAlmostEqual(metrics["macro_f1"], 1 / 3)
        self.assertAlmostEqual(metrics["macro_f1"], metrics["classification_report"]["macro avg"]["f1-score"])
        self.assertEqual(metrics["accuracy"], 1.0)

    def test_segmentation_failure_stops_both_preprocessors_before_saving(self):
        for module, example in [
            (preprocess_vihsd, {"free_text": ["sample"], "label_id": [0]}),
            (segment_vianli, {"premise": ["sample"], "hypothesis": ["sample"], "label": ["neutral"]}),
        ]:
            with self.subTest(module=module.__name__):
                with patch("sys.argv", [module.__name__]):
                    with patch.object(module, "ensure_vncorenlp", return_value="dummy.jar"):
                        with patch.object(module, "load_dataset", return_value={"train": example}):
                            with patch.object(module, "VnCoreNLP") as segmenter:
                                segmenter.return_value.__enter__.return_value.tokenize.side_effect = ConnectionError("service down")
                                with patch("datasets.DatasetDict.save_to_disk") as save:
                                    with self.assertRaisesRegex(ConnectionError, "service down"):
                                        module.main()
                                    save.assert_not_called()
                                segmenter.return_value.__exit__.assert_called_once()

    def test_empty_and_valid_segmentation(self):
        for module in [preprocess_vihsd, segment_vianli]:
            with self.subTest(module=module.__name__):
                with patch.object(module, "VnCoreNLP") as segmenter:
                    segmenter.tokenize.return_value = [["học_sinh", "đọc"], ["sách"]]
                    self.assertEqual(module.segment_text(segmenter, "sample"), "học_sinh đọc sách")
                    segmenter.tokenize.reset_mock()
                    self.assertEqual(module.segment_text(segmenter, None), "")
                    self.assertEqual(module.segment_text(segmenter, "  "), "")
                    segmenter.tokenize.assert_not_called()
