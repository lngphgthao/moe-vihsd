"""Notebook workflow contracts; no notebook cell or GPU run is executed here."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import yaml

from src.training.engine import apply_training_profile
from src.utils import apply_overrides, load_config

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    ROOT / "notebooks/parallel_kaggle_smoke.ipynb",
    ROOT / "notebooks/main_kaggle.ipynb",
    ROOT / "notebooks/main_colab.ipynb",
]
PAIR_CONFIGS = [
    "../experiments/vianli/exp-010.yaml",
    "../experiments/vihsd/exp-011.yaml",
]
PAIR_GROUPS = {
    "phobert_original_pair_baseline.yaml": (False, True, ["run-022", "run-025"]),
}


def load_notebook(path):
    return json.loads(path.read_text(encoding="utf-8"))


def python_source(cell):
    # Giữ suite của if/for hợp lệ khi thay lệnh shell IPython bằng Python để compile.
    return "\n".join(
        line[:len(line) - len(line.lstrip())] + "pass"
        if line.lstrip().startswith("!") else line
        for line in "".join(cell["source"]).splitlines()
    )


class TestNotebookWorkflows(unittest.TestCase):
    def test_notebooks_compile_and_expose_both_run_options(self):
        for path in NOTEBOOKS:
            with self.subTest(notebook=path.name):
                notebook = load_notebook(path)
                code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
                source = "\n".join("".join(cell["source"]) for cell in code_cells)
                for cell in code_cells:
                    compile(python_source(cell), f"{path.name}:cell", "exec")
                    self.assertIsNone(cell["execution_count"])
                    self.assertEqual(cell["outputs"], [])

                mask = "0,1" if path.name.startswith("parallel_") else "0"
                self.assertIn(f'os.environ["CUDA_VISIBLE_DEVICES"] = "{mask}"', source)
                self.assertIn("OPTION A", source)
                self.assertIn("OPTION B", source)
                self.assertIn("configs/experiments/vianli/exp-010.yaml", source)
                self.assertIn("configs/experiments/vihsd/exp-011.yaml", source)
                self.assertIn("scripts/run_group.py", source)
                self.assertIn("scripts/aggregate_results.py", source)
                self.assertIn("evaluate.py", source)
                self.assertNotIn("subprocess", source)
                self.assertNotIn("unittest discover", source)

    def test_parallel_waits_for_both_workers_and_keeps_gpu_masks_separate(self):
        notebook = load_notebook(NOTEBOOKS[0])
        source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
        self.assertIn("CUDA_VISIBLE_DEVICES=0", source)
        self.assertIn("CUDA_VISIBLE_DEVICES=1", source)
        self.assertIn('wait "$first_pid"', source)
        self.assertIn('wait "$second_pid"', source)
        self.assertIn('[ "$first_status" -eq 0 ] && [ "$second_status" -eq 0 ]', source)

    def test_runtime_groups_keep_smoke_ids_and_logging_separate_from_full(self):
        for path in NOTEBOOKS:
            with self.subTest(notebook=path.name), tempfile.TemporaryDirectory() as directory:
                namespace = {"REPO_DIR": ROOT, "OUTPUT_DIR": Path(directory)}
                cell = next(cell for cell in load_notebook(path)["cells"]
                            if "".join(cell["source"]).startswith("from copy import deepcopy"))
                exec("".join(cell["source"]), namespace)
                smoke_entries, full_entries = [], []
                for group_path in Path(directory).glob("*.yaml"):
                    group = load_config(group_path)
                    smoke = "smoke" in group_path.name
                    target = smoke_entries if smoke else full_entries
                    for entry in group["runs"]:
                        config = load_config(entry["config"])
                        config = apply_overrides(config, [
                            f"{key}={str(value).lower()}"
                            for key, value in group.get("overrides", {}).items()
                        ])
                        self.assertEqual(config["logging"]["use_wandb"], not smoke)
                        self.assertTrue(config["logging"]["use_tracker"])
                        self.assertEqual(entry["seed"], 42)
                        target.append((config["experiment_id"], entry["run_id"]))
                self.assertEqual(sorted(smoke_entries), [("exp-010", "run-020"), ("exp-011", "run-021")])
                if path.name.startswith("parallel_"):
                    self.assertEqual(sorted(full_entries), [("exp-010", "run-022"), ("exp-011", "run-025")])

    def test_pair_groups_resolve_to_the_two_phobert_original_configs(self):
        for filename, (smoke, wandb_enabled, run_ids) in PAIR_GROUPS.items():
            path = ROOT / "configs/groups" / filename
            group = load_config(path)
            self.assertEqual([entry["config"] for entry in group["runs"]], PAIR_CONFIGS)
            self.assertEqual([entry["seed"] for entry in group["runs"]], [42, 42])
            self.assertEqual([entry["run_id"] for entry in group["runs"]], run_ids)
            overrides = [
                f"{key}={str(value).lower()}"
                for key, value in group.get("overrides", {}).items()
            ]
            for entry, experiment_id in zip(group["runs"], ("exp-010", "exp-011")):
                config = load_config(path.parent / entry["config"])
                config = apply_overrides(config, overrides)
                self.assertEqual(config["experiment_id"], experiment_id)
                self.assertEqual(config["logging"]["use_wandb"], wandb_enabled)
                for smoke in (True, False):
                    profiled = deepcopy(config)
                    self.assertEqual(apply_training_profile(profiled, smoke), smoke)
                    training = profiled["training"]
                    self.assertEqual((training["epochs"], training["max_train_samples"],
                                      training["max_eval_samples"]),
                                     (1, 64, 32) if smoke else (7, None, None))

    def test_main_notebooks_leave_offline_suite_to_ci(self):
        workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
        job = workflow["jobs"]["tests"]
        commands = "\n".join(step.get("run", "") for step in job["steps"])
        self.assertIn("python -m unittest discover -s tests -v", commands)
        for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
            self.assertEqual(job["env"][key], "1")
        self.assertEqual(job["env"]["WANDB_MODE"], "offline")
        for path in NOTEBOOKS:
            source = "\n".join(
                "".join(cell["source"]) for cell in load_notebook(path)["cells"]
                if cell["cell_type"] == "code"
            )
            self.assertNotIn("unittest discover", source)
