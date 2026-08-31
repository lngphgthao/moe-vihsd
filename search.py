"""Run reproducible grid or random hyperparameter searches."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


def dotted_metric(values: dict, path: str):
    result = values
    for key in path.split("."):
        result = result[key]
    return result


def make_candidates(parameters: dict, strategy: str, trials: int, seed: int):
    keys = list(parameters)
    if strategy == "grid":
        all_candidates = [dict(zip(keys, values)) for values in itertools.product(*(parameters[key] for key in keys))]
        return all_candidates[:trials] if trials else all_candidates
    if strategy != "random":
        raise ValueError("strategy must be 'grid' or 'random'.")
    generator = random.Random(seed)
    return [{key: generator.choice(parameters[key]) for key in keys} for _ in range(trials)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space", default="configs/search.yaml", help="YAML search-space definition.")
    parser.add_argument("--trials", type=int, default=None, help="Override the number of candidate configurations.")
    args = parser.parse_args()
    spec = yaml.safe_load(Path(args.space).read_text(encoding="utf-8"))
    strategy = spec.get("strategy", "random")
    trials = args.trials if args.trials is not None else int(spec.get("trials", 20))
    candidates = make_candidates(spec["parameters"], strategy, trials, int(spec.get("search_seed", 42)))
    started = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_root = Path(os.getenv("RESULTS_DIR", spec.get("results_dir", "results")))
    records = []
    for trial_index, candidate in enumerate(candidates, start=1):
        overrides = {**spec.get("fixed", {}), **candidate}
        for run_seed in spec.get("seeds", [42]):
            overrides["seed"] = run_seed
            run_id = f"search-{started}-t{trial_index:03d}-s{run_seed}"
            command = [sys.executable, "train.py", "--config", spec.get("base_config", "configs/vihsd.yaml"), "--run-id", run_id]
            for key, value in overrides.items():
                command.extend(["--set", f"{key}={json.dumps(value)}"])
            completed = subprocess.run(command, check=False)
            record = {"trial": trial_index, "seed": run_seed, "run_id": run_id, "overrides": overrides, "return_code": completed.returncode}
            metrics_path = results_root / run_id / "run_metrics.json"
            if completed.returncode == 0 and metrics_path.exists():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                record["validation_macro_f1"] = dotted_metric(metrics, "validation.macro_f1")
                record["validation_accuracy"] = dotted_metric(metrics, "validation.accuracy")
            records.append(record)
            print(record)
    records.sort(key=lambda record: record.get("validation_macro_f1", float("-inf")), reverse=True)
    output_path = results_root / f"search-{started}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"spec": spec, "records": records}, indent=2), encoding="utf-8")
    print(f"Search summary: {output_path}")


if __name__ == "__main__":
    main()
