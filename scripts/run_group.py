"""Run a YAML-defined group of reproducible training experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

# Allow direct execution with `python scripts/run_group.py` from the repository root.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import load_config


def _override_args(overrides: dict[str, Any]) -> list[str]:
    args: list[str] = []
    for key, value in overrides.items():
        serialized = yaml.safe_dump(value, default_flow_style=True).strip()
        args.extend(["--set", f"{key}={serialized}"])
    return args


def _run_entry(config_path: str, run_id: str, overrides: dict[str, Any], smoke_test: bool | None) -> None:
    from train import main as train_main

    argv = ["train.py", "--config", config_path, "--run-id", run_id]
    argv.extend(_override_args(overrides))
    if smoke_test is not None:
        argv.append("--smoke-test" if smoke_test else "--no-smoke-test")
    previous_argv = sys.argv
    try:
        sys.argv = argv
        train_main()
    finally:
        sys.argv = previous_argv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-config", required=True, help="YAML file describing the experiment group")
    parser.add_argument("--smoke-test", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()

    group_path = Path(args.group_config)
    group = load_config(group_path)
    default_config = str(group.get("base_config", "configs/vianli.yaml"))
    default_seeds = [int(seed) for seed in group.get("seeds", [42])]
    runs = group.get("runs", [])
    if not runs:
        raise ValueError(f"Group config contains no runs: {group_path}")

    group_overrides = group.get("overrides", {})
    if not isinstance(group_overrides, dict):
        raise TypeError("group.overrides must be a mapping")

    for run in runs:
        if not isinstance(run, dict) or "id" not in run:
            raise ValueError("Each group run must contain an 'id'")
        config_path = str(run.get("config", default_config))
        run_overrides = dict(group_overrides)
        run_overrides.update(run.get("overrides", {}))
        seeds = [int(seed) for seed in run.get("seeds", default_seeds)]
        for seed in seeds:
            overrides = dict(run_overrides)
            overrides["seed"] = seed
            run_id = f"{run['id']}-s{seed}"
            print(f"\n=== Starting {run_id} ===")
            _run_entry(config_path, run_id, overrides, args.smoke_test)


if __name__ == "__main__":
    main()
