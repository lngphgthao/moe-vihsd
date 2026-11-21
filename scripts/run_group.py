"""Đăng ký group run trước khi thực thi; mỗi retry dùng run ID mới."""
import argparse
from pathlib import Path
import sys
import yaml
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.utils import apply_overrides, load_config
from src.utils.artifacts import TRACKER, reserve_run


def _override_args(overrides):
    return [f"{key}={yaml.safe_dump(value, default_flow_style=True).removesuffix('...\n').strip()}"
            for key, value in overrides.items()]


def run_group(group_path, smoke_test=None, *, tracker=TRACKER):
    from train import main as train_main
    group_path = Path(group_path)
    group = load_config(group_path)
    planned = []
    ids = set()
    for entry in group.get("runs", []):
        run_id = entry["run_id"]
        if run_id in ids:
            raise ValueError(f"Duplicate run ID in group: {run_id}")
        ids.add(run_id)
        config_path = (group_path.parent / entry["config"]).resolve()
        overrides = {**group.get("overrides", {}), **entry.get("overrides", {}), "seed": entry["seed"]}
        arguments = _override_args(overrides)
        config = apply_overrides(load_config(config_path), arguments)
        planned.append((config, config_path, run_id, arguments))
    if not planned:
        raise ValueError("Group must contain explicit runs with config, seed and run_id")
    for config, _, run_id, _ in planned:
        reserve_run(config, run_id, tracker=tracker)
    for _, path, run_id, overrides in planned:
        argv = ["--config", str(path), "--run-id", run_id]
        for override in overrides:
            argv.extend(["--set", override])
        if smoke_test is not None:
            argv.append("--smoke-test" if smoke_test else "--no-smoke-test")
        train_main(argv)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reserve and execute a group of experiments.")
    parser.add_argument("--group-config", required=True)
    parser.add_argument("--smoke-test", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args(argv)
    run_group(args.group_config, args.smoke_test)


if __name__ == "__main__":
    main()
