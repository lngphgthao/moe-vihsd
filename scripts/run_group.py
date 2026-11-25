"""Chạy group experiment; CSV tracker chỉ dùng khi bật rõ ràng."""
import argparse
import os
import re
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


def run_group(group_path, smoke_test=None, *, tracker=TRACKER, set_overrides=None, auto_run_ids=False):
    from train import main as train_main
    group_path = Path(group_path)
    group = load_config(group_path)
    set_overrides = set_overrides or []
    planned = []
    ids = set()
    for entry in group.get("runs", []):
        run_id = entry.get("run_id")
        if not auto_run_ids and run_id in ids:
            raise ValueError(f"Duplicate run ID in group: {run_id}")
        if not auto_run_ids and run_id not in (None, "auto"):
            ids.add(run_id)
        config_path = (group_path.parent / entry["config"]).resolve()
        overrides = {**group.get("overrides", {}), **entry.get("overrides", {}), "seed": entry["seed"]}
        arguments = _override_args(overrides) + set_overrides
        config = apply_overrides(load_config(config_path), arguments)
        planned.append((config, config_path, run_id, arguments))
    if not planned:
        raise ValueError("Group must contain runs with config and seed")
    if auto_run_ids and any(not config.get("logging", {}).get("use_tracker", True)
                            for config, _, _, _ in planned):
        raise ValueError(
            "--auto-run-ids requires logging.use_tracker=true; "
            "with the CSV tracker disabled, provide explicit run IDs."
        )
    reserved = []
    for config, path, requested_run_id, overrides in planned:
        csv_tracker = tracker if config.get("logging", {}).get("use_tracker", True) else None
        allocation = "auto" if auto_run_ids else requested_run_id
        if csv_tracker is None and allocation in (None, "auto"):
            run_root = Path(os.getenv("RUNS_DIR", config.get("paths", {}).get("runs_dir", "runs")))
            used = [2]
            if run_root.exists():
                used.extend(int(path.name[4:]) for path in run_root.iterdir()
                            if re.fullmatch(r"run-\d+", path.name))
            candidate = max(used) + 1
            while (run_root / f"run-{candidate:03d}").exists():
                candidate += 1
            allocation = f"run-{candidate:03d}"
        run_id = reserve_run(config, allocation, tracker=csv_tracker)
        reserved.append((config, path, run_id, overrides, allocation))
    collisions = []
    for config, _, run_id, _, allocation in reserved:
        run_root = Path(os.getenv("RUNS_DIR", config.get("paths", {}).get("runs_dir", "runs")))
        existing = run_root / run_id
        if existing.exists() and allocation not in (None, "auto"):
            collisions.append(str(existing))
    if collisions:
        raise ValueError(
            "Artifact run đã tồn tại: "
            + ", ".join(collisions)
            + ". Không rerun ID cũ; giữ artifact/CSV hiện trạng và dùng run-NNN mới."
        )
    for _, path, run_id, overrides, _ in reserved:
        argv = ["--config", str(path), "--run-id", run_id]
        for override in overrides:
            argv.extend(["--set", override])
        if smoke_test is not None:
            argv.append("--smoke-test" if smoke_test else "--no-smoke-test")
        train_main(argv)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Execute a group with configured or automatically allocated run IDs.")
    parser.add_argument("--group-config", required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument(
        "--auto-run-ids",
        action="store_true",
        help="Allocate fresh run IDs from the active tracker/artifact directory instead of group IDs.",
    )
    parser.add_argument("--smoke-test", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args(argv)
    run_group(
        args.group_config,
        args.smoke_test,
        set_overrides=args.overrides,
        auto_run_ids=args.auto_run_ids,
    )


if __name__ == "__main__":
    main()
