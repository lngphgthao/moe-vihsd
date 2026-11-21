"""Tải snapshot W&B để agent có thể phân tích offline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path, PureWindowsPath
import re
import sys
import tempfile


def path_component(value: str) -> str:
    if (
        not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", value)
        or value.endswith(".")
        or PureWindowsPath(value).is_reserved()
    ):
        raise ValueError(f"Invalid W&B path component: {value!r}")
    return value


def json_value(value):
    # JSON chuẩn dùng null cho metric NaN/Infinity để các công cụ khác đọc được.
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            normalized_key = key.lower().replace("-", "_").replace(".", "_")
            sensitive = ("api_key", "token", "password", "secret", "credentials", "private_key")
            if any(normalized_key == name or normalized_key.endswith("_" + name) for name in sensitive):
                result[key] = "[REDACTED]"
            else:
                result[key] = json_value(item)
        return result
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def export_run(run, project_dir: Path) -> tuple[Path, int]:
    run_id = path_component(run.id)
    project_dir.mkdir(parents=True, exist_ok=True)
    destination = project_dir / f"{run_id}.json"
    snapshot = {
        "metadata": {
            "id": run.id,
            "name": run.name,
            "path": run.path,
            "url": run.url,
            "state": run.state,
            "created_at": run.created_at,
            "tags": run.tags,
            "notes": run.notes,
            "verification": "unverified",
            "export_started_at": datetime.now(timezone.utc).isoformat(),
        },
        "config": dict(run.config),
        "summary": dict(run.summary),
    }
    temporary_path = None
    row_count = 0
    try:
        # File tạm cùng filesystem để replace nguyên tử, kể cả khi chạy đồng thời.
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=project_dir, suffix=".tmp", delete=False
        ) as output:
            temporary_path = Path(output.name)
            header = json.dumps(json_value(snapshot), ensure_ascii=False, allow_nan=False)
            output.write(header[:-1] + ', "history": [\n')
            for row in run.scan_history():
                if row_count:
                    output.write(",\n")
                json.dump(json_value(row), output, ensure_ascii=False, allow_nan=False)
                row_count += 1
            output.write('\n], "export_completed_at": ')
            json.dump(datetime.now(timezone.utc).isoformat(), output)
            output.write("}\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination, row_count


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Download W&B metadata, config, summary, and full metric history."
    )
    parser.add_argument("project", help="W&B project path: ENTITY/PROJECT")
    parser.add_argument("--run-id", action="append", help="Select a run ID; repeat for multiple runs")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent.parent / "runs" / "wandb_exports",
        help="Local output root (default: code/runs/wandb_exports)",
    )
    args = parser.parse_args(argv)
    try:
        entity, project = args.project.split("/")
        project_dir = args.output_dir / path_component(entity) / path_component(project)
        for run_id in args.run_id or []:
            path_component(run_id)
    except ValueError as error:
        parser.error(f"Use ENTITY/PROJECT and valid run IDs. {error}")

    try:
        import wandb
    except ImportError:
        parser.error("Install wandb first: python -m pip install wandb")

    failures = 0
    exported = 0
    try:
        api = wandb.Api(timeout=60)
        if args.run_id:
            runs = dict.fromkeys(args.run_id)
        else:
            runs = api.runs(args.project)
        for selected in runs:
            run_id = selected if args.run_id else selected.id
            try:
                run = api.run(f"{args.project}/{run_id}") if args.run_id else selected
                destination, row_count = export_run(run, project_dir)
                print(f"Saved {run.id}: {row_count} history rows -> {destination}")
                exported += 1
            except Exception as error:
                failures += 1
                print(f"Failed run {run_id}: {error}", file=sys.stderr)
    except Exception as error:
        failures += 1
        print(f"W&B access failed (check wandb login and project path): {error}", file=sys.stderr)
    print(f"Exported {exported} run(s); {failures} failure(s).")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
