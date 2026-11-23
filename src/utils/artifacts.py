"""Artifact bền vững, provenance và tracker cho run thuộc scope hiện hành."""
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

import yaml
from filelock import FileLock

ROOT = Path(__file__).resolve().parents[2]
TRACKER = ROOT / "experiments/experiment_tracker.csv"
FIELDS = ["experiment_id", "run_id", "status", "verification", "wandb_sync", "profile",
          "dataset", "backbone", "architecture", "variant", "seed", "config_path", "git_commit",
          "artifact_uri", "wandb_url", "val_macro_f1", "val_accuracy", "val_weighted_f1", "val_loss",
          "test_macro_f1", "test_accuracy", "test_weighted_f1", "notes"]


def now():
    return datetime.now(timezone.utc).isoformat()


def sanitized(value):
    if isinstance(value, dict):
        return {k: "[REDACTED]" if re.search(r"(^|[_.])(token|password|secret|api_key|credentials?|private_key)$", str(k), re.I)
                else sanitized(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitized(v) for v in value]
    if isinstance(value, str):
        for key, secret in os.environ.items():
            if secret and len(secret) >= 8 and re.search(r"(^|_)(token|password|secret|api_key)$", key, re.I):
                value = value.replace(secret, "[REDACTED]")
    return value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(sanitized(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(sanitized(value), ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def tracker_rows(path=TRACKER, *, timeout=30):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # OS lock is released even if a worker exits; the .lock file may remain.
    with FileLock(str(path.with_suffix(".lock")), timeout=timeout):
        rows = []
        if path.exists():
            with path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
        yield rows
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)


def reserve_run(config, run_id=None, *, tracker=TRACKER):
    experiment_id = config.get("experiment_id", "")
    if not re.fullmatch(r"exp-\d{3,}", experiment_id) or int(experiment_id[4:]) <= 2:
        raise ValueError("experiment_id must have the form exp-003")
    if run_id is not None and run_id != "auto" and (not re.fullmatch(r"run-\d{3,}", run_id) or int(run_id[4:]) <= 2):
        raise ValueError("Use a new run-NNN ID after historical run-001/002")
    if tracker is None:
        if run_id in (None, "auto"):
            raise ValueError("CSV tracker disabled: pass an explicit --run-id run-NNN")
        return run_id
    with tracker_rows(tracker) as rows:
        if run_id in (None, "auto"):
            # run-001/002 are historical even when the active tracker is empty.
            used = [2] + [int(r["run_id"][4:]) for r in rows if re.fullmatch(r"run-\d+", r["run_id"])]
            run_root = Path(os.getenv("RUNS_DIR", config.get("paths", {}).get("runs_dir", "runs")))
            if run_root.exists():
                used.extend(int(path.name[4:]) for path in run_root.iterdir()
                            if re.fullmatch(r"run-\d+", path.name))
            candidate = max(used) + 1
            while (run_root / f"run-{candidate:03d}").exists():
                candidate += 1
            run_id = f"run-{candidate:03d}"
        if not re.fullmatch(r"run-\d{3,}", run_id) or int(run_id[4:]) <= 2:
            raise ValueError("Use a new run-NNN ID after historical run-001/002")
        existing = next((row for row in rows if row["run_id"] == run_id), None)
        if existing:
            if existing["status"] != "planned" or existing["experiment_id"] != experiment_id or str(existing["seed"]) != str(config["seed"]):
                raise ValueError(f"Run ID already reserved or executed: {run_id}")
        else:
            row = dict.fromkeys(FIELDS, "")
            row.update(experiment_id=experiment_id, run_id=run_id, status="planned", verification="unverified",
                       seed=config["seed"], dataset=config["dataset"].get("id", config["dataset"]["name"]),
                       backbone=config["model"]["backbone"], architecture="dense_encoder",
                       variant=config["model"]["head"]["type"])
            rows.append(row)
    return run_id


def update_tracker(run_id, values, *, tracker=TRACKER):
    if tracker is None:
        return
    values = dict(values)
    dataset = values.get("dataset")
    if isinstance(dataset, dict):
        values["dataset"] = dataset.get("id", dataset.get("name"))
    model = values.get("model")
    if isinstance(model, dict):
        values.update(backbone=model.get("backbone"), architecture="dense_encoder",
                      variant=model.get("head", {}).get("type"))
    with tracker_rows(tracker) as rows:
        row = next(row for row in rows if row["run_id"] == run_id)
        row.update({k: v if v is not None else "" for k, v in values.items() if k in FIELDS})


def git_output(*args):
    result = subprocess.run(["git", "-c", f"safe.directory={ROOT.as_posix()}", *args], cwd=ROOT,
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
    return result.stdout.strip() if result.returncode == 0 else None


def config_fingerprint(config):
    selected = {k: config[k] for k in ("dataset", "model", "training", "routing")}
    return hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()


class Tee:
    def __init__(self, original, log):
        self.original, self.log = original, log
    def __getattr__(self, name):
        # Libraries query stream metadata (isatty, encoding, fileno) while logging.
        return getattr(self.original, name)
    def write(self, value):
        consumed = len(value)
        value = sanitized(value)
        self.original.write(value)
        self.log.write(value)
        self.log.flush()
        return consumed
    def flush(self):
        self.original.flush()
        self.log.flush()


class RunArtifacts:
    def __init__(self, config, run_id, config_path, overrides, *, tracker=TRACKER):
        self.tracker = tracker if config.get("logging", {}).get("use_tracker", True) else None
        self.run_id = reserve_run(config, run_id, tracker=self.tracker)
        run_root = Path(os.getenv("RUNS_DIR", config.get("paths", {}).get("runs_dir", "runs")))
        run_root.mkdir(parents=True, exist_ok=True)
        self.path = run_root / self.run_id
        self.path.mkdir(exist_ok=False)
        for directory in ("checkpoints", "evaluation", "diagnostics"):
            (self.path / directory).mkdir()
        self.config = config
        (self.path / "resolved_config.yaml").write_text(yaml.safe_dump(sanitized(config), allow_unicode=True, sort_keys=False), encoding="utf-8")
        commit = git_output("rev-parse", "HEAD")
        dirty = git_output("status", "--porcelain")
        packages = {}
        for name in ("torch", "transformers", "datasets", "safetensors", "wandb", "vncorenlp", "filelock"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        self.manifest = dict(schema_version=1, experiment_id=config["experiment_id"], run_id=self.run_id,
                             status="running", verification="unverified", profile=config["profile"], seed=config["seed"],
                             csv_tracker_enabled=self.tracker is not None,
                             started_at=now(), finished_at=None, config_path=str(config_path), cli_overrides=[{key: value} for key, value in (item.split("=", 1) for item in overrides)],
                             git_commit=commit, working_tree_dirty=bool(dirty) if dirty is not None else None,
                             config_fingerprint=config_fingerprint(config), dataset=config["dataset"], model=config["model"],
                             python=platform.python_version(), platform=platform.platform(), libraries=packages,
                             artifact_uri=self.path.resolve().as_uri(), wandb_url=None,
                             wandb_sync="pending" if config.get("logging", {}).get("use_wandb") else "disabled")
        # Snapshot only reviewed source, never .env, raw data or local outputs.
        if dirty:
            snapshot = self.path / "source"
            sources = [ROOT / "train.py", ROOT / "evaluate.py", ROOT / "requirements.txt"]
            sources += list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").glob("*.py"))
            for source in sources:
                target = snapshot / source.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(sanitized(source.read_text(encoding="utf-8")), encoding="utf-8")
        self.save_manifest()

    def save_manifest(self):
        write_json(self.path / "manifest.json", self.manifest)
        update_tracker(self.run_id, self.manifest, tracker=self.tracker)

    def metric(self, split, metrics, epoch, step):
        append_jsonl(self.path / "metrics.jsonl", dict(split=split, epoch=epoch, global_step=step, time=now(), **metrics))

    @contextmanager
    def console(self):
        stdout, stderr = sys.stdout, sys.stderr
        with (self.path / "console.log").open("a", encoding="utf-8") as stream:
            sys.stdout, sys.stderr = Tee(stdout, stream), Tee(stderr, stream)
            try:
                yield
            finally:
                sys.stdout, sys.stderr = stdout, stderr
