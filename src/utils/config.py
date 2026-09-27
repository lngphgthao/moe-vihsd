"""General utility helpers for configuration, run tracking, and reproducibility."""

from __future__ import annotations

import os
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    """Set random seed across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_output_path(configured_path: str, environment_name: str) -> Path:
    """Resolve an output directory from an environment variable override or config default."""
    return Path(os.getenv(environment_name, configured_path)).expanduser()


def deep_merge_dict(base: dict, overlay: dict) -> dict:
    """Recursively merge two dictionaries."""
    merged = base.copy()
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path) -> dict:
    """Load a YAML config, resolving optional ``base_config`` parent overlays."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base_ref = raw.pop("base_config", None)
    if base_ref:
        base_path = (path.parent / base_ref).resolve()
        if not base_path.exists():
            base_path = Path(base_ref).resolve()
        base_dict = load_config(base_path)
        return deep_merge_dict(base_dict, raw)
    return raw


def flatten_hyperparameters(values: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Return nested configuration values as stable dotted keys for comparison."""
    flattened = {}
    for key, value in values.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(flatten_hyperparameters(value, full_key))
        else:
            flattened[full_key] = value
    return flattened


def create_run_id(smoke_test: bool, requested_run_id: str | None = None) -> str:
    """Generate a timestamped run ID in Asia/Ho_Chi_Minh timezone, or return the requested ID."""
    if requested_run_id:
        return requested_run_id
    hanoi_timezone = timezone(timedelta(hours=7), name="Asia/Ho_Chi_Minh")
    timestamp = datetime.now(hanoi_timezone).strftime("%Y%m%dT%H%M%S")
    profile = "smoke" if smoke_test else "full"
    return f"{timestamp}-{profile}"


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    """Apply ``section.key=value`` overrides parsed as YAML scalars."""
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid --set value {override!r}; expected section.key=value.")
        key_path, raw_value = override.split("=", 1)
        keys = key_path.split(".")
        if not key_path or any(not key for key in keys):
            raise ValueError(f"Invalid configuration key {key_path!r}.")
        target = config
        for key in keys[:-1]:
            if key not in target or not isinstance(target[key], dict):
                raise KeyError(f"Unknown configuration section {key_path!r}.")
            target = target[key]
        if keys[-1] not in target:
            raise KeyError(f"Unknown configuration key {key_path!r}.")
        target[keys[-1]] = yaml.safe_load(raw_value)
    return config


def find_run_checkpoint(run_dir: Path, architecture: str) -> Path:
    """Find a checkpoint while supporting both new and legacy run folders."""
    candidates = [
        run_dir / f"{architecture}_best.safetensors",
        run_dir / "vihsd_moe_best.safetensors",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    discovered = sorted(run_dir.glob("*_best.safetensors"))
    if len(discovered) == 1:
        return discovered[0]
    if not discovered:
        raise FileNotFoundError(f"No checkpoint found in run directory: {run_dir}")
    names = ", ".join(path.name for path in discovered)
    raise RuntimeError(f"Multiple checkpoints found in {run_dir}; choose one with --checkpoint: {names}")
