"""Backward-compatibility wrapper for src.dataset."""

from src.dataset import DatasetBundle, _ensure_splits, _label_info, prepare_data

__all__ = ["DatasetBundle", "_ensure_splits", "_label_info", "prepare_data"]
