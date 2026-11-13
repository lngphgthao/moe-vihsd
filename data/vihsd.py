"""Wrapper tương thích ngược cho shared dataset loader."""

from src.data.loader import DatasetBundle, _ensure_splits, _label_info, prepare_data

__all__ = ["DatasetBundle", "_ensure_splits", "_label_info", "prepare_data"]
