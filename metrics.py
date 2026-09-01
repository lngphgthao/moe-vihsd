"""Classification metric utilities for ViHSD evaluation and training."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score


def compute_classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    label_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compute accuracy, macro F1, weighted F1, per-class F1, and report dict."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if len(y_true_arr) == 0:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "per_class_f1": {},
            "classification_report": {},
        }

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0))

    target_names = [str(name) for name in label_names] if label_names is not None else None
    labels = list(range(len(label_names))) if label_names is not None else sorted(set(y_true_arr) | set(y_pred_arr))

    per_class_scores = f1_score(y_true_arr, y_pred_arr, labels=labels, average=None, zero_division=0)
    per_class_dict = {
        (target_names[i] if target_names and i < len(target_names) else str(labels[i])): float(per_class_scores[i])
        for i in range(len(labels))
    }

    report_dict = classification_report(
        y_true_arr,
        y_pred_arr,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_f1": per_class_dict,
        "classification_report": report_dict,
    }


def format_classification_report(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    label_names: Sequence[str] | None = None,
    digits: int = 4,
) -> str:
    """Return a formatted text classification report table."""
    target_names = [str(name) for name in label_names] if label_names is not None else None
    labels = list(range(len(label_names))) if label_names is not None else None
    return classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        digits=digits,
        zero_division=0,
    )

