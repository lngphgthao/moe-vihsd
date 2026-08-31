"""Classification metrics shared by training and evaluation."""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def classification_metrics(labels: Sequence[int], predictions: Sequence[int], label_names: Sequence[str]) -> dict:
    """Return JSON-serializable overall and per-class classification metrics."""
    label_ids = list(range(len(label_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=label_ids, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1.mean()),
        "per_class": {
            label_name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label_name in enumerate(label_names)
        },
        "confusion_matrix": confusion_matrix(labels, predictions, labels=label_ids).tolist(),
    }
