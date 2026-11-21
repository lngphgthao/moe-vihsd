"""Đếm phân phối nhãn trong các split của dataset theo config."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import _ensure_splits, _label_info, _load_raw
from src.utils import load_config


def count_labels(config_path: str | Path) -> dict[str, object]:
    """Đếm số lượng và tỷ lệ từng nhãn trong train, validation và test."""
    config = load_config(config_path)
    raw = _ensure_splits(_load_raw(config["dataset"]), config["dataset"])
    dataset_config = config["dataset"]
    label_column = dataset_config["label_column"]
    label_names, label_to_id = _label_info(
        raw[dataset_config["train_split"]],
        label_column,
        dataset_config.get("label_names"),
    )

    split_names = {
        "train": dataset_config["train_split"],
        "dev": dataset_config["validation_split"],
        "test": dataset_config["test_split"],
    }
    result: dict[str, object] = {
        "label_mapping": {str(name): index for name, index in label_to_id.items()},
        "splits": {},
    }

    for output_name, split_name in split_names.items():
        values = raw[split_name][label_column]
        unknown = sorted({value for value in values if value not in label_to_id}, key=str)
        if unknown:
            raise ValueError(
                f"Unknown labels in split {split_name!r}: {unknown}. "
                f"Expected: {label_names}"
            )
        counts = Counter(values)
        total = len(values)
        result["splits"][output_name] = {
            "source_split": split_name,
            "total": total,
            "labels": {
                str(label_names[index]): {
                    "id": index,
                    "count": sum(count for value, count in counts.items() if label_to_id[value] == index),
                    "percentage": round(100 * sum(count for value, count in counts.items() if label_to_id[value] == index) / total, 4)
                    if total
                    else 0.0,
                }
                for index in range(len(label_names))
            },
        }

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Count dataset labels by split.")
    parser.add_argument("--config", default="configs/base/vianli.yaml")
    parser.add_argument("--json", action="store_true", help="Print results as JSON")
    args = parser.parse_args()

    result = count_labels(args.config)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print("Label mapping:", result["label_mapping"])
    for split_name, split in result["splits"].items():
        print(f"\n{split_name} ({split['source_split']}): {split['total']} examples")
        for label_name, values in split["labels"].items():
            print(
                f"  {values['id']}: {label_name} = "
                f"{values['count']} ({values['percentage']:.2f}%)"
            )


if __name__ == "__main__":
    main()
