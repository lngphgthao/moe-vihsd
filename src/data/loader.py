"""Load dataset ViHSD/Hugging Face và chuẩn bị PyTorch DataLoader.

Hỗ trợ hai input mode, điều khiển bởi ``dataset.input_mode`` trong config:
- ``"single"`` (mặc định): một text column → ``tokenizer(text, ...)``
- ``"nli"``: premise + hypothesis column → ``tokenizer(premise, hypothesis, ...)``
  và tự động map label từ string sang int.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from datasets import ClassLabel, DatasetDict, load_dataset
from datasets import config as datasets_config
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# `datasets` kiểm tra optional torchvision VideoReader kể cả với workload chỉ có text.
# Một số môi trường Colab có torchvision nhưng thiếu video API cần thiết, làm hỏng
# torch-formatting của dataset. Tắt nhánh optional này khi API không khả dụng.
try:
    import torchvision  # noqa: F401
    from torchvision.io import VideoReader  # noqa: F401
except Exception:
    datasets_config.TORCHVISION_AVAILABLE = False


@dataclass
class DatasetBundle:
    tokenizer: Any
    loaders: dict[str, DataLoader]
    label_names: list[str]
    label_mapping: dict[Any, int]
    num_labels: int


def _ensure_splits(dataset: DatasetDict, config: dict) -> DatasetDict:
    train_name = config["train_split"]
    validation_name = config["validation_split"]
    test_name = config["test_split"]
    if len({train_name, validation_name, test_name}) != 3:
        raise ValueError("train_split, validation_split, and test_split must be distinct")
    for split_name in config.get("expected_split_sizes", {}):
        if split_name not in dataset:
            raise ValueError(f"Expected official split '{split_name}' is missing")

    if validation_name not in dataset:
        split = dataset[train_name].train_test_split(test_size=0.1, seed=42)
        dataset = DatasetDict({**dataset, train_name: split["train"], validation_name: split["test"]})
    if test_name not in dataset:
        raise ValueError(
            f"Required test split '{test_name}' is missing. Refusing to reuse the validation "
            "split as test data; provide an independent test split."
        )
    return dataset


def _label_info(dataset, label_column: str, label_names_override: list[str] | None = None) -> tuple[list[str], dict[Any, int]]:
    """Trả về (label_names, label_to_id mapping).

    Thứ tự ưu tiên:
    1. ``label_names_override`` từ config, dùng cho NLI string label.
    2. Feature ``ClassLabel`` của HuggingFace.
    3. Unique value trong column sau khi sort.
    """
    if label_names_override:
        label_to_id = {name: idx for idx, name in enumerate(label_names_override)}
        return label_names_override, label_to_id
    feature = dataset.features.get(label_column)
    if isinstance(feature, ClassLabel):
        return feature.names, {index: index for index in range(feature.num_classes)}
    values = sorted(set(dataset[label_column]))
    return [str(value) for value in values], {value: index for index, value in enumerate(values)}


def _load_raw(config: dict) -> DatasetDict:
    """Load raw dataset từ disk hoặc HuggingFace Hub.

    Với ViHSD, ưu tiên bản segmented trên disk (data/vihsd_segmented).
    Với dataset từ HuggingFace Hub, ví dụ uitnlp/ViANLI, load trực tiếp từ Hub
    khi không có DatasetDict local.
    """
    import os
    from pathlib import Path

    dataset_name = config["name"]

    # --- Local disk path, dùng cho bản copy đã segment ---
    # Thử chính xác dataset name trước, nếu đó là local path như
    # "data/vihsd_segmented" hoặc "data/vianli_segmented".
    if os.path.isdir(dataset_name):
        from datasets import load_from_disk
        print(f"Loading preprocessed dataset from '{dataset_name}'...")
        return load_from_disk(dataset_name)

    # Legacy fallback path cho ViHSD trên Kaggle hoặc local.
    kaggle_path = "/kaggle/input/vihsd-segmented"
    local_path = "data/vihsd_segmented"
    repo_path = str(Path(__file__).resolve().parent.parent / "data" / "vihsd_segmented")

    # Chỉ dùng fallback riêng cho ViHSD khi config thực sự trỏ tới dataset
    # ViHSD segmented, tránh load nhầm dữ liệu.
    if "vihsd" in dataset_name.lower():
        for path in [local_path, repo_path, kaggle_path]:
            if os.path.isdir(path):
                from datasets import load_from_disk
                print(f"Loading preprocessed dataset from '{path}'...")
                return load_from_disk(path)

    # --- HuggingFace Hub ---
    dataset_config = config.get("config")
    kwargs = {} if dataset_config is None else {"name": dataset_config}
    if config.get("revision") is not None:
        kwargs["revision"] = config["revision"]
    print(f"Loading dataset '{dataset_name}' from HuggingFace Hub...")
    return load_dataset(dataset_name, **kwargs)


def prepare_data(config: dict) -> DatasetBundle:
    input_mode = config.get("input_mode", "single")
    if input_mode not in ("single", "nli"):
        raise ValueError(f"Unknown dataset.input_mode '{input_mode}'. Must be 'single' or 'nli'.")

    raw = _load_raw(config)
    raw = _ensure_splits(raw, config)

    label_column = config["label_column"]
    label_names_override = config.get("label_names")  # list string label cho NLI.

    # --- Kiểm tra column tồn tại ---
    train_columns = raw[config["train_split"]].column_names
    if label_column not in train_columns:
        raise KeyError(f"Label column {label_column!r} not found in {train_columns}")
    if input_mode == "single":
        text_column = config["text_column"]
        if text_column not in train_columns:
            raise KeyError(f"Text column {text_column!r} not found in {train_columns}")
    else:  # nli
        premise_column = config["premise_column"]
        hypothesis_column = config["hypothesis_column"]
        for col in (premise_column, hypothesis_column):
            if col not in train_columns:
                raise KeyError(f"NLI column {col!r} not found in {train_columns}")

    # --- Lọc text bị thiếu ---
    if input_mode == "single":
        raw = raw.filter(
            lambda example: example[text_column] is not None,
            desc="Removing examples with missing text",
        )
    else:
        raw = raw.filter(
            lambda example: example[premise_column] is not None and example[hypothesis_column] is not None,
            desc="Removing examples with missing premise/hypothesis",
        )

    for split_name, expected_size in config.get("expected_split_sizes", {}).items():
        if len(raw[split_name]) != expected_size:
            raise ValueError(
                f"Expected {expected_size} examples in '{split_name}' after filtering, "
                f"got {len(raw[split_name])}"
            )

    label_names, label_to_id = _label_info(
        raw[config["train_split"]], label_column, label_names_override
    )
    tokenizer_kwargs = {}
    if config.get("tokenizer_revision") is not None:
        tokenizer_kwargs["revision"] = config["tokenizer_revision"]
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"], **tokenizer_kwargs)

    # --- Tokenization ---
    if input_mode == "single":
        def tokenize(batch):
            encoded = tokenizer(
                batch[text_column],
                truncation=True,
                padding="max_length",
                max_length=config["max_length"],
            )
            encoded["labels"] = [label_to_id[value] for value in batch[label_column]]
            return encoded
    else:
        def tokenize(batch):
            # Tự động tạo [CLS] premise [SEP] hypothesis [SEP].
            encoded = tokenizer(
                batch[premise_column],
                batch[hypothesis_column],
                truncation=True,
                padding="max_length",
                max_length=config["max_length"],
            )
            encoded["labels"] = [label_to_id[value] for value in batch[label_column]]
            return encoded

    tokenized = raw.map(
        tokenize,
        batched=True,
        num_proc=config.get("tokenization_num_proc"),
        load_from_cache_file=True,
        desc="Tokenizing dataset",
    )
    keep_columns = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in tokenized[config["train_split"]].column_names:
        keep_columns.append("token_type_ids")
    tokenized.set_format(type="torch", columns=keep_columns)

    limit = config.get("max_train_samples")
    if limit:
        train_name = config["train_split"]
        limit = min(int(limit), len(tokenized[train_name]))
        if config.get("train_subset_strategy") == "stratified":
            labels = tokenized[train_name]["labels"]
            selected_indices = []
            class_indices = {
                class_id: [index for index, label in enumerate(labels) if int(label) == class_id]
                for class_id in range(len(label_names))
            }
            quota, remainder = divmod(limit, len(label_names))
            for class_id in range(len(label_names)):
                class_quota = quota + (1 if class_id < remainder else 0)
                selected_indices.extend(class_indices[class_id][:class_quota])
            if len(selected_indices) < limit:
                selected_set = set(selected_indices)
                selected_indices.extend(
                    index for index in range(len(labels)) if index not in selected_set
                )
                selected_indices = selected_indices[:limit]
            tokenized[train_name] = tokenized[train_name].select(selected_indices)
        else:
            tokenized[train_name] = tokenized[train_name].select(range(limit))
    eval_limit = config.get("max_eval_samples")
    if eval_limit:
        for split_name in [config["validation_split"], config["test_split"]]:
            if split_name in tokenized:
                tokenized[split_name] = tokenized[split_name].select(range(min(eval_limit, len(tokenized[split_name]))))

    loaders = {
        "train": DataLoader(tokenized[config["train_split"]], batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"]),
        "validation": DataLoader(tokenized[config["validation_split"]], batch_size=config["batch_size"], num_workers=config["num_workers"]),
        "test": DataLoader(tokenized[config["test_split"]], batch_size=config["batch_size"], num_workers=config["num_workers"]),
    }
    return DatasetBundle(tokenizer, loaders, label_names, label_to_id, len(label_names))
