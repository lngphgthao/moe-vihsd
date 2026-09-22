"""ViHSD/Hugging Face dataset loading and PyTorch DataLoader preparation.

Supports two input modes controlled by ``dataset.input_mode`` in the config:
- ``"single"`` (default): single text column → ``tokenizer(text, ...)``
- ``"nli"``: premise + hypothesis columns → ``tokenizer(premise, hypothesis, ...)``
  with automatic string-to-int label mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from datasets import ClassLabel, DatasetDict, load_dataset
from datasets import config as datasets_config
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# `datasets` checks the optional torchvision VideoReader path even for text-only
# workloads. Some Colab installs expose torchvision without the required video API,
# which breaks dataset torch-formatting. Disable that optional branch when it is not
# available so the text pipeline still works.
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
    num_labels: int


def _ensure_splits(dataset: DatasetDict, config: dict) -> DatasetDict:
    train_name = config["train_split"]
    validation_name = config["validation_split"]
    test_name = config["test_split"]
    if validation_name not in dataset:
        split = dataset[train_name].train_test_split(test_size=0.1, seed=42)
        dataset = DatasetDict({**dataset, train_name: split["train"], validation_name: split["test"]})
    if test_name not in dataset:
        dataset = DatasetDict({**dataset, test_name: dataset[validation_name]})
    return dataset


def _label_info(dataset, label_column: str, label_names_override: list[str] | None = None) -> tuple[list[str], dict[Any, int]]:
    """Return (label_names, label_to_id mapping).

    Priority:
    1. ``label_names_override`` from config (used for NLI string labels)
    2. HuggingFace ``ClassLabel`` feature
    3. Auto-sorted unique values in the column
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
    """Load the raw dataset from disk or HuggingFace Hub.

    For ViHSD, the segmented on-disk copy is preferred (data/vihsd_segmented).
    For HuggingFace Hub datasets (e.g. uitnlp/ViANLI), loading from disk is
    skipped and the dataset is fetched from the Hub directly.
    """
    import os
    from pathlib import Path

    dataset_name = config["name"]

    # --- Local disk paths (used for pre-segmented copies) ---
    # Try the exact name as a directory first (works if it is a local path like
    # "data/vihsd_segmented" or "data/vianli_segmented").
    if os.path.isdir(dataset_name):
        from datasets import load_from_disk
        print(f"Loading preprocessed dataset from '{dataset_name}'...")
        return load_from_disk(dataset_name)

    # Legacy fallback paths for ViHSD on Kaggle / local
    kaggle_path = "/kaggle/input/vihsd-segmented"
    local_path = "data/vihsd_segmented"
    repo_path = str(Path(__file__).resolve().parent.parent / "data" / "vihsd_segmented")

    # Only apply the ViHSD-specific fallback when the config actually targets
    # the ViHSD segmented dataset (avoid accidentally loading the wrong data).
    if "vihsd" in dataset_name.lower():
        for path in [local_path, repo_path, kaggle_path]:
            if os.path.isdir(path):
                from datasets import load_from_disk
                print(f"Loading preprocessed dataset from '{path}'...")
                return load_from_disk(path)

    # --- HuggingFace Hub ---
    dataset_config = config.get("config")
    kwargs = {} if dataset_config is None else {"name": dataset_config}
    print(f"Loading dataset '{dataset_name}' from HuggingFace Hub...")
    return load_dataset(dataset_name, **kwargs)


def prepare_data(config: dict) -> DatasetBundle:
    input_mode = config.get("input_mode", "single")
    if input_mode not in ("single", "nli"):
        raise ValueError(f"Unknown dataset.input_mode '{input_mode}'. Must be 'single' or 'nli'.")

    raw = _load_raw(config)
    raw = _ensure_splits(raw, config)

    label_column = config["label_column"]
    label_names_override = config.get("label_names")  # list of strings for NLI

    # --- Validate columns exist ---
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

    # --- Filter missing texts ---
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

    label_names, label_to_id = _label_info(
        raw[config["train_split"]], label_column, label_names_override
    )
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"])

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
            # Produces [CLS] premise [SEP] hypothesis [SEP] automatically.
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
    tokenized.set_format(type="torch", columns=keep_columns)

    limit = config.get("max_train_samples")
    if limit:
        train_name = config["train_split"]
        tokenized[train_name] = tokenized[train_name].select(range(min(limit, len(tokenized[train_name]))))
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
    return DatasetBundle(tokenizer, loaders, label_names, len(label_names))
