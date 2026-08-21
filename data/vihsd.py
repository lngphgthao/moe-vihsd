"""ViHSD/Hugging Face dataset loading and PyTorch DataLoader preparation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from datasets import ClassLabel, DatasetDict, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


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


def _label_info(dataset, label_column: str) -> tuple[list[str], dict[Any, int]]:
    feature = dataset.features.get(label_column)
    if isinstance(feature, ClassLabel):
        return feature.names, {index: index for index in range(feature.num_classes)}
    values = sorted(set(dataset[label_column]))
    return [str(value) for value in values], {value: index for index, value in enumerate(values)}


def prepare_data(config: dict) -> DatasetBundle:
    dataset_config = config["config"]
    kwargs = {} if dataset_config is None else {"name": dataset_config}
    raw = load_dataset(config["name"], **kwargs)
    raw = _ensure_splits(raw, config)
    text_column = config["text_column"]
    label_column = config["label_column"]
    if text_column not in raw[config["train_split"]].column_names:
        raise KeyError(f"Text column {text_column!r} not found in {raw[config['train_split']].column_names}")
    if label_column not in raw[config["train_split"]].column_names:
        raise KeyError(f"Label column {label_column!r} not found in {raw[config['train_split']].column_names}")

    label_names, label_to_id = _label_info(raw[config["train_split"]], label_column)
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"])

    def tokenize(batch):
        encoded = tokenizer(batch[text_column], truncation=True, padding="max_length", max_length=config["max_length"])
        encoded["labels"] = [label_to_id[value] for value in batch[label_column]]
        return encoded

    tokenized = raw.map(tokenize, batched=True)
    keep_columns = ["input_ids", "attention_mask", "labels"]
    tokenized.set_format(type="torch", columns=keep_columns)
    limit = config.get("max_train_samples")
    if limit:
        train_name = config["train_split"]
        tokenized[train_name] = tokenized[train_name].select(range(min(limit, len(tokenized[train_name]))))

    loaders = {
        "train": DataLoader(tokenized[config["train_split"]], batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"]),
        "validation": DataLoader(tokenized[config["validation_split"]], batch_size=config["batch_size"], num_workers=config["num_workers"]),
        "test": DataLoader(tokenized[config["test_split"]], batch_size=config["batch_size"], num_workers=config["num_workers"]),
    }
    return DatasetBundle(tokenizer, loaders, label_names, len(label_names))
