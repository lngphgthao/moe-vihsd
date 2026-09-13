"""Evaluate Hugging Face ViHSD classifiers on the test split."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import ClassLabel, load_dataset
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_MODELS = [
    "nd-khoa/vihsd-uit-visobert",
    "nd-khoa/vihsd-uit-visobert-v2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="uitnlp/vihsd")
    parser.add_argument("--split", default="test")
    parser.add_argument("--text-column", default="free_text")
    parser.add_argument("--label-column", default="label_id")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("models", nargs="*", default=DEFAULT_MODELS)
    return parser.parse_args()


def get_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device_name)


def get_label_names(dataset, label_column: str) -> list[str]:
    feature = dataset.features.get(label_column)
    if isinstance(feature, ClassLabel):
        return feature.names
    return [str(value) for value in sorted(set(dataset[label_column]))]


def evaluate_model(model_name: str, dataset, label_names: list[str], args, device: torch.device) -> str:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    def tokenize(batch):
        return tokenizer(
            batch[args.text_column],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )

    tokenized = dataset.map(tokenize, batched=True, desc=f"Tokenizing for {model_name}")
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", args.label_column])
    loader = DataLoader(tokenized, batch_size=args.batch_size)

    predictions = []
    labels = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"Evaluating {model_name}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
            labels.extend(batch[args.label_column].tolist())

    target_names = label_names[: model.config.num_labels]
    report = classification_report(
        labels,
        predictions,
        labels=list(range(model.config.num_labels)),
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
    return f"\nModel: {model_name}\nDevice: {device}\n{report}"


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    raw_dataset = load_dataset(args.dataset, split=args.split)
    required_columns = {args.text_column, args.label_column}
    missing_columns = required_columns.difference(raw_dataset.column_names)
    if missing_columns:
        raise KeyError(
            f"Missing columns {sorted(missing_columns)}. Available columns: {raw_dataset.column_names}"
        )

    label_names = get_label_names(raw_dataset, args.label_column)
    reports = [evaluate_model(model_name, raw_dataset, label_names, args, device) for model_name in args.models]
    output = "\n".join(reports)
    print(output)
    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")
        print(f"Saved report to {args.output}")


if __name__ == "__main__":
    main()