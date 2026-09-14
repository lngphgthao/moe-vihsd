"""Train the original dense PhoBERT baseline independently from the MoE pipeline."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm.auto import tqdm

from data.vihsd import prepare_data
from metrics import compute_classification_metrics
from models.dense_phobert import DensePhoBERTClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, optimizer, device, label_names, training):
    model.train(training)
    total_loss = 0.0
    total_examples = 0
    all_preds = []
    all_labels = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in tqdm(loader, desc="train" if training else "eval", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids, attention_mask)
            loss = F.cross_entropy(logits, labels)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            total_examples += labels.size(0)
            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    metrics = compute_classification_metrics(all_labels, all_preds, label_names=label_names)
    metrics["loss"] = total_loss / total_examples if total_examples else 0.0
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/vihsd.yaml")
    parser.add_argument("--run-id", default="dense-phobert")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    set_seed(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_config = {**config["dataset"], **config["training"]}
    bundle = prepare_data(data_config)
    model_config = {**config["model"], "pooling": "mean"}
    model = DensePhoBERTClassifier(
        bundle.tokenizer.vocab_size,
        bundle.num_labels,
        model_config,
    ).to(device)
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    output_dir = Path(config["paths"]["results_dir"]) / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    best_macro_f1 = -1.0
    history = []
    for epoch in range(int(config["training"]["epochs"])):
        train_metrics = run_epoch(
            model, bundle.loaders["train"], optimizer, device, bundle.label_names, True
        )
        validation_metrics = run_epoch(
            model, bundle.loaders["validation"], None, device, bundle.label_names, False
        )
        record = {
            "epoch": epoch + 1,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        print(json.dumps(record, indent=2))
        if validation_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = validation_metrics["macro_f1"]
            torch.save(model.state_dict(), output_dir / "dense_phobert_best.pt")

    model.load_state_dict(torch.load(output_dir / "dense_phobert_best.pt", map_location=device))
    test_metrics = run_epoch(
        model, bundle.loaders["test"], None, device, bundle.label_names, False
    )
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    print(json.dumps({"run_id": args.run_id, "test": test_metrics}, indent=2))


if __name__ == "__main__":
    main()