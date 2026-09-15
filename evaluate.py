"""Evaluate a saved ViHSD checkpoint and write JSON predictions."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure repository root is on sys.path for Colab execution
REPO_ROOT = str(Path(__file__).resolve().parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file
from tqdm.auto import tqdm

from models.factory import build_model, standardize_model_output
from src.dataset import prepare_data
from src.metrics import compute_classification_metrics, format_classification_report
from src.utils import find_run_checkpoint, load_config, resolve_output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/vihsd.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--run-id", default=None, help="Evaluate a specific run directory.")
    args = parser.parse_args()
    config = load_config(args.config)
    checkpoint_root = resolve_output_path(config["paths"]["checkpoint_dir"], "CHECKPOINT_DIR")
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    elif args.run_id:
        architecture = str(config.get("model", {}).get("architecture", "phobert_moe"))
        checkpoint_path = find_run_checkpoint(checkpoint_root / args.run_id, architecture)
    else:
        latest_path = checkpoint_root / "latest_run.json"
        if not latest_path.exists():
            raise FileNotFoundError("No latest run found. Train first or pass --checkpoint/--run-id.")
        checkpoint_path = Path(json.loads(latest_path.read_text(encoding="utf-8"))["checkpoint"])
    resolved_config_path = checkpoint_path.parent / "resolved_config.yaml"
    if resolved_config_path.exists():
        config = load_config(resolved_config_path)
        print(f"Using run configuration: {resolved_config_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = prepare_data({**config["dataset"], **config["training"]})
    model_config = {**config["model"], "pad_token_id": bundle.tokenizer.pad_token_id or 0}
    model_config.setdefault("architecture", "phobert_moe")
    model = build_model(model_config, bundle.tokenizer.vocab_size, bundle.num_labels).to(device)
    model.load_state_dict(load_file(str(checkpoint_path), device=str(device)))
    model.eval()
    predictions = []
    all_preds = []
    all_labels = []
    total_loss = 0.0
    total_examples = 0
    routing_counts = torch.zeros(getattr(model, "num_experts", 0), dtype=torch.long)
    with torch.no_grad():
        for batch in tqdm(bundle.loaders["test"], desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits, aux = standardize_model_output(model(input_ids, attention_mask))
            predicted = logits.argmax(dim=-1)
            total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
            total_examples += labels.numel()
            if "top_indices" in aux and routing_counts.numel() > 0:
                routing_counts += torch.bincount(aux["top_indices"].reshape(-1).cpu(), minlength=model.num_experts)
            preds_cpu = predicted.cpu().tolist()
            labels_cpu = labels.cpu().tolist()
            all_preds.extend(preds_cpu)
            all_labels.extend(labels_cpu)
            predictions.extend({"prediction": int(p), "label": int(l)} for p, l in zip(preds_cpu, labels_cpu))
    cls_metrics = compute_classification_metrics(all_labels, all_preds, label_names=bundle.label_names)
    results = {
        "loss": (total_loss / total_examples) if total_examples > 0 else 0.0,
        "accuracy": cls_metrics["accuracy"],
        "macro_f1": cls_metrics["macro_f1"],
        "weighted_f1": cls_metrics["weighted_f1"],
        "per_class_f1": cls_metrics["per_class_f1"],
        "classification_report": cls_metrics["classification_report"],
        "label_names": bundle.label_names,
        "routing_counts": routing_counts.tolist(),
        "predictions": predictions,
    }
    results_root = resolve_output_path(config["paths"]["results_dir"], "RESULTS_DIR")
    run_id = checkpoint_path.parent.name
    results_dir = results_root / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "vihsd_predictions.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Test loss: {results['loss']:.4f}")
    print(f"Test accuracy: {results['accuracy']:.4f}")
    print(f"Test macro F1: {results['macro_f1']:.4f}")
    print(f"Test weighted F1: {results['weighted_f1']:.4f}")
    print("\nClassification Report:\n" + format_classification_report(all_labels, all_preds, label_names=bundle.label_names))
    print(f"Run ID: {run_id}")
    print(f"Saved predictions: {output_path}")


if __name__ == "__main__":
    main()
