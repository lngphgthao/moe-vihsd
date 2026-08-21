"""Evaluate a saved ViHSD MoE checkpoint and write JSON predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file
from tqdm.auto import tqdm

from data.vihsd import prepare_data
from models.moe import ViHSDMoEClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/vihsd.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--run-id", default=None, help="Evaluate a specific run directory.")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = prepare_data({**config["dataset"], **config["training"]})
    model_config = {**config["model"], "pad_token_id": bundle.tokenizer.pad_token_id or 0}
    model = ViHSDMoEClassifier(bundle.tokenizer.vocab_size, bundle.num_labels, model_config).to(device)
    checkpoint_root = Path(config["paths"]["checkpoint_dir"])
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    elif args.run_id:
        checkpoint_path = checkpoint_root / args.run_id / "vihsd_moe_best.safetensors"
    else:
        latest_path = checkpoint_root / "latest_run.json"
        if not latest_path.exists():
            raise FileNotFoundError("No latest run found. Train first or pass --checkpoint/--run-id.")
        checkpoint_path = Path(json.loads(latest_path.read_text(encoding="utf-8"))["checkpoint"])
    model.load_state_dict(load_file(str(checkpoint_path), device=str(device)))
    model.eval()
    predictions = []
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    routing_counts = torch.zeros(model.num_experts, dtype=torch.long)
    with torch.no_grad():
        for batch in tqdm(bundle.loaders["test"], desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits, routing = model(input_ids, attention_mask)
            predicted = logits.argmax(dim=-1)
            total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
            total_correct += (predicted == labels).sum().item()
            total_examples += labels.numel()
            routing_counts += torch.bincount(routing["top_indices"].reshape(-1).cpu(), minlength=model.num_experts)
            predictions.extend({"prediction": int(prediction), "label": int(label)} for prediction, label in zip(predicted.cpu(), labels.cpu()))
    results = {"loss": total_loss / total_examples, "accuracy": total_correct / total_examples, "label_names": bundle.label_names, "routing_counts": routing_counts.tolist(), "predictions": predictions}
    results_root = Path(config["paths"]["results_dir"])
    run_id = checkpoint_path.parent.name
    results_dir = results_root / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "vihsd_predictions.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Test loss: {results['loss']:.4f}")
    print(f"Test accuracy: {results['accuracy']:.4f}")
    print(f"Run ID: {run_id}")
    print(f"Saved predictions: {output_path}")


if __name__ == "__main__":
    main()
