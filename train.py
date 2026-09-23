"""Train the full-parameter ViHSD Mixture of Experts model."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Ensure repository root is on sys.path for Colab notebook execution
REPO_ROOT = str(Path(__file__).resolve().parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file, save_file
from tqdm.auto import tqdm

from models.factory import build_model, standardize_model_output
from src.dataset import prepare_data
from src.losses import compute_task_loss
from src.metrics import compute_classification_metrics
from src.utils import (
    apply_overrides,
    create_run_id,
    flatten_hyperparameters,
    load_config,
    resolve_output_path,
    set_seed,
)


def evaluate(model, loader, device, label_names=None):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits, _ = standardize_model_output(model(input_ids, attention_mask))
            total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
            total_examples += labels.numel()
            preds = logits.argmax(dim=-1).cpu().tolist()
            labs = labels.cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labs)
    cls_metrics = compute_classification_metrics(all_labels, all_preds, label_names=label_names)
    avg_loss = (total_loss / total_examples) if total_examples > 0 else 0.0
    return {
        "loss": avg_loss,
        "accuracy": cls_metrics["accuracy"],
        "macro_f1": cls_metrics["macro_f1"],
        "weighted_f1": cls_metrics["weighted_f1"],
        "per_class_f1": cls_metrics["per_class_f1"],
    }


def train_epoch(model, loader, optimizer, device, balance_factor, epoch, total_epochs, config, label_names=None):
    model.train()
    total_loss = 0.0
    total_auxiliary_loss = 0.0
    total_correct = 0
    total_examples = 0
    all_preds = []
    all_labels = []
    progress = tqdm(
        loader,
        desc=f"Epoch {epoch}/{total_epochs}",
        leave=True,
        dynamic_ncols=True,
        unit="batch",
    )
    for batch in progress:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, aux = standardize_model_output(model(input_ids, attention_mask))
        classification_loss = compute_task_loss(logits, labels, config)
        balance_loss = aux.get("balance_loss", 0.0)
        loss = classification_loss + balance_factor * balance_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_auxiliary_loss += float(balance_loss) * labels.size(0)
        preds = logits.argmax(dim=-1).cpu().tolist()
        labs = labels.cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labs)
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_examples += labels.size(0)
        progress.set_postfix(loss=f"{total_loss / total_examples:.4f}", accuracy=f"{total_correct / total_examples:.3f}")
    cls_metrics = compute_classification_metrics(all_labels, all_preds, label_names=label_names)
    avg_loss = (total_loss / total_examples) if total_examples > 0 else 0.0
    return {
        "loss": avg_loss,
        "auxiliary_loss": (total_auxiliary_loss / total_examples) if total_examples > 0 else 0.0,
        "accuracy": cls_metrics["accuracy"],
        "macro_f1": cls_metrics["macro_f1"],
        "weighted_f1": cls_metrics["weighted_f1"],
        "per_class_f1": cls_metrics["per_class_f1"],
    }


def maybe_start_wandb(config, run_id):
    logging_config = config["logging"]
    if not logging_config.get("use_wandb", False):
        return None
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("Install wandb or set logging.use_wandb to false") from error
    return wandb.init(project=logging_config["project"], name=run_id, config=config)


def apply_training_profile(config, smoke_test_override):
    training_config = config["training"]
    smoke_test = training_config.get("smoke_test", False)
    if smoke_test_override is not None:
        smoke_test = smoke_test_override
    if smoke_test:
        smoke_config = training_config.get("smoke", {})
        training_config["epochs"] = smoke_config.get("epochs", 1)
        training_config["max_train_samples"] = smoke_config.get("max_train_samples", 2000)
        training_config["max_eval_samples"] = smoke_config.get("max_eval_samples", 500)
    return smoke_test


def create_hyperparameters_log(config, run_id, smoke_test):
    """Keep only experiment-defining settings, excluding paths and credentials."""
    hyperparameters = {
        "seed": config["seed"],
        "dataset": config["dataset"],
        "training": config["training"],
        "model": config["model"],
        "routing": config["routing"],
    }
    architecture = config.get("model", {}).get("architecture", "phobert_moe")
    return {
        "run_id": run_id,
        "profile": "smoke" if smoke_test else "full",
        "architecture": architecture,
        "hyperparameters": hyperparameters,
        "flat_hyperparameters": flatten_hyperparameters(hyperparameters),
    }


def collect_routing_diagnostics(model) -> dict:
    """Extract routing entropy and expert load fractions from the last forward pass."""
    diagnostics = {}
    moe_layers = getattr(model, "moe_layers", {})
    if not moe_layers:
        inner = getattr(model, "module", model)
        moe_layers = getattr(inner, "moe_layers", {})
    for layer_idx_str, moe_layer in moe_layers.items():
        info = getattr(moe_layer, "last_routing_info", {})
        if not info:
            continue
        probs = info.get("probabilities")
        indices = info.get("top_indices")
        if probs is None or indices is None:
            continue
        if probs.shape[-1] == 0:
            continue
        log_probs = torch.log(probs.clamp(min=1e-9))
        entropy = -(probs * log_probs).sum(dim=-1).mean().item()
        num_experts = probs.shape[-1]
        flat_indices = indices.reshape(-1)
        counts = torch.bincount(flat_indices, minlength=num_experts).float()
        fractions = (counts / counts.sum().clamp_min(1.0)).tolist()
        diagnostics[f"layer_{layer_idx_str}"] = {
            "routing_entropy": entropy,
            "expert_load_fractions": fractions,
        }
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/vihsd.yaml")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="Override a YAML value for this run; repeat as needed (for example, training.epochs=10).",
    )
    parser.add_argument(
        "--smoke-test",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the YAML profile and run the small smoke-test configuration.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional identifier for this run. Defaults to UTC timestamp plus profile.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    apply_overrides(config, args.overrides)
    smoke_test = apply_training_profile(config, args.smoke_test)
    run_id = create_run_id(smoke_test, args.run_id)
    set_seed(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_config = {**config["dataset"], **config["training"]}
    bundle = prepare_data(data_config)
    model_config = {**config["model"], "pad_token_id": bundle.tokenizer.pad_token_id or 0}
    model_config.setdefault("architecture", "phobert_moe")
    model = build_model(model_config, bundle.tokenizer.vocab_size, bundle.num_labels).to(device)
    checkpoint_filename = f"{model_config['architecture']}_best.safetensors"
    print("Model architecture:")
    print(f"  variant: {model_config['architecture']}")
    if model_config["architecture"] in {"pretrained_backbone", "dense_phobert"}:
        print(f"  pretrained_model_name: {model_config.get('pretrained_model_name')}")
        print(f"  freeze_backbone: {model_config.get('freeze_backbone')}")
    if model_config["architecture"] == "phobert_moe":
        print(f"  pretrained_model_name: {model_config.get('pretrained_model_name')}")
        print(f"  moe_layers: {model_config.get('moe_layers', [8, 9, 10, 11])}")
        print(f"  freeze_attention: {model_config.get('freeze_attention', False)}")
        print(f"  upcycle: {model_config.get('upcycle', True)}")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    balance_factor = float(config["routing"]["load_balance_loss_factor"])
    checkpoint_root = resolve_output_path(config["paths"]["checkpoint_dir"], "CHECKPOINT_DIR")
    checkpoint_dir = checkpoint_root / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_root = resolve_output_path(config["paths"]["results_dir"], "RESULTS_DIR")
    results_dir = results_root / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    hyperparameters_path = results_dir / "hyperparameters.json"
    hyperparameters_path.write_text(
        json.dumps(create_hyperparameters_log(config, run_id, smoke_test), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    wandb_run = maybe_start_wandb(config, run_id)
    print(f"Training profile: {'smoke test' if smoke_test else 'full run'}")
    print(f"Run ID: {run_id}")
    best_macro_f1 = -1.0
    best_record = None
    history = []
    total_epochs = int(config["training"]["epochs"])
    for epoch in range(total_epochs):
        train_metrics = train_epoch(
            model,
            bundle.loaders["train"],
            optimizer,
            device,
            balance_factor,
            epoch + 1,
            total_epochs,
            config,
            label_names=bundle.label_names,
        )
        val_metrics = evaluate(model, bundle.loaders["validation"], device, label_names=bundle.label_names)
        record = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_auxiliary_loss": train_metrics["auxiliary_loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "train_weighted_f1": train_metrics["weighted_f1"],
            "validation_loss": val_metrics["loss"],
            "validation_accuracy": val_metrics["accuracy"],
            "validation_macro_f1": val_metrics["macro_f1"],
            "validation_weighted_f1": val_metrics["weighted_f1"],
        }
        history.append(record)
        print(record)
        if wandb_run is not None:
            wandb_run.log(record)
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            save_file(
                {name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()},
                str(checkpoint_dir / checkpoint_filename),
            )
            (checkpoint_dir / f"{model_config['architecture']}_metadata.json").write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "architecture": model_config["architecture"],
                        "checkpoint": checkpoint_filename,
                        "label_names": bundle.label_names,
                        "model_config": model_config,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            best_record = {
                **record,
                "train_per_class_f1": train_metrics["per_class_f1"],
                "validation_per_class_f1": val_metrics["per_class_f1"],
            }
    (results_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    if best_record is None:
        raise RuntimeError("Training produced no checkpoint; set training.epochs to at least 1.")
    best_checkpoint_path = checkpoint_dir / checkpoint_filename
    model.load_state_dict(load_file(str(best_checkpoint_path), device=str(device)))
    test_metrics = evaluate(model, bundle.loaders["test"], device, label_names=bundle.label_names)
    routing_diagnostics = {}
    if hasattr(model, "moe_layers") or hasattr(getattr(model, "module", model), "moe_layers"):
        model.eval()
        with torch.no_grad():
            for batch in bundle.loaders["validation"]:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                standardize_model_output(model(input_ids, attention_mask))
                break
        routing_diagnostics = collect_routing_diagnostics(model)
    run_metrics = {
        "run_id": run_id,
        "best_epoch": best_record["epoch"],
        "routing_diagnostics": routing_diagnostics,
        "train": {
            "loss": best_record["train_loss"],
            "accuracy": best_record["train_accuracy"],
            "macro_f1": best_record["train_macro_f1"],
            "weighted_f1": best_record["train_weighted_f1"],
            "per_class_f1": best_record.get("train_per_class_f1", {}),
        },
        "validation": {
            "loss": best_record["validation_loss"],
            "accuracy": best_record["validation_accuracy"],
            "macro_f1": best_record["validation_macro_f1"],
            "weighted_f1": best_record["validation_weighted_f1"],
            "per_class_f1": best_record.get("validation_per_class_f1", {}),
        },
        "test": {
            "loss": test_metrics["loss"],
            "accuracy": test_metrics["accuracy"],
            "macro_f1": test_metrics["macro_f1"],
            "weighted_f1": test_metrics["weighted_f1"],
            "per_class_f1": test_metrics["per_class_f1"],
        },
    }
    (results_dir / "run_metrics.json").write_text(
        json.dumps(run_metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(run_metrics, indent=2))
    if wandb_run is not None:
        wandb_run.log({
            "best_epoch": best_record["epoch"],
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_weighted_f1": test_metrics["weighted_f1"],
        })
    (checkpoint_root / "latest_run.json").write_text(json.dumps({"run_id": run_id, "checkpoint": str(best_checkpoint_path)}, indent=2), encoding="utf-8")
    (results_root / "latest_run.json").write_text(json.dumps({"run_id": run_id, "results_dir": str(results_dir)}, indent=2), encoding="utf-8")
    if wandb_run is not None:
        wandb_run.finish()
    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"Run results: {results_dir}")
    print(f"Hyperparameters: {hyperparameters_path}")


if __name__ == "__main__":
    main()
