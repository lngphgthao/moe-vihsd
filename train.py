"""Train the full-parameter ViHSD Mixture of Experts model."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Ensure repository root is in sys.path for direct imports and notebook environments
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file, save_file
from tqdm.auto import tqdm

from data.vihsd import prepare_data
from metrics import classification_metrics
from models.moe import ViHSDMoEClassifier
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _routing_counts(model, routing, counts):
    if getattr(model, "num_experts", 0) and routing["top_indices"].numel():
        counts += torch.bincount(routing["top_indices"].reshape(-1).cpu(), minlength=model.num_experts)
    return counts


def evaluate(model, loader, device, label_names):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    predictions, labels_all = [], []
    routing_counts = torch.zeros(getattr(model, "num_experts", 0), dtype=torch.long)
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits, routing = model(input_ids, attention_mask)
            total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
            total_examples += labels.numel()
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
            labels_all.extend(labels.cpu().tolist())
            routing_counts = _routing_counts(model, routing, routing_counts)
    return {
        "loss": total_loss / total_examples,
        **classification_metrics(labels_all, predictions, label_names),
        "routing_counts": routing_counts.tolist(),
    }


def train_epoch(model, loader, optimizer, device, balance_factor, class_weights, label_names, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    total_examples = 0
    predictions, labels_all = [], []
    routing_counts = torch.zeros(getattr(model, "num_experts", 0), dtype=torch.long)
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
        logits, routing = model(input_ids, attention_mask)
        classification_loss = F.cross_entropy(logits, labels, weight=class_weights)
        balance_loss = routing["balance_loss"]
        loss = classification_loss + balance_factor * balance_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_examples += labels.size(0)
        predictions.extend(logits.argmax(dim=-1).detach().cpu().tolist())
        labels_all.extend(labels.detach().cpu().tolist())
        routing_counts = _routing_counts(model, routing, routing_counts)
        progress.set_postfix(loss=f"{total_loss / total_examples:.4f}")
    return {
        "loss": total_loss / total_examples,
        **classification_metrics(labels_all, predictions, label_names),
        "routing_counts": routing_counts.tolist(),
    }


def maybe_start_wandb(config):
    logging_config = config["logging"]
    if not logging_config.get("use_wandb", False):
        return None
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("Install wandb or set logging.use_wandb to false") from error
    return wandb.init(project=logging_config["project"], name=logging_config.get("run_name"), config=config)


def apply_training_profile(config, smoke_test_override):
    training_config = config["training"]
    smoke_test = training_config.get("smoke_test", False)
    if smoke_test_override is not None:
        smoke_test = smoke_test_override
    if smoke_test:
        smoke_config = training_config.get("smoke", {})
        training_config["epochs"] = smoke_config.get("epochs", 1)
        training_config["max_train_samples"] = smoke_config.get("max_train_samples", 2000)
    return smoke_test


def apply_overrides(config, overrides):
    """Apply ``section.key=value`` overrides parsed as YAML scalars."""
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid --set value {override!r}; expected section.key=value.")
        key_path, raw_value = override.split("=", 1)
        keys = key_path.split(".")
        if not key_path or any(not key for key in keys):
            raise ValueError(f"Invalid configuration key {key_path!r}.")
        target = config
        for key in keys[:-1]:
            if key not in target or not isinstance(target[key], dict):
                raise KeyError(f"Unknown configuration section {key_path!r}.")
            target = target[key]
        if keys[-1] not in target:
            raise KeyError(f"Unknown configuration key {key_path!r}.")
        target[keys[-1]] = yaml.safe_load(raw_value)
    return config


def create_run_id(smoke_test, requested_run_id=None):
    if requested_run_id:
        return requested_run_id
    hanoi_timezone = timezone(timedelta(hours=7), name="Asia/Ho_Chi_Minh")
    timestamp = datetime.now(hanoi_timezone).strftime("%Y%m%dT%H%M%S")
    profile = "smoke" if smoke_test else "full"
    return f"{timestamp}-{profile}"


def resolve_output_path(configured_path, environment_name):
    return Path(os.getenv(environment_name, configured_path)).expanduser()


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def resolve_class_weights(mode, counts, device):
    if mode == "none":
        return None
    if mode != "balanced":
        raise ValueError("training.class_weighting must be 'none' or 'balanced'.")
    counts_tensor = torch.tensor(counts, dtype=torch.float, device=device)
    if (counts_tensor == 0).any():
        raise ValueError("Cannot calculate balanced weights when a training label has no examples.")
    return counts_tensor.sum() / (len(counts) * counts_tensor)


def flatten_hyperparameters(values, prefix=""):
    """Return nested configuration values as stable dotted keys for comparison."""
    flattened = {}
    for key, value in values.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(flatten_hyperparameters(value, full_key))
        else:
            flattened[full_key] = value
    return flattened


def create_hyperparameters_log(config, run_id, smoke_test):
    """Keep only experiment-defining settings, excluding paths and credentials."""
    hyperparameters = {
        "seed": config["seed"],
        "dataset": config["dataset"],
        "training": config["training"],
        "model": config["model"],
        "routing": config["routing"],
    }
    return {
        "run_id": run_id,
        "profile": "smoke" if smoke_test else "full",
        "hyperparameters": hyperparameters,
        "flat_hyperparameters": flatten_hyperparameters(hyperparameters),
    }


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
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    apply_overrides(config, args.overrides)
    smoke_test = apply_training_profile(config, args.smoke_test)
    run_id = create_run_id(smoke_test, args.run_id)
    set_seed(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_config = {**config["dataset"], **config["training"]}
    bundle = prepare_data(data_config)
    model_config = {**config["model"], "pad_token_id": bundle.tokenizer.pad_token_id or 0}
    model = ViHSDMoEClassifier(bundle.tokenizer.vocab_size, bundle.num_labels, model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(config["training"]["scheduler_factor"]),
        patience=int(config["training"]["scheduler_patience"]),
    )
    class_weights = resolve_class_weights(
        config["training"]["class_weighting"], bundle.label_counts[config["dataset"]["train_split"]], device
    )
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
    (results_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "git_commit": git_commit(),
                "device": str(device),
                "torch_version": torch.__version__,
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "split_sizes": bundle.split_sizes,
                "label_names": bundle.label_names,
                "label_counts": bundle.label_counts,
                "class_weights": class_weights.tolist() if class_weights is not None else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    wandb_run = maybe_start_wandb(config)
    print(f"Training profile: {'smoke test' if smoke_test else 'full run'}")
    print(f"Run ID: {run_id}")
    selection_metric = config["training"]["selection_metric"]
    best_score = float("-inf")
    epochs_without_improvement = 0
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
            class_weights,
            bundle.label_names,
            epoch + 1,
            total_epochs,
        )
        validation_metrics = evaluate(model, bundle.loaders["validation"], device, bundle.label_names)
        record = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        print(record)
        if wandb_run is not None:
            wandb_run.log(record)
        score = validation_metrics.get(selection_metric)
        if score is None:
            raise KeyError(f"Unknown training.selection_metric {selection_metric!r}.")
        scheduler.step(score)
        if score > best_score:
            best_score = score
            epochs_without_improvement = 0
            save_file({name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()}, str(checkpoint_dir / "vihsd_moe_best.safetensors"))
            (checkpoint_dir / "vihsd_moe_metadata.json").write_text(json.dumps({"run_id": run_id, "label_names": bundle.label_names, "model_config": model_config}, indent=2), encoding="utf-8")
            best_record = record
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(config["training"]["early_stopping_patience"]):
                print(f"Early stopping at epoch {epoch + 1}; best validation {selection_metric}: {best_score:.4f}")
                break
    (results_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    if best_record is None:
        raise RuntimeError("Training produced no checkpoint; set training.epochs to at least 1.")
    best_checkpoint_path = checkpoint_dir / "vihsd_moe_best.safetensors"
    model.load_state_dict(load_file(str(best_checkpoint_path), device=str(device)))
    test_metrics = None
    if config["training"].get("evaluate_test", True):
        test_metrics = evaluate(model, bundle.loaders["test"], device, bundle.label_names)
    run_metrics = {
        "run_id": run_id,
        "best_epoch": best_record["epoch"],
        "selection_metric": selection_metric,
        "train": best_record["train"],
        "validation": best_record["validation"],
        "test": test_metrics,
    }
    (results_dir / "run_metrics.json").write_text(
        json.dumps(run_metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(run_metrics, indent=2))
    if wandb_run is not None and test_metrics is not None:
        wandb_run.log({"best_epoch": best_record["epoch"], "test_loss": test_metrics["loss"], "test_accuracy": test_metrics["accuracy"], "test_macro_f1": test_metrics["macro_f1"]})
    (checkpoint_root / "latest_run.json").write_text(json.dumps({"run_id": run_id, "checkpoint": str(checkpoint_dir / "vihsd_moe_best.safetensors")}, indent=2), encoding="utf-8")
    (results_root / "latest_run.json").write_text(json.dumps({"run_id": run_id, "results_dir": str(results_dir)}, indent=2), encoding="utf-8")
    if wandb_run is not None:
        wandb_run.finish()
    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"Run results: {results_dir}")
    print(f"Hyperparameters: {hyperparameters_path}")


if __name__ == "__main__":
    main()
