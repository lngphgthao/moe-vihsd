"""Train the full-parameter ViHSD Mixture of Experts model."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import save_file
from tqdm.auto import tqdm

from data.vihsd import prepare_data
from models.moe import ViHSDMoEClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits, _ = model(input_ids, attention_mask)
            total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
            total_correct += (logits.argmax(dim=-1) == labels).sum().item()
            total_examples += labels.numel()
    return total_loss / total_examples, total_correct / total_examples


def train_epoch(model, loader, optimizer, device, balance_factor):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for batch in tqdm(loader, desc="Training", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, routing = model(input_ids, attention_mask)
        classification_loss = F.cross_entropy(logits, labels)
        balance_loss = routing["balance_loss"]
        loss = classification_loss + balance_factor * balance_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_examples += labels.size(0)
    return total_loss / total_examples, total_correct / total_examples


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/vihsd.yaml")
    parser.add_argument(
        "--smoke-test",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the YAML profile and run the small smoke-test configuration.",
    )
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    smoke_test = apply_training_profile(config, args.smoke_test)
    set_seed(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_config = {**config["dataset"], **config["training"]}
    bundle = prepare_data(data_config)
    model_config = {**config["model"], "pad_token_id": bundle.tokenizer.pad_token_id or 0}
    model = ViHSDMoEClassifier(bundle.tokenizer.vocab_size, bundle.num_labels, model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    balance_factor = float(config["routing"]["load_balance_loss_factor"])
    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = maybe_start_wandb(config)
    print(f"Training profile: {'smoke test' if smoke_test else 'full run'}")
    best_accuracy = -1.0
    history = []
    for epoch in range(int(config["training"]["epochs"])):
        train_loss, train_accuracy = train_epoch(model, bundle.loaders["train"], optimizer, device, balance_factor)
        validation_loss, validation_accuracy = evaluate(model, bundle.loaders["validation"], device)
        record = {"epoch": epoch + 1, "train_loss": train_loss, "train_accuracy": train_accuracy, "validation_loss": validation_loss, "validation_accuracy": validation_accuracy}
        history.append(record)
        print(record)
        if wandb_run is not None:
            wandb_run.log(record)
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            save_file({name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()}, str(checkpoint_dir / "vihsd_moe_best.safetensors"))
            (checkpoint_dir / "vihsd_moe_metadata.json").write_text(json.dumps({"label_names": bundle.label_names, "model_config": model_config}, indent=2), encoding="utf-8")
    (results_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    if wandb_run is not None:
        wandb_run.finish()
    print(f"Best checkpoint: {checkpoint_dir / 'vihsd_moe_best.safetensors'}")


if __name__ == "__main__":
    main()
