"""Train/validation và optimizer dùng chung cho hai head."""
import math
import os
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from src.models.factory import model_inputs_from_batch, standardize_model_output
from src.training.losses import compute_task_loss
from src.evaluation.metrics import compute_classification_metrics

def evaluate(model, loader, device, label_names=None):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"].to(device)
            logits, _ = standardize_model_output(model(**model_inputs_from_batch(batch, device)))
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


def train_epoch(
    model, loader, optimizer, device, balance_factor, epoch, total_epochs, config,
    label_names=None, *, initial_step=0, on_evaluate=None,
):
    model.train()
    total_loss = 0.0
    total_auxiliary_loss = 0.0
    total_classification_loss = 0.0
    total_balance_loss = 0.0
    total_correct = 0
    total_examples = 0
    all_preds = []
    all_labels = []
    eval_frequency = config["training"].get("eval_frequency")

    def current_metrics():
        cls_metrics = compute_classification_metrics(all_labels, all_preds, label_names=label_names)
        denominator = max(total_examples, 1)
        return {
            "loss": total_loss / denominator,
            "classification_loss": total_classification_loss / denominator,
            "balance_loss": total_balance_loss / denominator,
            "auxiliary_loss": total_auxiliary_loss / denominator,
            "weighted_auxiliary_loss": (
                balance_factor * total_balance_loss
            ) / denominator,
            "accuracy": cls_metrics["accuracy"],
            "macro_f1": cls_metrics["macro_f1"],
            "weighted_f1": cls_metrics["weighted_f1"],
            "per_class_f1": cls_metrics["per_class_f1"],
        }

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch}/{total_epochs}",
        leave=True,
        dynamic_ncols=True,
        unit="batch",
    )
    for batch_index, batch in enumerate(progress, start=1):
        labels = batch["labels"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, aux = standardize_model_output(model(**model_inputs_from_batch(batch, device)))
        if not torch.isfinite(logits).all():
            raise ValueError(
                f"Non-finite logits at epoch={epoch} batch={batch_index}; "
                f"min={logits.min().item():.6g} max={logits.max().item():.6g}"
            )
        classification_loss = compute_task_loss(logits, labels, config)
        balance_loss = aux.get("balance_loss", 0.0)
        loss = classification_loss + balance_factor * balance_loss
        if not torch.isfinite(loss):
            raise ValueError(
                f"Non-finite training loss at epoch={epoch} batch={batch_index}: "
                f"classification={float(classification_loss.detach()) if torch.isfinite(classification_loss) else 'non-finite'}, "
                f"balance={float(torch.as_tensor(balance_loss).detach()) if torch.isfinite(torch.as_tensor(balance_loss)) else 'non-finite'}"
            )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            raise ValueError(
                f"Non-finite gradient norm at epoch={epoch} batch={batch_index}; "
                "check logits, backbone precision and optimizer inputs"
            )
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_classification_loss += float(classification_loss.detach()) * labels.size(0)
        total_balance_loss += float(torch.as_tensor(balance_loss).detach()) * labels.size(0)
        total_auxiliary_loss += float(torch.as_tensor(balance_loss).detach()) * labels.size(0)
        preds = logits.argmax(dim=-1).cpu().tolist()
        labs = labels.cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labs)
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_examples += labels.size(0)
        progress.set_postfix(loss=f"{total_loss / total_examples:.4f}", accuracy=f"{total_correct / total_examples:.3f}")
        global_step = initial_step + batch_index
        final_step = epoch == total_epochs and batch_index == len(loader)
        if eval_frequency and on_evaluate is not None and (
            global_step % eval_frequency == 0 or final_step
        ):
            try:
                on_evaluate(current_metrics(), epoch, global_step)
            finally:
                model.train()
    return current_metrics()


def apply_training_profile(config, smoke_test_override):
    training_config = config["training"]
    eval_frequency = training_config.get("eval_frequency")
    if eval_frequency is not None and (
        type(eval_frequency) is not int or eval_frequency <= 0
    ):
        raise ValueError("training.eval_frequency must be a positive integer of optimizer steps or null")
    smoke_test = training_config.get("smoke_test", False)
    if smoke_test_override is not None:
        smoke_test = smoke_test_override
    if smoke_test:
        smoke_config = training_config.get("smoke", {})
        training_config["epochs"] = smoke_config.get("epochs", 1)
        training_config["max_train_samples"] = smoke_config.get("max_train_samples", 2000)
        training_config["max_eval_samples"] = smoke_config.get("max_eval_samples", 500)
    tiny_overfit = config.get("diagnostics", {}).get("tiny_overfit", {})
    if tiny_overfit.get("enabled", False):
        training_config["epochs"] = int(tiny_overfit.get("epochs", 75))
        training_config["max_train_samples"] = int(tiny_overfit.get("samples", 150))
        training_config["train_subset_strategy"] = "stratified"
    return smoke_test


def create_optimizer_param_groups(model, training):
    """Chia tham số trainable theo tên, ưu tiên router trước head chứa nó."""
    learning_rate = float(training["learning_rate"])
    groups = {}
    for name in ("backbone", "head", "router"):
        configured_lr = None if name == "backbone" else training.get(f"{name}_learning_rate")
        group_lr = learning_rate if configured_lr is None else float(configured_lr)
        if not math.isfinite(group_lr) or group_lr < 0:
            raise ValueError(f"Invalid learning rate for {name}: {group_lr}")
        groups[name] = {
            "name": name,
            "params": [],
            "lr": group_lr,
        }
    encoder_ids = {id(p) for p in model.encoder.parameters()}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "router" in name.split(".")[:-1]:
            group = "router"
        elif id(parameter) not in encoder_ids:
            group = "head"
        else:
            group = "backbone"
        groups[group]["params"].append(parameter)
    return [group for group in groups.values() if group["params"]]


def prepare_run_data(config, *, final=False):
    from src.data.loader import prepare_data
    data_config = {**config["dataset"], **config["training"]}
    data_config["tokenizer"] = config["model"]["backbone"]
    if not data_config.get("tokenizer_revision"):
        data_config["tokenizer_revision"] = config["model"].get("revision")
    if final:
        data_config["max_train_samples"] = None
        data_config["max_eval_samples"] = None
    return prepare_data(data_config)


def run_training(config, config_path, overrides, run_id=None, *, tracker=None):
    import time
    import traceback
    from safetensors.torch import load_file, save_file
    from src.models.factory import build_model, validate_model_config
    from src.training.diagnostics import collect_model_diagnostics
    from src.utils import set_seed
    from src.utils.artifacts import RunArtifacts, TRACKER, now, sanitized, update_tracker, write_json

    validate_model_config(config["model"])
    if int(config["training"]["epochs"]) < 1:
        raise ValueError("training.epochs must be positive")
    factor = float(config.get("routing", {}).get("balance_loss_coef", 0.0))
    if not math.isfinite(factor) or factor < 0:
        raise ValueError("routing.balance_loss_coef must be finite and non-negative")
    artifacts = RunArtifacts(config, run_id, config_path, overrides, tracker=tracker or TRACKER)
    wb = None
    sync_failed = False
    started = time.perf_counter()
    with artifacts.console():
        try:
            print(f"Run: {artifacts.run_id}; profile: {config['profile']}")
            set_seed(int(config["seed"]))
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            artifacts.manifest["hardware"] = {
                "device": str(device), "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
                "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
                "process_id": os.getpid(),
                "visible_gpu_count": torch.cuda.device_count() if device.type == "cuda" else 0,
            }
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            bundle = prepare_run_data(config)
            if not len(bundle.loaders["train"]) or not len(bundle.loaders["validation"]):
                raise ValueError("Training and validation loaders must be nonempty")
            if config["profile"] == "tiny_overfit":
                counts = torch.bincount(torch.as_tensor(bundle.loaders["train"].dataset["labels"]), minlength=bundle.num_labels)
                if (counts == 0).any():
                    raise ValueError("Tiny-overfit subset must contain every class")
            write_json(artifacts.path / "labels.json", {"label_names": bundle.label_names,
                       "label_mapping": {str(k): v for k, v in bundle.label_mapping.items()}})
            artifacts.manifest["data_provenance"] = bundle.provenance
            model = build_model(config["model"], num_labels=bundle.num_labels).to(device)
            native_config = model.model.config if hasattr(model, "model") else model.encoder.config
            write_json(artifacts.path / "hf_config.json", native_config.to_dict())
            artifacts.manifest["model_revision"] = getattr(native_config, "_commit_hash", None)
            artifacts.manifest["parameters"] = {"total": sum(p.numel() for p in model.parameters()),
                                                "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
            groups = create_optimizer_param_groups(model, config["training"])
            artifacts.manifest["optimizer_groups"] = [{"name": g["name"], "lr": g["lr"],
                "parameters": sum(p.numel() for p in g["params"])} for g in groups]
            print("Optimizer groups:", artifacts.manifest["optimizer_groups"])
            optimizer = torch.optim.AdamW(groups, weight_decay=float(config["training"].get("weight_decay", 0.0)))
            if config.get("logging", {}).get("use_wandb", False):
                try:
                    import wandb
                    wb = wandb.init(project=config["logging"]["project"], name=artifacts.run_id,
                                    config=sanitized({**config, "run_id": artifacts.run_id}), dir=str(artifacts.path),
                                    mode=config["logging"].get("mode", "online"))
                    artifacts.manifest["wandb_url"] = wb.url
                    artifacts.manifest["wandb_path"] = wb.path
                    artifacts.manifest["wandb_sync"] = "offline" if config["logging"].get("mode") == "offline" else "logging"
                except Exception as error:
                    sync_failed = True
                    print(f"W&B unavailable; local logging continues: {error}")
            else:
                print("W&B disabled for this run; artifacts are local.")
            artifacts.save_manifest()
            best_score = -math.inf
            best = {}
            step = 0
            checkpoint = artifacts.path / "checkpoints/best.safetensors"

            def validate(train_metrics, epoch, global_step):
                nonlocal best_score, best, sync_failed
                validation = evaluate(model, bundle.loaders["validation"], device, bundle.label_names)
                artifacts.metric("train", {**train_metrics, "balance_loss_coef": factor}, epoch, global_step)
                artifacts.metric("validation", validation, epoch, global_step)
                if not math.isfinite(validation["macro_f1"]) or not math.isfinite(validation["loss"]):
                    raise ValueError("Non-finite validation metric")
                if validation["macro_f1"] > best_score:
                    best_score = validation["macro_f1"]
                    save_file({k: v.detach().contiguous() for k, v in model.state_dict().items()}, str(checkpoint))
                    best = {"best_epoch": epoch, "best_global_step": global_step, "validation": validation,
                            "selection_metric": "validation.macro_f1", "checkpoint": "checkpoints/best.safetensors"}
                    write_json(artifacts.path / "summary.json", best)
                print(f"epoch={epoch} step={global_step} val_macro_f1={validation['macro_f1']:.6f}")
                if wb is not None:
                    try:
                        wb.log({**{f"train/{k}": v for k, v in train_metrics.items()},
                                **{f"validation/{k}": v for k, v in validation.items()}, "epoch": epoch}, step=global_step)
                    except Exception as error:
                        sync_failed = True
                        print(f"W&B logging pending: {error}")

            epochs = int(config["training"]["epochs"])
            artifacts.manifest["training_started_at"] = now()
            artifacts.save_manifest()
            for epoch in range(1, epochs + 1):
                train_metrics = train_epoch(model, bundle.loaders["train"], optimizer, device, factor,
                                            epoch, epochs, config, bundle.label_names,
                                            initial_step=step, on_evaluate=validate)
                step += len(bundle.loaders["train"])
                if not config["training"].get("eval_frequency"):
                    validate(train_metrics, epoch, step)
            artifacts.manifest["training_finished_at"] = now()
            artifacts.save_manifest()
            model.load_state_dict(load_file(str(checkpoint), device=str(device)))
            if config.get("diagnostics", {}).get("collect_model_diagnostics", False):
                diagnostic = collect_model_diagnostics(model, bundle.loaders["validation"], device, bundle.label_names)
                write_json(artifacts.path / "diagnostics/validation.json", {"split": "validation", "checkpoint": best["checkpoint"], **diagnostic})
            best["cost"] = {"elapsed_seconds": time.perf_counter() - started,
                            "scope": "data preparation through best-checkpoint reload and diagnostics",
                            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None}
            write_json(artifacts.path / "summary.json", best)
            update_tracker(artifacts.run_id, {f"val_{k}": v for k, v in best["validation"].items()}, tracker=artifacts.tracker)
            artifacts.manifest.update(status="completed", finished_at=now())
        except BaseException as error:
            traceback.print_exc()
            artifacts.manifest.update(status="failed", finished_at=now(), error=str(error))
            if wb is not None:
                try:
                    wb.finish(exit_code=1)
                    artifacts.manifest["wandb_sync"] = "offline" if config["logging"].get("mode") == "offline" else "synced"
                except Exception:
                    sync_failed = True
            artifacts.manifest["wandb_sync"] = "pending" if sync_failed else artifacts.manifest["wandb_sync"]
            artifacts.save_manifest()
            raise
        else:
            if wb is not None:
                try:
                    wb.finish()
                    artifacts.manifest["wandb_sync"] = "offline" if config["logging"].get("mode") == "offline" else "synced"
                except Exception as error:
                    sync_failed = True
                    print(f"W&B finish pending: {error}")
            if sync_failed:
                artifacts.manifest["wandb_sync"] = "pending"
            artifacts.save_manifest()
    return artifacts.path
