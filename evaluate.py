"""Đánh giá test bằng resolved config của một run và finalist đã khóa."""
import argparse
import csv
import json
from pathlib import Path
import torch
import yaml
from safetensors.torch import load_file
from src.models.factory import build_model, model_inputs_from_batch
from src.training.engine import prepare_run_data
from src.evaluation.metrics import compute_classification_metrics, compute_prediction_diagnostics
from src.utils import load_config, set_seed
from src.utils.artifacts import TRACKER, config_fingerprint, now, update_tracker, write_json


def evaluate_run(run_dir, finalist_path, *, tracker=TRACKER):
    run_dir = Path(run_dir)
    config = load_config(run_dir / "resolved_config.yaml")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if "base_config" in (yaml.safe_load(Path(finalist_path).read_text(encoding="utf-8")) or {}):
        raise ValueError("Finalist must be a self-contained resolved configuration")
    finalist = load_config(finalist_path)
    lock = finalist.get("finalization", {})
    if manifest["status"] != "completed" or config.get("profile") != "full":
        raise ValueError("Final evaluation requires a completed full run")
    if (not lock.get("locked") or not lock.get("locked_at") or not lock.get("selection_reason")
            or config["seed"] not in lock.get("seeds", [])
            or config["experiment_id"] != finalist.get("experiment_id")
            or config_fingerprint(config) != config_fingerprint(finalist)):
        raise ValueError("Finalist must lock the exact configuration, experiment and seed set")
    output = run_dir / "evaluation"
    if (output / "test.json").exists():
        raise FileExistsError("Test evaluation already recorded for this run")
    set_seed(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = prepare_run_data(config, final=True)
    saved_labels = json.loads((run_dir / "labels.json").read_text(encoding="utf-8"))
    if saved_labels["label_names"] != bundle.label_names:
        raise ValueError("Dataset label mapping differs from the training run")
    saved_data = manifest.get("data_provenance")
    if saved_data and bundle.provenance != saved_data:
        raise ValueError("Dataset/tokenizer provenance differs from the training run")
    hf_config = json.loads((run_dir / "hf_config.json").read_text(encoding="utf-8"))
    model = build_model(config["model"], num_labels=bundle.num_labels, hf_config=hf_config).to(device)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    checkpoint = run_dir / summary["checkpoint"]
    if not checkpoint.resolve().is_relative_to(run_dir.resolve()):
        raise ValueError("Checkpoint must be inside the selected run")
    model.load_state_dict(load_file(str(checkpoint), device=str(device)))
    model.eval()
    labels, predictions, probabilities = [], [], []
    loss_sum = 0.0
    with torch.no_grad():
        for batch in bundle.loaders["test"]:
            actual = batch["labels"].to(device)
            logits, _ = model(**model_inputs_from_batch(batch, device))
            loss_sum += float(torch.nn.functional.cross_entropy(logits, actual, reduction="sum"))
            labels.extend(actual.cpu().tolist())
            predictions.extend(logits.argmax(-1).cpu().tolist())
            probabilities.extend(logits.softmax(-1).cpu().tolist())
    metrics = compute_classification_metrics(labels, predictions, bundle.label_names)
    diagnostics = compute_prediction_diagnostics(labels, predictions, bundle.label_names)
    result = {"split": "test", "checkpoint": summary["checkpoint"], "evaluated_at": now(),
              "finalist_path": str(finalist_path), "seed_set": lock["seeds"],
              "loss": loss_sum / len(labels), **metrics, **diagnostics}
    output.mkdir(exist_ok=True)
    write_json(output / "test.json", result)
    write_json(output / "predictions.json", {"split": "test", "checkpoint": summary["checkpoint"],
               "label_names": bundle.label_names, "predictions": [
                   {"label": y, "prediction": p, "probabilities": probs}
                   for y, p, probs in zip(labels, predictions, probabilities)]})
    with (output / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["actual/predicted", *bundle.label_names])
        writer.writerows(zip(bundle.label_names, diagnostics["confusion_matrix"]))
    update_tracker(manifest["run_id"], {f"test_{k}": v for k, v in metrics.items()}, tracker=tracker)
    if manifest.get("wandb_path") and config.get("logging", {}).get("mode") != "offline":
        try:
            import wandb
            remote = wandb.Api().run(manifest["wandb_path"])
            remote.summary.update({f"test/{k}": metrics[k] for k in ("macro_f1", "accuracy", "weighted_f1")})
        except Exception as error:
            manifest["wandb_sync"] = "pending"
            manifest["test_sync_error"] = str(error)
            write_json(run_dir / "manifest.json", manifest)
            update_tracker(manifest["run_id"], {"wandb_sync": "pending"}, tracker=tracker)
            print("Test metrics saved locally; W&B test summary update is pending")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a completed run with a locked finalist configuration.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--finalist-config", required=True)
    parser.add_argument("--final-evaluation", action="store_true")
    args = parser.parse_args(argv)
    if not args.final_evaluation:
        parser.error("Test is reserved for locked finalists; --final-evaluation is required")
    result = evaluate_run(args.run_dir, args.finalist_config)
    print(f"Test macro F1: {result['macro_f1']:.6f}")


if __name__ == "__main__":
    main()
