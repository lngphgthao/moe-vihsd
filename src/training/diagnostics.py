"""Chẩn đoán theo example và class, chỉ dành cho classification head."""
from collections import defaultdict
import torch
from src.models.factory import model_inputs_from_batch
from src.evaluation.metrics import compute_prediction_diagnostics


def collect_model_diagnostics(model, loader, device, label_names):
    model.eval()
    records = defaultdict(lambda: {"example_count": 0, "probability_sum": None, "assignment_counts": None, "entropy_sum": 0.0})
    classification = {}
    actual, predicted = [], []
    with torch.no_grad():
        for batch in loader:
            logits, aux = model(**model_inputs_from_batch(batch, device))
            labels = batch["labels"].to(device)
            actual.extend(labels.cpu().tolist())
            predicted.extend(logits.argmax(-1).cpu().tolist())
            class_probs = logits.softmax(-1)
            for class_id in labels.unique().tolist():
                mask = labels == class_id
                name = label_names[class_id]
                values = {"logits": logits[mask].sum(0).cpu(), "probabilities": class_probs[mask].sum(0).cpu()}
                if name not in classification:
                    classification[name] = {"example_count": 0, "logits": torch.zeros_like(values["logits"]),
                                            "probabilities": torch.zeros_like(values["probabilities"])}
                classification[name]["example_count"] += int(mask.sum())
                for key, value in values.items():
                    classification[name][key] += value
            probabilities = aux.get("probabilities")
            indices = aux.get("top_indices")
            if probabilities is None or indices is None:
                continue
            labels = batch["labels"].to(device)
            for class_id in labels.unique().tolist():
                mask = labels == class_id
                probs = probabilities[mask].detach().cpu()
                record = records[label_names[class_id]]
                counts = torch.bincount(indices[mask].reshape(-1).cpu(), minlength=probs.shape[-1])
                record["example_count"] += int(mask.sum())
                record["entropy_sum"] += float(-(probs * probs.clamp_min(1e-9).log()).sum())
                for key, value in (("probability_sum", probs.sum(0)), ("assignment_counts", counts)):
                    record[key] = value if record[key] is None else record[key] + value
    result = {"scope": "example", "by_class": {}, "logits_by_true_class": {},
              "probabilities_by_true_class": {},
              "prediction_diagnostics": compute_prediction_diagnostics(actual, predicted, label_names)}
    for name, record in classification.items():
        result["logits_by_true_class"][name] = (record["logits"] / record["example_count"]).tolist()
        result["probabilities_by_true_class"][name] = (record["probabilities"] / record["example_count"]).tolist()
    for name, record in records.items():
        count = record["example_count"]
        result["by_class"][name] = {
            "example_count": count, "routing_entropy": record["entropy_sum"] / count,
            "mean_probabilities": (record["probability_sum"] / count).tolist(),
            "assignment_counts": record["assignment_counts"].tolist(),
            "expert_load_fractions": (record["assignment_counts"] / record["assignment_counts"].sum()).tolist()}
    return result
