"""Opt-in diagnostics for investigating classifier and MoE behavior."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from src.models.factory import standardize_model_output


def _class_names(label_names: list[str], class_id: int) -> str:
    return label_names[class_id] if class_id < len(label_names) else str(class_id)


def collect_model_diagnostics(model, loader, device, label_names: list[str]) -> dict[str, Any]:
    """Aggregate logits, probabilities, and routing by true class over a loader."""
    model.eval()
    logits_by_class: dict[int, list[torch.Tensor]] = defaultdict(list)
    probabilities_by_class: dict[int, list[torch.Tensor]] = defaultdict(list)
    routing: dict[str, dict[int, dict[str, Any]]] = defaultdict(
        lambda: defaultdict(lambda: {"probability_sum": None, "token_count": 0, "assignment_counts": None})
    )

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits, aux = standardize_model_output(model(input_ids, attention_mask))
            probabilities = logits.softmax(dim=-1)
            for class_id in labels.unique().tolist():
                mask = labels == class_id
                logits_by_class[class_id].append(logits[mask].detach().cpu())
                probabilities_by_class[class_id].append(probabilities[mask].detach().cpu())

            layer_routing = aux.get("layer_routing", {})
            if not layer_routing and aux.get("probabilities") is not None:
                layer_routing = {"classifier_moe": aux}
            token_labels = labels.unsqueeze(1).expand_as(attention_mask)
            valid_tokens = attention_mask.bool()
            for layer_name, info in layer_routing.items():
                route_probabilities = info.get("probabilities")
                top_indices = info.get("top_indices")
                if route_probabilities is None or top_indices is None or route_probabilities.shape[-1] == 0:
                    continue
                if route_probabilities.dim() == 2 and route_probabilities.shape[0] == labels.shape[0]:
                    for class_id in labels.unique().tolist():
                        class_mask = labels == class_id
                        class_probabilities = route_probabilities[class_mask].detach().cpu()
                        if class_probabilities.numel() == 0:
                            continue
                        record = routing[layer_name][class_id]
                        probability_sum = class_probabilities.sum(dim=0)
                        class_indices = top_indices[class_mask].detach().cpu()
                        assignment_counts = torch.bincount(
                            class_indices.reshape(-1), minlength=route_probabilities.shape[-1]
                        )
                        record["probability_sum"] = (
                            probability_sum
                            if record["probability_sum"] is None
                            else record["probability_sum"] + probability_sum
                        )
                        record["assignment_counts"] = (
                            assignment_counts
                            if record["assignment_counts"] is None
                            else record["assignment_counts"] + assignment_counts
                        )
                        record["token_count"] += int(class_probabilities.shape[0])
                    continue
                route_probabilities = route_probabilities.reshape(
                    labels.shape[0], attention_mask.shape[1], -1
                )
                selected_mask = info.get("selected_mask")
                if selected_mask is not None:
                    assignments = selected_mask.reshape(
                        labels.shape[0], attention_mask.shape[1], -1
                    )
                else:
                    top_indices = top_indices.reshape(labels.shape[0], attention_mask.shape[1], -1)
                for class_id in labels.unique().tolist():
                    mask = valid_tokens & (token_labels == class_id)
                    class_probabilities = route_probabilities[mask].detach().cpu()
                    if class_probabilities.numel() == 0:
                        continue
                    record = routing[layer_name][class_id]
                    probability_sum = class_probabilities.sum(dim=0)
                    if selected_mask is not None:
                        assignment_counts = assignments[mask].detach().cpu().sum(dim=0).long()
                    else:
                        class_indices = top_indices[mask].detach().cpu()
                        assignment_counts = torch.bincount(
                            class_indices.reshape(-1), minlength=route_probabilities.shape[-1]
                        )
                    record["probability_sum"] = (
                        probability_sum
                        if record["probability_sum"] is None
                        else record["probability_sum"] + probability_sum
                    )
                    record["assignment_counts"] = (
                        assignment_counts
                        if record["assignment_counts"] is None
                        else record["assignment_counts"] + assignment_counts
                    )
                    record["token_count"] += int(class_probabilities.shape[0])

    result: dict[str, Any] = {"logits_by_true_class": {}, "probabilities_by_true_class": {}, "routing_by_true_class": {}}
    for class_id in sorted(logits_by_class):
        name = _class_names(label_names, class_id)
        result["logits_by_true_class"][name] = torch.cat(logits_by_class[class_id]).mean(dim=0).tolist()
        result["probabilities_by_true_class"][name] = torch.cat(probabilities_by_class[class_id]).mean(dim=0).tolist()

    for layer_name, class_records in routing.items():
        result["routing_by_true_class"][layer_name] = {}
        for class_id, record in sorted(class_records.items()):
            name = _class_names(label_names, class_id)
            count = max(record["token_count"], 1)
            result["routing_by_true_class"][layer_name][name] = {
                "mean_routing_probabilities": (record["probability_sum"] / count).tolist(),
                "assignment_fractions": (record["assignment_counts"].float() / record["assignment_counts"].sum().clamp_min(1)).tolist(),
                "token_count": record["token_count"],
            }
    return result
