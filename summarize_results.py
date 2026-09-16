"""Aggregate and format experimental results across all training runs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None


def resolve_results_dir(configured_dir: str = "results") -> Path:
    return Path(os.getenv("RESULTS_DIR", configured_dir)).expanduser()


def is_ignored(path: Path) -> bool:
    """Ignore hidden files, Colab checkpoints, and cache directories."""
    return any(part.startswith(".") or part == "__pycache__" for part in path.parts)


def find_runs(results_dir: Path) -> list[tuple[Path, Path]]:
    """Find (run_dir, metrics_path) pairs under results_dir.
    
    Supports:
      1. results_dir/<folder>/run_metrics.json (direct child)
      2. results_dir/<folder>/**/run_metrics.json (nested runs/subdirectories)
      3. Fallback to metrics.json or vihsd_predictions.json
      4. results_dir itself if pointed directly at a run folder
    """
    discovered: list[tuple[Path, Path]] = []
    seen_dirs: set[Path] = set()

    if not results_dir.exists():
        return discovered

    candidate_names = ["run_metrics.json", "metrics.json", "vihsd_predictions.json"]

    # Check if results_dir itself is a single run folder
    for name in candidate_names:
        direct_file = results_dir / name
        if direct_file.is_file() and not is_ignored(direct_file):
            discovered.append((results_dir, direct_file))
            seen_dirs.add(results_dir)
            return discovered

    # Inspect each child entry
    for child in sorted(results_dir.iterdir()):
        if not child.is_dir() or is_ignored(child):
            continue

        # 1. Direct file under folder
        metrics_file = None
        for name in candidate_names:
            candidate = child / name
            if candidate.is_file() and not is_ignored(candidate):
                metrics_file = candidate
                break

        if metrics_file is not None:
            discovered.append((child, metrics_file))
            seen_dirs.add(child)
            continue

        # 2. Nested files inside child folder (e.g. results/<folder>/<run_id>/run_metrics.json)
        for name in candidate_names:
            nested_files = [p for p in sorted(child.rglob(name)) if not is_ignored(p)]
            new_found = False
            for nf in nested_files:
                parent_dir = nf.parent
                if parent_dir not in seen_dirs:
                    discovered.append((parent_dir, nf))
                    seen_dirs.add(parent_dir)
                    new_found = True
            if new_found:
                break

    return discovered


def normalize_pct(val: Any) -> float:
    """Normalize decimal metric (e.g. 0.648) to percentage (64.8) or keep percentage."""
    if val is None:
        return 0.0
    try:
        f = float(val)
    except (ValueError, TypeError):
        return 0.0
    if 0.0 < f <= 1.0:
        return f * 100.0
    return f


def infer_metadata_from_name(name: str) -> dict[str, Any]:
    """Fallback metadata inference from run folder name."""
    lower = name.lower()
    meta: dict[str, Any] = {}

    if "dense" in lower:
        meta["architecture"] = "dense_phobert"
        meta["experts"] = "-"
        meta["top_k"] = "-"
    elif "moe" in lower:
        meta["architecture"] = "phobert_moe"
        meta["experts"] = 4
        meta["top_k"] = 1

    # Extract num_experts (e.g. nex8, nex2, 8experts, 2experts)
    nex_match = re.search(r"nex(\d+)|(\d+)experts?", lower)
    if nex_match:
        meta["experts"] = int(nex_match.group(1) or nex_match.group(2))

    # Extract top_k (e.g. topk1, topk2, top-1, top-2)
    topk_match = re.search(r"top[-_]?k?(\d+)", lower)
    if topk_match:
        meta["top_k"] = int(topk_match.group(1))

    # Extract loss type
    if "wce-balanced" in lower:
        meta["loss_type"] = "weighted_ce (balanced)"
    elif "wce-sqrt" in lower:
        meta["loss_type"] = "weighted_ce (sqrt)"
    elif "focal" in lower:
        meta["loss_type"] = "focal"
    elif "ce" in lower:
        meta["loss_type"] = "cross_entropy"

    return meta


def parse_run(run_dir: Path, metrics_path: Path, results_root: Path) -> dict[str, Any] | None:
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    # Determine display run ID from directory path relative to results root
    rel_path = run_dir.relative_to(results_root).as_posix()
    run_id = rel_path if rel_path != "." else (metrics.get("run_id") or run_dir.name)

    # 1. Look for hyperparameters / configs
    params: dict[str, Any] = {}
    for cand_name in ["hyperparameters.json", "resolved_config.yaml", "config.yaml"]:
        cand_path = run_dir / cand_name
        if cand_path.exists():
            try:
                if cand_name.endswith(".json"):
                    params = json.loads(cand_path.read_text(encoding="utf-8"))
                elif cand_name.endswith((".yaml", ".yml")) and yaml is not None:
                    loaded_yaml = yaml.safe_load(cand_path.read_text(encoding="utf-8"))
                    if isinstance(loaded_yaml, dict):
                        params = {
                            "architecture": loaded_yaml.get("model", {}).get("architecture"),
                            "flat_hyperparameters": {
                                "model.architecture": loaded_yaml.get("model", {}).get("architecture"),
                                "model.num_experts": loaded_yaml.get("model", {}).get("num_experts"),
                                "model.top_k": loaded_yaml.get("model", {}).get("top_k"),
                                "training.loss_type": loaded_yaml.get("training", {}).get("loss_type"),
                            },
                        }
                break
            except Exception:
                pass

    flat_params = params.get("flat_hyperparameters", {})
    inferred = infer_metadata_from_name(run_id)

    arch = params.get("architecture") or flat_params.get("model.architecture") or inferred.get("architecture", "phobert_moe" if "moe" in run_id else "unknown")
    num_experts = flat_params.get("model.num_experts") or inferred.get("experts", ("-" if "dense" in str(arch) else 4))
    top_k = flat_params.get("model.top_k") or inferred.get("top_k", ("-" if "dense" in str(arch) else 1))
    loss_type = flat_params.get("training.loss_type") or inferred.get("loss_type", "cross_entropy")
    profile = params.get("profile", "full")
    best_epoch = metrics.get("best_epoch", metrics.get("epoch", "-"))

    # 2. Extract metrics (support test, validation, and flat metrics schemas)
    test_metrics = metrics.get("test") or {}
    val_metrics = metrics.get("validation") or metrics.get("val") or {}

    # If metrics is flat (e.g. from evaluate.py/predictions)
    if not test_metrics and ("macro_f1" in metrics or "accuracy" in metrics):
        test_metrics = metrics

    test_macro_f1 = normalize_pct(test_metrics.get("macro_f1", 0.0))
    val_macro_f1 = normalize_pct(val_metrics.get("macro_f1", 0.0))

    # Overall macro_f1 for sorting and primary reporting
    macro_f1 = test_macro_f1 if test_macro_f1 > 0 else val_macro_f1

    acc = normalize_pct(test_metrics.get("accuracy", val_metrics.get("accuracy", 0.0)))
    weighted_f1 = normalize_pct(test_metrics.get("weighted_f1", val_metrics.get("weighted_f1", 0.0)))

    per_class = test_metrics.get("per_class_f1") or val_metrics.get("per_class_f1") or {}
    f1_clean = normalize_pct(per_class.get("0", per_class.get(0, per_class.get("CLEAN", per_class.get("clean", 0.0)))))
    f1_offensive = normalize_pct(per_class.get("1", per_class.get(1, per_class.get("OFFENSIVE", per_class.get("offensive", 0.0)))))
    f1_hate = normalize_pct(per_class.get("2", per_class.get(2, per_class.get("HATE", per_class.get("hate", 0.0)))))

    routing = metrics.get("routing_diagnostics", {})
    avg_entropy = None
    if isinstance(routing, dict):
        entropies = [v["routing_entropy"] for v in routing.values() if isinstance(v, dict) and "routing_entropy" in v]
        if entropies:
            avg_entropy = sum(entropies) / len(entropies)

    return {
        "run_id": run_id,
        "profile": profile,
        "architecture": arch,
        "experts": num_experts,
        "top_k": top_k,
        "loss_type": loss_type,
        "epoch": best_epoch,
        "routing_entropy": avg_entropy,
        "val_macro_f1": val_macro_f1,
        "test_macro_f1": test_macro_f1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "accuracy": acc,
        "f1_clean": f1_clean,
        "f1_offensive": f1_offensive,
        "f1_hate": f1_hate,
    }


def format_markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "Run ID",
        "Architecture",
        "E",
        "Top-k",
        "Loss",
        "Epoch",
        "Val F1",
        "Test Macro F1",
        "Weighted F1",
        "Accuracy",
        "F1 (Clean)",
        "F1 (Off)",
        "F1 (Hate)",
        "Entropy",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in rows:
        entropy_val = r.get("routing_entropy")
        entropy_text = "-" if entropy_val is None else f"{entropy_val:.3f}"
        
        val_f1_text = f"{r['val_macro_f1']:.2f}%" if r.get("val_macro_f1", 0) > 0 else "-"
        test_f1_val = r.get("test_macro_f1", 0)
        test_f1_text = f"**{test_f1_val:.2f}%**" if test_f1_val > 0 else "-"

        line = (
            f"| `{r['run_id']}` "
            f"| `{r['architecture']}` "
            f"| {r['experts']} "
            f"| {r['top_k']} "
            f"| {r['loss_type']} "
            f"| {r['epoch']} "
            f"| {val_f1_text} "
            f"| {test_f1_text} "
            f"| {r['weighted_f1']:.2f}% "
            f"| {r['accuracy']:.2f}% "
            f"| {r['f1_clean']:.2f}% "
            f"| {r['f1_offensive']:.2f}% "
            f"| {r['f1_hate']:.2f}% "
            f"| {entropy_text} |"
        )
        lines.append(line)
    return "\n".join(lines)


def format_latex_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{l l c c c c c c c c c}",
        r"\toprule",
        r"\textbf{Run ID} & \textbf{Architecture} & \textbf{E} & \textbf{Top-k} & \textbf{Loss} & \textbf{Val F1} & \textbf{Macro F1} & \textbf{W-F1} & \textbf{Acc} & \textbf{F1-Off} & \textbf{F1-Hate} \\",
        r"\midrule",
    ]
    for r in rows:
        val_f1_text = f"{r['val_macro_f1']:.2f}" if r.get("val_macro_f1", 0) > 0 else "-"
        macro_f1_val = r.get("test_macro_f1", 0) or r.get("macro_f1", 0)
        macro_f1_text = f"\\textbf{{{macro_f1_val:.2f}}}" if macro_f1_val > 0 else "-"

        line = (
            f"{r['run_id'].replace('_', r'\_')} & "
            f"{str(r['architecture']).replace('_', r'\_')} & "
            f"{r['experts']} & "
            f"{r['top_k']} & "
            f"{str(r['loss_type']).replace('_', r'\_')} & "
            f"{val_f1_text} & "
            f"{macro_f1_text} & "
            f"{r['weighted_f1']:.2f} & "
            f"{r['accuracy']:.2f} & "
            f"{r['f1_offensive']:.2f} & "
            f"{r['f1_hate']:.2f} \\\\"
        )
        lines.append(line)
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Experimental comparison on the ViHSD dataset. Validation Macro F1 is used for checkpoint selection; test Macro F1 is the final confirmation metric.}",
        r"\label{tab:vihsd_results}",
        r"\end{table*}",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize ViHSD experiment runs.")
    parser.add_argument("--results-dir", default="results", help="Directory containing run results.")
    parser.add_argument("--filter-profile", default=None, choices=["smoke", "full"], help="Filter by profile.")
    parser.add_argument("--sort-by", default="macro_f1", help="Column to sort by (default: macro_f1).")
    args = parser.parse_args()

    results_dir = resolve_results_dir(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    run_pairs = find_runs(results_dir)
    if not run_pairs:
        print(f"No completed runs with metrics found in {results_dir}.")
        return

    rows = []
    for run_dir, metrics_file in run_pairs:
        run_data = parse_run(run_dir, metrics_file, results_root=results_dir)
        if run_data is not None:
            if args.filter_profile and run_data.get("profile") != args.filter_profile:
                continue
            rows.append(run_data)

    if not rows:
        print(f"No valid run metrics could be parsed from {results_dir}.")
        return

    # Sort rows
    rows.sort(key=lambda r: r.get(args.sort_by, 0), reverse=True)

    md_table = format_markdown_table(rows)
    print("\n" + "=" * 80)
    print(f"EXPERIMENT RESULTS TRACKING TABLE ({len(rows)} runs found)")
    print("=" * 80)
    print(md_table)
    print("=" * 80 + "\n")

    # Save Markdown and CSV summaries in results directory
    summary_md_path = results_dir / "summary_table.md"
    summary_md_path.write_text(md_table + "\n\n### LaTeX Table:\n\n```latex\n" + format_latex_table(rows) + "\n```\n", encoding="utf-8")

    csv_path = results_dir / "summary_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Summary saved to:")
    print(f"  - Markdown: {summary_md_path}")
    print(f"  - CSV:      {csv_path}")


if __name__ == "__main__":
    main()


