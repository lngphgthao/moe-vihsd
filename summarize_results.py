"""Aggregate and format experimental results across all training runs."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


def resolve_results_dir(configured_dir: str = "results") -> Path:
    return Path(os.getenv("RESULTS_DIR", configured_dir)).expanduser()


def parse_run(run_dir: Path) -> dict[str, Any] | None:
    metrics_path = run_dir / "run_metrics.json"
    if not metrics_path.exists():
        return None

    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    params_path = run_dir / "hyperparameters.json"
    params = {}
    if params_path.exists():
        try:
            params = json.loads(params_path.read_text(encoding="utf-8"))
        except Exception:
            params = {}

    flat_params = params.get("flat_hyperparameters", {})
    arch = params.get("architecture") or flat_params.get("model.architecture", "unknown")
    num_experts = flat_params.get("model.num_experts", "-")
    top_k = flat_params.get("model.top_k", "-")
    loss_type = flat_params.get("training.loss_type", "ce")
    profile = params.get("profile", "full")
    best_epoch = metrics.get("best_epoch", "-")

    test_metrics = metrics.get("test", {})
    acc = test_metrics.get("accuracy", 0.0) * 100
    macro_f1 = test_metrics.get("macro_f1", 0.0) * 100
    weighted_f1 = test_metrics.get("weighted_f1", 0.0) * 100

    per_class = test_metrics.get("per_class_f1", {})
    f1_clean = per_class.get("0", 0.0) * 100
    f1_offensive = per_class.get("1", 0.0) * 100
    f1_hate = per_class.get("2", 0.0) * 100

    routing = metrics.get("routing_diagnostics", {})
    entropies = [v["routing_entropy"] for v in routing.values() if isinstance(v, dict) and "routing_entropy" in v]
    avg_entropy = sum(entropies) / len(entropies) if entropies else None

    return {
        "run_id": run_dir.name,
        "profile": profile,
        "architecture": arch,
        "experts": num_experts,
        "top_k": top_k,
        "loss_type": loss_type,
        "epoch": best_epoch,
        "routing_entropy": avg_entropy,
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
        "Routing Entropy",
        "Macro F1",
        "Weighted F1",
        "Accuracy",
        "F1 (Clean)",
        "F1 (Offensive)",
        "F1 (Hate)",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in rows:
        entropy_val = r.get("routing_entropy")
        entropy_text = "n/a" if entropy_val is None else f"{entropy_val:.3f}"
        line = (
            f"| `{r['run_id']}` "
            f"| `{r['architecture']}` "
            f"| {r['experts']} "
            f"| {r['top_k']} "
            f"| {r['loss_type']} "
            f"| {r['epoch']} "
            f"| {entropy_text} "
            f"| **{r['macro_f1']:.2f}%** "
            f"| {r['weighted_f1']:.2f}% "
            f"| {r['accuracy']:.2f}% "
            f"| {r['f1_clean']:.2f}% "
            f"| {r['f1_offensive']:.2f}% "
            f"| {r['f1_hate']:.2f}% |"
        )
        lines.append(line)
    return "\n".join(lines)


def format_latex_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{l l c c c c c c c c}",
        r"\toprule",
        r"\textbf{Run ID} & \textbf{Architecture} & \textbf{E} & \textbf{Top-k} & \textbf{Loss} & \textbf{Macro F1} & \textbf{W-F1} & \textbf{Acc} & \textbf{F1-Off} & \textbf{F1-Hate} \\",
        r"\midrule",
    ]
    for r in rows:
        line = (
            f"{r['run_id'].replace('_', r'\_')} & "
            f"{r['architecture'].replace('_', r'\_')} & "
            f"{r['experts']} & "
            f"{r['top_k']} & "
            f"{r['loss_type']} & "
            f"\\textbf{{{r['macro_f1']:.2f}}} & "
            f"{r['weighted_f1']:.2f} & "
            f"{r['accuracy']:.2f} & "
            f"{r['f1_offensive']:.2f} & "
            f"{r['f1_hate']:.2f} \\\\"
        )
        lines.append(line)
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Experimental comparison on the ViHSD test set. Macro F1 is the primary evaluation metric.}",
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

    rows = []
    for child in sorted(results_dir.iterdir()):
        if child.is_dir():
            run_data = parse_run(child)
            if run_data is not None:
                if args.filter_profile and run_data["profile"] != args.filter_profile:
                    continue
                rows.append(run_data)

    if not rows:
        print(f"No completed runs with run_metrics.json found in {results_dir}.")
        return

    # Sort rows
    rows.sort(key=lambda r: r.get(args.sort_by, 0), reverse=True)

    md_table = format_markdown_table(rows)
    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS TRACKING TABLE (ViHSD Test Set)")
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

