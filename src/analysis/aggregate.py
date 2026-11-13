"""Aggregate và format result giữa các dense và MoE experiment group."""

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
    """Bỏ qua hidden file, Colab checkpoint và cache directory."""
    return any(part.startswith(".") or part == "__pycache__" for part in path.parts)


def find_runs(results_dir: Path) -> list[tuple[Path, Path]]:
    """Tìm các cặp (run_dir, metrics_path) trong results_dir.
    
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

    # Kiểm tra results_dir có phải một run folder đơn hay không.
    for name in candidate_names:
        direct_file = results_dir / name
        if direct_file.is_file() and not is_ignored(direct_file):
            discovered.append((results_dir, direct_file))
            seen_dirs.add(results_dir)
            return discovered

    # Kiểm tra từng entry con.
    for child in sorted(results_dir.iterdir()):
        if not child.is_dir() or is_ignored(child):
            continue

        # 1. File trực tiếp trong folder.
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

        # 2. File nested trong folder con, ví dụ results/<folder>/<run_id>/run_metrics.json.
        for name in candidate_names:
            nested_files = [p for p in sorted(child.rglob(name)) if not is_ignored(p)]
            for nf in nested_files:
                parent_dir = nf.parent
                if parent_dir not in seen_dirs:
                    discovered.append((parent_dir, nf))
                    seen_dirs.add(parent_dir)

    return discovered


def normalize_pct(val: Any) -> float | None:
    """Chuẩn hóa metric decimal, ví dụ 0.648 thành percentage 64.8."""
    if val is None:
        return None
    try:
        f = float(val)
    except (ValueError, TypeError):
        return None
    if 0.0 < f <= 1.0:
        return f * 100.0
    return f


def format_score(value: float | None, suffix: str = "") -> str:
    return "-" if value is None else f"{value:.2f}{suffix}"


def format_lr(lr: Any) -> str:
    """Format learning rate theo scientific notation dễ đọc, ví dụ 2e-5."""
    if lr is None or lr == "-":
        return "-"
    try:
        f = float(lr)
        return f"{f:.0e}".replace("e-0", "e-")
    except Exception:
        return str(lr)


def extract_run_date(run_dir: Path, metrics_path: Path, metrics: dict[str, Any], params: dict[str, Any]) -> str:
    """Lấy hoặc suy ra timestamp/date khi training chạy."""
    # 1. Kiểm tra trong metrics hoặc params dict.
    for cand in ["timestamp", "created_at", "run_date", "date", "start_time"]:
        if cand in metrics and metrics[cand]:
            return str(metrics[cand])
        if cand in params and params[cand]:
            return str(params[cand])

    # 2. Kiểm tra tên run_dir cho timestamp ISO hoặc Hanoi dạng compact.
    name = run_dir.name
    m = re.search(r"(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})", name)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)} {m.group(4)}:{m.group(5)}"
    m = re.search(r"(\d{4}-\d{2}-\d{2}[ _T]\d{2}[:.-]\d{2})", name)
    if m:
        return m.group(1).replace("T", " ")
    m = re.search(r"(\d{4}-\d{2}-\d{2})", name)
    if m:
        return m.group(1)

    # 3. Fallback về thời gian sửa file.
    try:
        from datetime import datetime
        mtime = metrics_path.stat().st_mtime
        return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "-"


def parse_run(run_dir: Path, metrics_path: Path, results_root: Path) -> dict[str, Any] | None:
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    # Xác định run ID hiển thị từ path tương đối với results root.
    rel_path = run_dir.relative_to(results_root).as_posix()
    run_id = rel_path if rel_path != "." else (metrics.get("run_id") or run_dir.name)

    # 1. Tìm hyperparameters hoặc config.
    params: dict[str, Any] = {}
    raw_config: dict[str, Any] = {}
    for cand_name in ["hyperparameters.json", "resolved_config.yaml", "config.yaml"]:
        cand_path = run_dir / cand_name
        if cand_path.exists():
            try:
                if cand_name.endswith(".json"):
                    params = json.loads(cand_path.read_text(encoding="utf-8"))
                elif cand_name.endswith((".yaml", ".yml")) and yaml is not None:
                    raw_config = yaml.safe_load(cand_path.read_text(encoding="utf-8")) or {}
                    if isinstance(raw_config, dict):
                        break
                    raw_config = {}
            except Exception:
                pass

    flat_params = params.get("flat_hyperparameters", {})
    nested_hp = params.get("hyperparameters", {})

    def recorded_value(section: str, key: str):
        values = (
            flat_params.get(f"{section}.{key}"),
            nested_hp.get(section, {}).get(key),
            raw_config.get(section, {}).get(key),
        )
        return next((value for value in values if value is not None), None)

    arch = params.get("architecture") or recorded_value("model", "architecture") or "unknown"
    num_experts = recorded_value("model", "num_experts")
    top_k = recorded_value("model", "top_k")
    freeze_attention = recorded_value("model", "freeze_attention")
    learning_rate = recorded_value("training", "learning_rate")
    pooling = recorded_value("model", "pooling")
    loss_type = recorded_value("training", "loss_type")
    profile = params.get("profile", "unknown")
    best_epoch = metrics.get("best_epoch", metrics.get("epoch", "-"))

    # Lấy ngày và giờ của run.
    run_date = extract_run_date(run_dir, metrics_path, metrics, params)

    # Evaluation ghi artifact riêng cạnh training metric.
    test_metrics = metrics.get("test") or {}
    val_metrics = metrics.get("validation") or metrics.get("val") or {}
    if not test_metrics and ("macro_f1" in metrics or "accuracy" in metrics):
        test_metrics = metrics
    predictions_path = run_dir / "vihsd_predictions.json"
    if predictions_path.exists() and predictions_path != metrics_path:
        predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
        test_metrics = predictions.get("test") or predictions

    test_macro_f1 = normalize_pct(test_metrics.get("macro_f1"))
    test_weighted_f1 = normalize_pct(test_metrics.get("weighted_f1"))
    test_acc = normalize_pct(test_metrics.get("accuracy"))
    val_macro_f1 = normalize_pct(val_metrics.get("macro_f1"))
    val_weighted_f1 = normalize_pct(val_metrics.get("weighted_f1"))
    val_acc = normalize_pct(val_metrics.get("accuracy"))

    macro_f1 = test_macro_f1 if test_macro_f1 is not None else val_macro_f1
    weighted_f1 = test_weighted_f1 if test_weighted_f1 is not None else val_weighted_f1
    acc = test_acc if test_acc is not None else val_acc

    per_class = test_metrics.get("per_class_f1") or val_metrics.get("per_class_f1") or {}
    f1_clean = normalize_pct(per_class.get("0", per_class.get(0, per_class.get("CLEAN", per_class.get("clean")))))
    f1_offensive = normalize_pct(per_class.get("1", per_class.get(1, per_class.get("OFFENSIVE", per_class.get("offensive")))))
    f1_hate = normalize_pct(per_class.get("2", per_class.get(2, per_class.get("HATE", per_class.get("hate")))))

    routing = metrics.get("routing_diagnostics", {})
    avg_entropy = None
    if isinstance(routing, dict):
        entropies = [v["routing_entropy"] for v in routing.values() if isinstance(v, dict) and "routing_entropy" in v]
        if entropies:
            avg_entropy = sum(entropies) / len(entropies)

    return {
        "run_id": run_id,
        "run_date": run_date,
        "profile": profile,
        "architecture": arch,
        "experts": num_experts if num_experts is not None else "-",
        "top_k": top_k if top_k is not None else "-",
        "freeze_attention": freeze_attention if freeze_attention is not None else "-",
        "learning_rate": format_lr(learning_rate),
        "pooling": pooling if pooling is not None else "-",
        "loss_type": loss_type if loss_type is not None else "-",
        "epoch": best_epoch,
        "val_macro_f1": val_macro_f1,
        "val_weighted_f1": val_weighted_f1,
        "val_accuracy": val_acc,
        "test_macro_f1": test_macro_f1,
        "test_weighted_f1": test_weighted_f1,
        "test_accuracy": test_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "accuracy": acc,
        "f1_clean": f1_clean,
        "f1_offensive": f1_offensive,
        "f1_hate": f1_hate,
        "routing_entropy": avg_entropy,
    }


def format_markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "Run ID",
        "Date",
        "Architecture",
        "E",
        "Top-k",
        "Freeze Attn",
        "LR",
        "Pool",
        "Loss",
        "Epoch",
        "Val F1",
        "Val W-F1",
        "Val Acc",
        "Test F1",
        "Test W-F1",
        "Test Acc",
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
        
        val_f1_text = format_score(r.get("val_macro_f1"), "%")
        val_wf1_text = format_score(r.get("val_weighted_f1"), "%")
        val_acc_text = format_score(r.get("val_accuracy"), "%")
        test_f1_text = format_score(r.get("test_macro_f1"), "%")
        if r.get("test_macro_f1") is not None:
            test_f1_text = f"**{test_f1_text}**"
        test_wf1_text = format_score(r.get("test_weighted_f1"), "%")
        test_acc_text = format_score(r.get("test_accuracy"), "%")

        line = (
            f"| `{r['run_id']}` "
            f"| {r['run_date']} "
            f"| `{r['architecture']}` "
            f"| {r['experts']} "
            f"| {r['top_k']} "
            f"| {r['freeze_attention']} "
            f"| {r['learning_rate']} "
            f"| {r['pooling']} "
            f"| {r['loss_type']} "
            f"| {r['epoch']} "
            f"| {val_f1_text} "
            f"| {val_wf1_text} "
            f"| {val_acc_text} "
            f"| {test_f1_text} "
            f"| {test_wf1_text} "
            f"| {test_acc_text} "
            f"| {format_score(r.get('f1_clean'), '%')} "
            f"| {format_score(r.get('f1_offensive'), '%')} "
            f"| {format_score(r.get('f1_hate'), '%')} "
            f"| {entropy_text} |"
        )
        lines.append(line)
    return "\n".join(lines)


def format_latex_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{l c l c c c c c c c c c c}",
        r"\toprule",
        r"\textbf{Run ID} & \textbf{Date} & \textbf{Arch} & \textbf{E} & \textbf{Top-k} & \textbf{Frz} & \textbf{LR} & \textbf{Pool} & \textbf{Loss} & \textbf{Val F1} & \textbf{Val W-F1} & \textbf{Val Acc} & \textbf{Test F1} \\",
        r"\midrule",
    ]
    for r in rows:
        escaped_run_id = str(r["run_id"]).replace("_", "\\_")
        escaped_architecture = str(r["architecture"]).replace("_", "\\_")
        escaped_loss_type = str(r["loss_type"]).replace("_", "\\_")
        val_f1_text = format_score(r.get("val_macro_f1"))
        val_wf1_text = format_score(r.get("val_weighted_f1"))
        val_acc_text = format_score(r.get("val_accuracy"))
        macro_f1_text = format_score(r.get("test_macro_f1"))
        if r.get("test_macro_f1") is not None:
            macro_f1_text = f"\\textbf{{{macro_f1_text}}}"

        line = (
            f"{escaped_run_id} & "
            f"{r['run_date']} & "
            f"{escaped_architecture} & "
            f"{r['experts']} & "
            f"{r['top_k']} & "
            f"{str(r['freeze_attention'])} & "
            f"{r['learning_rate']} & "
            f"{r['pooling']} & "
            f"{escaped_loss_type} & "
            f"{val_f1_text} & "
            f"{val_wf1_text} & "
            f"{val_acc_text} & "
            f"{macro_f1_text} \\\\"
        )
        lines.append(line)
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Experimental comparison on the ViHSD dataset. Validation metrics guide model selection; Test Macro F1 is the confirmation metric.}",
        r"\label{tab:vihsd_results}",
        r"\end{table*}",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate experiment runs into CSV and Markdown.")
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

    # Sort row.
    rows.sort(key=lambda r: r.get(args.sort_by) if r.get(args.sort_by) is not None else float("-inf"), reverse=True)

    md_table = format_markdown_table(rows)
    print("\n" + "=" * 80)
    print(f"EXPERIMENT RESULTS TRACKING TABLE ({len(rows)} runs found)")
    print("=" * 80)
    print(md_table)
    print("=" * 80 + "\n")

    # Lưu summary Markdown và CSV trong results directory.
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



