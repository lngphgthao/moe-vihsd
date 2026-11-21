"""Tổng hợp artifact hiện hành; không đoán metadata từ tên thư mục."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev


def collect_runs(root):
    rows = []
    for path in sorted(Path(root).glob("*/manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        row = {key: manifest.get(key) for key in ("run_id", "experiment_id", "seed", "status", "profile", "verification")}
        row["config_fingerprint"] = manifest["config_fingerprint"]
        row["backbone"] = manifest["model"]["backbone"]
        row["head"] = manifest["model"]["head"]["type"]
        summary_path = path.parent / "summary.json"
        test_path = path.parent / "evaluation/test.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        test = json.loads(test_path.read_text(encoding="utf-8")) if test_path.exists() else {}
        for split, metrics in (("validation", summary.get("validation", {})), ("test", test)):
            for key in ("macro_f1", "accuracy", "weighted_f1", "loss"):
                row[f"{split}_{key}"] = metrics.get(key)
        rows.append(row)
    return rows


def aggregate(rows):
    groups = {}
    for row in rows:
        if row["status"] != "completed" or row["profile"] != "full":
            continue
        key = row["experiment_id"], row["config_fingerprint"]
        group = groups.setdefault(key, [])
        if any(previous["seed"] == row["seed"] for previous in group):
            raise ValueError(f"Multiple completed runs for the same experiment/config/seed: {key}, seed={row['seed']}")
        group.append(row)
    result = []
    for (experiment, fingerprint), group in sorted(groups.items()):
        record = {"experiment_id": experiment, "config_fingerprint": fingerprint,
                  "seeds": ",".join(str(r["seed"]) for r in sorted(group, key=lambda r: r["seed"]))}
        for split in ("validation", "test"):
            scores = [r[f"{split}_macro_f1"] for r in group if r[f"{split}_macro_f1"] is not None]
            record[f"{split}_n"] = len(scores)
            record[f"{split}_mean_macro_f1"] = mean(scores) if scores else None
            record[f"{split}_sample_sd"] = stdev(scores) if len(scores) > 1 else None
        result.append(record)
    return result


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Aggregate current run artifacts by experiment and configuration.")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--output-dir", default="runs/analysis")
    args = parser.parse_args(argv)
    rows = collect_runs(args.runs_dir)
    grouped = aggregate(rows)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "runs.csv", rows)
    write_csv(output / "summary.csv", grouped)
    lines = ["| Experiment | Validation n | Mean macro F1 | Sample SD | Test n |", "|---|---:|---:|---:|---:|"]
    def score(value):
        return "—" if value is None else f"{value:.6f}"
    for row in grouped:
        lines.append(f"| {row['experiment_id']} | {row['validation_n']} | {score(row['validation_mean_macro_f1'])} | {score(row['validation_sample_sd'])} | {row['test_n']} |")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Read {len(rows)} runs; wrote {len(grouped)} groups to {output}")
