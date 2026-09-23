"""Train a dense ViANLI baseline with a selectable Hugging Face encoder."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.utils import load_config


SEGMENTED_DATASET = "data/vianli_segmented"
ORIGINAL_DATASET = "uitnlp/ViANLI"


def dataset_for_model(model_name: str) -> str:
    """Use VnCoreNLP-segmented data only for PhoBERT baselines."""
    if "phobert" in model_name.lower():
        return SEGMENTED_DATASET
    return ORIGINAL_DATASET


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/vianli_dense.yaml")
    parser.add_argument("--model-name", default=None, help="Hugging Face model ID or local path")
    parser.add_argument("--pooling", choices=("cls", "mean"), default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--smoke-test", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    model_name = args.model_name or config["model"]["pretrained_model_name"]
    dataset_name = dataset_for_model(model_name)
    if dataset_name == SEGMENTED_DATASET and not Path(dataset_name).is_dir():
        raise FileNotFoundError(
            f"PhoBERT requires the segmented ViANLI dataset at '{dataset_name}'. "
            "Run: python scripts/segment_vianli.py --output-dir data/vianli_segmented"
        )

    # Reuse the main trainer so dense runs have identical selection and artifacts.
    train_args = ["train.py", "--config", args.config]
    train_args += ["--set", f"model.pretrained_model_name={model_name}"]
    train_args += ["--set", f"dataset.tokenizer={model_name}"]
    train_args += ["--set", f"dataset.name={dataset_name}"]
    if args.pooling:
        train_args += ["--set", f"model.pooling={args.pooling}"]
    if args.seed is not None:
        train_args += ["--set", f"seed={args.seed}"]
    if args.run_id:
        train_args += ["--run-id", args.run_id]
    if args.smoke_test is not None:
        train_args += ["--smoke-test" if args.smoke_test else "--no-smoke-test"]
    sys.argv = train_args
    from train import main as train_main

    train_main()


if __name__ == "__main__":
    main()