"""Huấn luyện encoder dense với head original hoặc moe."""
import argparse
from src.training.engine import apply_training_profile, run_training
from src.utils import apply_overrides, load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train a dense encoder with an original or MoE classification head.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", help="Unique run-NNN ID; required when CSV tracking is disabled.")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--smoke-test", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args(argv)
    config = apply_overrides(load_config(args.config), args.overrides)
    smoke = apply_training_profile(config, args.smoke_test)
    config["profile"] = "tiny_overfit" if config.get("diagnostics", {}).get("tiny_overfit", {}).get("enabled") else ("smoke" if smoke else "full")
    return run_training(config, args.config, args.overrides, args.run_id)


if __name__ == "__main__":
    main()
