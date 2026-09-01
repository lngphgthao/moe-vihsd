# ViHSD Mixture of Experts experiment

This project supports a consistent experiment workflow for multiple model variants while keeping the original MoE baseline as the default. The current training and evaluation pipeline is built around a shared config, a run-id directory, and saved metadata so each architecture is easy to compare later.

## Architecture workflow

The repo now supports multiple model architectures through a single model factory in [models/factory.py](models/factory.py).

Available architecture names:

- `current_moe` — the original baseline implementation
- `stronger_moe` — a stronger multi-expert variant in [models/moe_v2.py](models/moe_v2.py)

You can switch architectures through the config or by `--set` overrides without editing the code path used by training and evaluation.

Example:

```bash
python train.py --config configs/vihsd.yaml --set model.architecture=current_moe --run-id baseline-current-moe
python train.py --config configs/vihsd.yaml --set model.architecture=stronger_moe --run-id variant-stronger-moe
```

The YAML default is:

```yaml
model:
  architecture: current_moe
```

## Colab

Open `main.ipynb` in Google Colab. The notebook mounts Drive, installs `requirements.txt`, runs `train.py`, and runs `evaluate.py`.

For repeated experiments, edit the `EXPERIMENT_OVERRIDES` cell in the notebook instead of editing and pushing `configs/vihsd.yaml`. It passes only the changed values to the training command, so the versioned YAML remains the shared baseline. Each checkpoint folder stores the exact `resolved_config.yaml`, and evaluation automatically uses it.

## Experiment overrides

An override uses the same path as a value in `configs/vihsd.yaml`, with sections separated by dots. The YAML file is never modified: omitted values keep their defaults.

For example, this Colab configuration compares the stronger MoE variant with the default baseline:

```python
EXPERIMENT_OVERRIDES = {
    "model.architecture": "stronger_moe",
    "training.epochs": 10,
    "training.learning_rate": 0.0001,
    "model.num_experts": 8,
    "model.top_k": 2,
}
RUN_ID = "stronger-moe-8-2"
SMOKE_TEST = False
```

Set `EXPERIMENT_OVERRIDES = {}` to run the unmodified YAML defaults. Use a unique `RUN_ID` for a readable experiment folder, or set it to `None` for an automatic UTC timestamp.

### What to tune first

| Priority | Settings                                                                      | Compact guidance                                                                                                |
| -------- | ----------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| 1        | `training.learning_rate`, `training.epochs`, `training.loss_type`             | Start with conservative learning rates and use a stable task loss; focal loss is available for class imbalance. |
| 2        | `model.architecture`, `model.num_experts`, `model.top_k`                      | Compare baseline MoE vs stronger MoE using the same workflow. Keep `1 <= top_k <= num_experts`.                 |
| 3        | `model.model_dim`, `model.num_layers`, `dataset.max_length`                   | Increase capacity carefully; this raises training cost and memory use.                                          |
| 4        | `training.weight_decay`, `model.dropout`                                      | Regularization tuning is helpful when validation performance starts to diverge from training performance.       |
| 5        | `model.expert_hidden_dim`, `model.num_attention_heads`, `training.batch_size` | Secondary capacity and optimization controls.                                                                   |

`seed` affects repeatability, not the expected average score; use several seeds when comparing final candidates. `max_train_samples` and `smoke_test` are for fast debugging rather than final experiments. `num_workers`, output paths, and W&B settings do not change model quality. `routing.capacity_factor` is currently not used by the code, so changing it has no effect.

## Local commands

```bash
pip install -r requirements.txt
python train.py --config configs/vihsd.yaml
python evaluate.py --config configs/vihsd.yaml
```

To override values for one run without changing the file, repeat `--set` with a dotted YAML key:

```bash
python train.py --config configs/vihsd.yaml --set model.architecture=stronger_moe --set training.epochs=10 --set training.learning_rate=0.0001 --set model.num_experts=8
```

Values are parsed as YAML, so use `true`, `false`, `null`, numbers, quoted strings, or YAML lists as appropriate. Only existing keys can be overridden; this catches misspellings before training begins.

Each training run receives a unique Hanoi-time (`UTC+07:00`) timestamp and profile identifier, for example:

```text
20260822T213015+0700-full
20260822T214420+0700-smoke
```

Checkpoints and results are stored in matching run folders. The latest run is recorded in `checkpoints/latest_run.json` and `results/latest_run.json`, so evaluation without extra options uses the newest run.

Each completed training run writes `results/<run-id>/run_metrics.json`. It records train and validation loss, accuracy, Macro F1, Weighted F1, and per-class F1 from the best-validation epoch (selected by validation Macro F1), plus test metrics after that best checkpoint is reloaded. `training_history.json` retains the train and validation loss, accuracy, and F1 metrics for every epoch. `hyperparameters.json` records the resolved experiment settings both as a nested object and as `flat_hyperparameters` with dotted keys, so runs are easy to diff or compare programmatically. It excludes paths and authentication settings. Use test metrics to report a final model, not to choose hyperparameters.

To evaluate an older run, pass its identifier:

```bash
python evaluate.py --config configs/vihsd.yaml --run-id 20260822T143015Z-full
```

You can also provide a direct checkpoint path with `--checkpoint`.

The YAML defaults to full training. Run a quick smoke test without editing the YAML:

```bash
python train.py --config configs/vihsd.yaml --smoke-test
```

To explicitly force the full profile, use:

```bash
python train.py --config configs/vihsd.yaml --no-smoke-test
```

The smoke profile is defined under `training.smoke` in the YAML. The normal profile uses the top-level `training.epochs` and `training.max_train_samples` values.

The default checkpoint path is the local `checkpoints` folder. In Colab, set `CHECKPOINT_DIR` to a Google Drive folder so the same scripts persist runs outside the temporary runtime. `RESULTS_DIR` can override the results location in the same way.

Hugging Face authentication is loaded from the local `.env` file using the `HF_TOKEN` variable. Keep `.env` private and create it with:

```text
HF_TOKEN=your_hugging_face_token
```

Tokenization uses the configured `dataset.tokenization_num_proc` workers and Hugging Face's cache. Set it to `1` if multiprocessing is unavailable in your environment. Data loading defaults to `training.num_workers: 0`, which avoids PyTorch worker-cleanup errors in Colab/Jupyter; for a script-only local run, you can increase it with `--set training.num_workers=2`. Already-tokenized data is reused from cache on later runs.

Set `logging.use_wandb: true` in the YAML to enable logging. In Colab, create a Google Secret named `WANDB_API_KEY`; the notebook loads it into the runtime and verifies the W&B login before training. Keep this key out of the notebook, YAML, and Git repository. For local runs, authenticate once with `wandb login --verify`.

## Multi-variant comparison workflow

To keep experiments comparable, follow the same order for every run:

1. choose one architecture (`current_moe` or `stronger_moe`)
2. set a unique `--run-id`
3. save the resolved config automatically in the checkpoint folder
4. evaluate with the same `evaluate.py` command
5. compare the `run_metrics.json` files side by side

Examples:

```bash
python train.py --config configs/vihsd.yaml --set model.architecture=current_moe --run-id comparison-baseline
python train.py --config configs/vihsd.yaml --set model.architecture=stronger_moe --run-id comparison-stronger
python evaluate.py --config configs/vihsd.yaml --run-id comparison-baseline
python evaluate.py --config configs/vihsd.yaml --run-id comparison-stronger
```

This preserves a single workflow for all variants and makes rollback or side-by-side comparison easy.
