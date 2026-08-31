# ViHSD Mixture of Experts experiment

## Colab

Open `main.ipynb` in Google Colab. The notebook mounts Drive, installs `requirements.txt`, runs `train.py`, and runs `evaluate.py`.

For repeated experiments, edit the `EXPERIMENT_OVERRIDES` cell in the notebook instead of editing and pushing `configs/vihsd.yaml`. It passes only the changed values to the training command, so the versioned YAML remains the shared baseline. Each checkpoint folder stores the exact `resolved_config.yaml`, and evaluation automatically uses it.

## Experiment overrides

An override uses the same path as a value in `configs/vihsd.yaml`, with sections separated by dots. The YAML file is never modified: omitted values keep their defaults.

For example, this Colab configuration compares an 8-expert, top-2 router against the default 4-expert, top-1 model:

```python
EXPERIMENT_OVERRIDES = {
    "training.epochs": 10,
    "training.learning_rate": 0.0001,
    "model.num_experts": 8,
    "model.top_k": 2,
}
RUN_ID = "experts-8-topk-2-lr-1e-4"
SMOKE_TEST = False
```

Set `EXPERIMENT_OVERRIDES = {}` to run the unmodified YAML defaults. Use a unique `RUN_ID` for a readable experiment folder, or set it to `None` for an automatic UTC timestamp. Keep `model.top_k` no greater than `model.num_experts`.

### What to tune first

| Priority | Settings                                                                      | Compact guidance                                                                                                                                                   |
| -------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1        | `training.learning_rate`, `training.epochs`                                   | Tune learning rate first (`5e-5`, `1e-4`, `2e-4` are useful starting points), then train long enough for validation performance to plateau.                        |
| 2        | `model.model_dim`, `model.num_layers`, `dataset.max_length`                   | Main model/context capacity. Increasing them may improve accuracy, but costs GPU memory and training time. `model_dim` must be divisible by `num_attention_heads`. |
| 3        | `training.weight_decay`, `model.dropout`                                      | Regularization. Increase if training metrics improve while validation metrics worsen.                                                                              |
| 4        | `model.num_experts`, `model.top_k`, `routing.load_balance_loss_factor`        | MoE behavior. Start at `4/1/0.01`; test `8/1` then `8/2`. Keep `1 <= top_k <= num_experts`; increase balance loss if routing collapses to a few experts.           |
| 5        | `model.expert_hidden_dim`, `model.num_attention_heads`, `training.batch_size` | Secondary capacity/optimization controls. Larger batch sizes may require learning-rate retuning.                                                                   |

`seed` affects repeatability, not the expected average score; use several seeds when comparing final candidates. `max_train_samples` and `smoke_test` are for fast debugging rather than final experiments. `num_workers`, output paths, and W&B settings do not change model quality. `routing.capacity_factor` is currently not used by the code, so changing it has no effect.

## Local commands

```bash
pip install -r requirements.txt
python train.py --config configs/vihsd.yaml
python evaluate.py --config configs/vihsd.yaml
```

To override values for one run without changing the file, repeat `--set` with a dotted YAML key:

```bash
python train.py --config configs/vihsd.yaml --set training.epochs=10 --set training.learning_rate=0.0001 --set model.num_experts=8
```

Values are parsed as YAML, so use `true`, `false`, `null`, numbers, quoted strings, or YAML lists as appropriate. Only existing keys can be overridden; this catches misspellings before training begins.

Each training run receives a unique Hanoi-time (`UTC+07:00`) timestamp and profile identifier, for example:

```text
20260822T213015+0700-full
20260822T214420+0700-smoke
```

Checkpoints and results are stored in matching run folders. The latest run is recorded in `checkpoints/latest_run.json` and `results/latest_run.json`, so evaluation without extra options uses the newest run.

Each completed training run also writes `results/<run-id>/run_metrics.json`. It records loss, accuracy, macro-F1, per-class precision/recall/F1, confusion matrices, and routing counts from the best-validation epoch and the final test evaluation. `training_history.json` retains the metrics for every epoch. `run_metadata.json` records the Git commit, hardware, split sizes, label counts, and any training class weights. `hyperparameters.json` records the resolved experiment settings both as a nested object and as `flat_hyperparameters` with dotted keys (such as `model.num_experts`), so runs are easy to diff or compare programmatically. It excludes paths and authentication settings. The best checkpoint is selected by `training.selection_metric` (macro-F1 by default); use test metrics only to report a final model, not to choose hyperparameters.

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

## Experiment workflow

Use validation macro-F1 (the default `training.selection_metric`) to select configurations; reserve test metrics for the final report. Every run now records class-wise metrics, confusion matrices, routing counts, data-label counts, hardware, and the Git commit under `results/<run-id>/`.

Run a randomized search with the supplied conservative search space:

```bash
python search.py --space configs/search.yaml
```

It writes a macro-F1-ranked `results/search-*.json` summary. Search runs set `training.evaluate_test: false`, keeping the test split unseen during selection. After selecting the best three candidates, rerun each candidate with `--set seed=42`, `--set seed=123`, and `--set seed=456`; then enable test evaluation once for the final selected configuration. To create the dense ablation (no MoE), use `--set model.use_moe=false` while keeping all other settings fixed.
