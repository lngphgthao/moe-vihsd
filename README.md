# ViHSD Mixture of Experts Experiment

This project trains and evaluates several Mixture of Experts (MoE) architectures for ViHSD. All experiments use the same configuration-driven workflow, run IDs, checkpoints, and saved metrics so model variants can be compared consistently.

## Choose a workflow

| Workflow       | Start here                                                                    |
| -------------- | ----------------------------------------------------------------------------- |
| Kaggle         | Open [main.ipynb](main.ipynb) in Kaggle and follow the setup cells.           |
| Google Colab   | Open [main.ipynb](main.ipynb) in Colab and follow the setup cells.            |
| Local terminal | Install dependencies, authenticate Hugging Face, then run the commands below. |

### Local setup

```bash
pip install -r requirements.txt
hf auth login
python train.py --config configs/vihsd.yaml --smoke-test --run-id smoke-check
python evaluate.py --config configs/vihsd.yaml --run-id smoke-check
```

Use the smoke test to verify the environment and dataset access. For a full experiment, replace `--smoke-test` with `--no-smoke-test`.

### Kaggle setup

Open or import [main.ipynb](main.ipynb) in a Kaggle Notebook:

1. In the right-hand **Notebook settings** sidebar:
   - **Accelerator**: select **GPU P100** or **GPU T4 x2**.
   - **Internet**: toggle **Internet ON** (required for dataset download, model weights, and W&B).
2. Under **Add-ons → Secrets**, add:
   - `HF_TOKEN`: Hugging Face dataset and model access
   - `WANDB_API_KEY`: Weights & Biases logging
3. Run the setup cells in [main.ipynb](main.ipynb) to configure outputs in `/kaggle/working/checkpoints` and `/kaggle/working/results`.
4. To persist checkpoints and results permanently, use **"Save Version" → "Save & Run All (Commit)"**. Output files will be accessible under the notebook's **Output** tab.

### Colab setup

Open [main.ipynb](main.ipynb) in Google Colab. The notebook:

1. mounts Google Drive
2. clones the repository and installs dependencies
3. authenticates Hugging Face and W&B using Colab Secrets
4. runs training and evaluation commands manually

Before running the authentication cell, add these secrets in Colab's Secrets panel:

| Secret          | Used for                              |
| --------------- | ------------------------------------- |
| `HF_TOKEN`      | Hugging Face dataset and model access |
| `WANDB_API_KEY` | Weights & Biases logging              |

The credentials are read at runtime and are not stored in the notebook, YAML files, or repository.

## Model architectures

The architectures are selected through the shared model factory in [models/factory.py](models/factory.py).

Available architecture names:

- `current_moe` — the original baseline implementation
- `stronger_moe` — a stronger multi-expert variant in [models/moe_v2.py](models/moe_v2.py)
- `pretrained_backbone` — PhoBERT contextual encoder followed by the MoE block in [models/pretrained_backbone.py](models/pretrained_backbone.py)

Select an architecture in the YAML file or override it for one run. No code changes are required.

Example:

```bash
python train.py --config configs/vihsd.yaml --set model.architecture=current_moe --run-id baseline-current-moe
python train.py --config configs/vihsd.yaml --set model.architecture=stronger_moe --run-id variant-stronger-moe
python train.py --config configs/vihsd.yaml --set model.architecture=pretrained_backbone --run-id phobert-moe
```

The YAML default is:

```yaml
model:
  architecture: current_moe
```

## Run an experiment

Training and evaluation are separate commands. Always use a unique `--run-id` for a new experiment.

### Train

```bash
python train.py \
  --config configs/vihsd.yaml \
  --no-smoke-test \
  --run-id baseline-current-moe
```

### Evaluate

```bash
python evaluate.py \
  --config configs/vihsd.yaml \
  --run-id baseline-current-moe
```

Evaluation reloads the best checkpoint for that run. To evaluate a checkpoint directly:

```bash
python evaluate.py \
  --config configs/vihsd.yaml \
  --checkpoint checkpoints/<run-id>/vihsd_moe_best.safetensors
```

### Override settings for one run

Use repeated `--set section.key=value` options. The YAML file is not modified, and only existing keys can be overridden.

```bash
python train.py \
  --config configs/vihsd.yaml \
  --no-smoke-test \
  --run-id stronger-moe-v1 \
  --set model.architecture=stronger_moe \
  --set model.num_experts=8 \
  --set model.top_k=2 \
  --set training.learning_rate=0.0001 \
  --set training.loss_type=focal
```

Supported task losses are `cross_entropy`, `weighted_cross_entropy`, and `focal`.
For class-weighted CE, provide one weight per class, for example:

```bash
python train.py \
  --config configs/vihsd.yaml \
  --set training.loss_type=weighted_cross_entropy \
  --set 'training.class_weights=[1.0,1.5,2.0]' \
  --run-id weighted-ce
```

Focal loss uses `training.focal_gamma` (default `2.0`) and optionally applies the
same `training.class_weights` as its class-balancing factor:

```bash
python train.py \
  --config configs/vihsd.yaml \
  --set training.loss_type=focal \
  --set training.focal_gamma=2.0 \
  --run-id focal
```

Values are parsed as YAML. Use `true`, `false`, `null`, numbers, quoted strings, or YAML lists as needed.

### What to tune first

| Priority | Settings                                                                      | Compact guidance                                                                                                |
| -------- | ----------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| 1        | `training.learning_rate`, `training.epochs`, `training.loss_type`             | Start with conservative learning rates and use a stable task loss; focal loss is available for class imbalance. |
| 2        | `model.architecture`, `model.num_experts`, `model.top_k`                      | Compare baseline MoE vs stronger MoE using the same workflow. Keep `1 <= top_k <= num_experts`.                 |
| 3        | `model.model_dim`, `model.num_layers`, `dataset.max_length`                   | Increase capacity carefully; this raises training cost and memory use.                                          |
| 4        | `training.weight_decay`, `model.dropout`                                      | Regularization tuning is helpful when validation performance starts to diverge from training performance.       |
| 5        | `model.expert_hidden_dim`, `model.num_attention_heads`, `training.batch_size` | Secondary capacity and optimization controls.                                                                   |

## Outputs and run IDs

Each training run receives a unique Hanoi-time (`UTC+07:00`) timestamp and profile identifier, for example:

```text
20260822T213015+0700-full
20260822T214420+0700-smoke
```

Checkpoints and results are stored in matching run folders. The latest run is recorded in `checkpoints/latest_run.json` and `results/latest_run.json`, so evaluation without `--run-id` uses the newest run.

Important files in `results/<run-id>/`:

| File                     | Contents                                 |
| ------------------------ | ---------------------------------------- |
| `run_metrics.json`       | Best-validation and final test metrics   |
| `training_history.json`  | Metrics for every training epoch         |
| `hyperparameters.json`   | Resolved settings, including dotted keys |
| `vihsd_predictions.json` | Saved test predictions, when generated   |

`run_metrics.json` includes loss, accuracy, Macro F1, Weighted F1, and per-class F1. The best checkpoint is selected by validation Macro F1. Use test metrics only for reporting the final model, not for choosing hyperparameters.

To evaluate an older run, pass its identifier:

```bash
python evaluate.py --config configs/vihsd.yaml --run-id 20260822T143015Z-full
```

The YAML defaults to full training. Use `--smoke-test` for the short profile and `--no-smoke-test` to explicitly force the full profile. The smoke profile uses `training.smoke.epochs` and `training.smoke.max_train_samples`; the full profile uses the top-level `training.epochs` and `training.max_train_samples` values.

## Storage and authentication

The default checkpoint path is the local `checkpoints` folder.

- **Kaggle**: The notebook sets `CHECKPOINT_DIR` to `/kaggle/working/checkpoints` and `RESULTS_DIR` to `/kaggle/working/results`. Use **"Save Version" → "Save & Run All (Commit)"** so outputs persist in the notebook's **Output** tab. Authenticate by adding `HF_TOKEN` and `WANDB_API_KEY` under **Add-ons → Secrets**.
- **Colab**: The notebook sets `CHECKPOINT_DIR` and `RESULTS_DIR` to Google Drive folders so outputs persist after the runtime ends. Authenticate via Colab's **Secrets** panel (`HF_TOKEN` and `WANDB_API_KEY`).
- **Local terminal**:
  ```bash
  hf auth login
  wandb login --verify
  ```

Credentials are read at runtime and are not stored in the notebook, YAML files, or repository. No `.env` file is used.

## Runtime notes

- Set `logging.use_wandb: true` in the YAML to enable W&B logging.
- Tokenization uses `dataset.tokenization_num_proc` workers and the Hugging Face cache. Set it to `1` if multiprocessing is unavailable.
- Data loading defaults to `training.num_workers: 0`, which is safest in Colab/Jupyter. For a script-only local run, you can increase it with `--set training.num_workers=2`.
- Already-tokenized data is reused from cache on later runs.
- `seed` affects repeatability, not expected average performance; use several seeds for final comparisons.
- `max_train_samples` and `smoke_test` are for fast debugging, not final experiments.
- `num_workers`, output paths, and W&B settings do not change model quality.
- `routing.capacity_factor` is currently unused by the code, so changing it has no effect.
