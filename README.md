# ViHSD Mixture of Experts Experiment

This project trains and evaluates PhoBERT and Mixture of Experts (MoE) architectures for ViHSD. All experiments use the same configuration-driven workflow, run IDs, checkpoints, and saved metrics so model variants can be compared consistently.

## Choose a workflow

| Workflow       | Start here                                                                                  |
| -------------- | ------------------------------------------------------------------------------------------- |
| Kaggle         | Open [main_kaggle.ipynb](notebooks/main_kaggle.ipynb) in Kaggle and follow the setup cells. |
| Google Colab   | Open [main_colab.ipynb](notebooks/main_colab.ipynb) in Colab and follow the setup cells.    |
| Local terminal | Install dependencies, authenticate Hugging Face, then run the commands below.               |

### Local setup

```bash
pip install -r requirements.txt
hf auth login
python train.py --config configs/vihsd.yaml --smoke-test --run-id smoke-check
python evaluate.py --config configs/vihsd.yaml --run-id smoke-check
```

Use the smoke test to verify the environment and dataset access. For a full experiment, replace `--smoke-test` with `--no-smoke-test`.

### Kaggle setup

Open or import [main_kaggle.ipynb](notebooks/main_kaggle.ipynb) in a Kaggle Notebook:

1. In the right-hand **Notebook settings** sidebar:
   - **Accelerator**: select **GPU P100** or **GPU T4 x2**.
   - **Internet**: toggle **Internet ON** (required for dataset download, model weights, and W&B).
2. Under **Add-ons → Secrets**, add:
   - `HF_TOKEN`: Hugging Face dataset and model access
   - `WANDB_API_KEY`: Weights & Biases logging
3. Run the setup cells in [main_kaggle.ipynb](notebooks/main_kaggle.ipynb) to configure outputs in `/kaggle/working/checkpoints` and `/kaggle/working/results`.
4. To persist checkpoints and results permanently, use **"Save Version" → "Save & Run All (Commit)"**. Output files will be accessible under the notebook's **Output** tab.

### Colab setup

Open [main_colab.ipynb](notebooks/main_colab.ipynb) in Google Colab. The notebook:

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

The architectures are selected through the shared model factory in [src/models/factory.py](src/models/factory.py).

Available architecture names:

- `dynamic_moe`: shared PhoBERT encoder with mean pooling and per-example dynamic-threshold routing (NLIMoE-style).
- `transformer_moe` with `model.routing_method=dynamic_threshold`: token-level dynamic-threshold routing inside the selected transformer FFNs. `top_k` remains the default.

- `transformer_moe` — true token-level MoE Transformer: loads a compatible BERT/RoBERTa-family encoder and replaces selected internal FFNs with Sparse MoE layers in [src/models/moe/layer.py](src/models/moe/layer.py)
- `phobert_moe` — compatibility alias for older MoE configurations
- `dense_transformer` — generic dense classifier for any compatible Hugging Face encoder
- `dense_phobert` — compatibility alias for `dense_transformer`

Select an architecture in the YAML file or override it for one run. No code changes are required.

Example dynamic-routing run (uses validation for checkpoint selection like the other architectures):

```bash
python train.py --config configs/vihsd.yaml --no-smoke-test \
  --run-id dynamic-moe-s42 --set model.architecture=dynamic_moe --set seed=42
```

To keep the token-level Transformer MoE and switch only its router:

```bash
python train.py --config configs/vihsd.yaml --no-smoke-test \
  --run-id phobert-moe-dynamic-router-s42 \
  --set model.architecture=transformer_moe \
  --set model.routing_method=dynamic_threshold --set seed=42
```

Example:

```bash
# True MoE Transformer (Full fine-tuning, default):
python train.py --config configs/vihsd.yaml --set model.architecture=transformer_moe --run-id transformer-moe-full

# True MoE Transformer (Parameter-efficient / frozen attention):
python train.py --config configs/vihsd.yaml --set model.architecture=transformer_moe --set model.freeze_attention=true --run-id transformer-moe-peft

# Dense baseline:
python train.py --config configs/vihsd.yaml --set model.architecture=dense_phobert --set model.pooling=mean --run-id dense-phobert-integrated
```

The YAML default is:

```yaml
model:
  architecture: transformer_moe
  pretrained_model_name: vinai/phobert-base
```

All model architectures load the encoder from `model.pretrained_model_name`. Training and
evaluation automatically use the same model as the tokenizer, so changing this one value is
enough for compatible multilingual encoders:

```bash
# XLM-R MoE
python train.py --config configs/vihsd.yaml --no-smoke-test \
  --run-id xlmr-moe-s42 --set model.pretrained_model_name=xlm-roberta-base

# mBERT dense baseline
python train.py --config configs/vihsd.yaml --no-smoke-test \
  --run-id mbert-dense-s42 --set model.architecture=dense_transformer \
  --set model.pretrained_model_name=bert-base-multilingual-cased \
  --set model.pooling=cls
```

`transformer_moe` requires a BERT/RoBERTa-style encoder with `encoder.layer`; XLM-R and mBERT
are supported. Other Hugging Face encoders can be used with `dense_transformer` when they
provide `last_hidden_state` and a `hidden_size` configuration value.

## Argument reference

### Command-line arguments

| Command       | Argument                           | Default                | What it does                                                            |
| ------------- | ---------------------------------- | ---------------------- | ----------------------------------------------------------------------- |
| `train.py`    | `--config PATH`                    | `configs/vihsd.yaml`   | Loads the YAML configuration and optional `base_config`.                |
| `train.py`    | `--set SECTION.KEY=VALUE`          | none                   | Overrides an existing YAML value; repeat it for multiple overrides.     |
| `train.py`    | `--smoke-test` / `--no-smoke-test` | config value (`false`) | Selects the short validation profile or the full experiment profile.    |
| `train.py`    | `--run-id ID`                      | timestamped ID         | Names the checkpoint and results directory. Use a unique ID per run.    |
| `evaluate.py` | `--config PATH`                    | `configs/vihsd.yaml`   | Supplies the configuration used to rebuild the model and data pipeline. |
| `evaluate.py` | `--run-id ID`                      | latest run             | Evaluates a saved run and loads its resolved configuration.             |
| `evaluate.py` | `--checkpoint PATH`                | none                   | Evaluates one explicit `.safetensors` checkpoint.                       |

### Configuration arguments

These are the defaults from `configs/vihsd.yaml`. A child config can override them, and
`--set` can override them for one run.

| Key                                | Default                | What it does                                                                                                                   |
| ---------------------------------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `seed`                             | `42`                   | Seeds Python, NumPy, and PyTorch for reproducible runs.                                                                        |
| `dataset.name`                     | `data/vihsd_segmented` | Dataset path or Hugging Face dataset ID.                                                                                       |
| `dataset.tokenizer`                | `vinai/phobert-base`   | Fallback tokenizer. It is automatically replaced by `model.pretrained_model_name` when that value is set.                      |
| `dataset.max_length`               | `128`                  | Maximum token sequence length; longer inputs are truncated and shorter inputs are padded.                                      |
| `model.architecture`               | `transformer_moe`      | Selects `transformer_moe`, `dense_transformer`, or compatibility aliases.                                                      |
| `model.pretrained_model_name`      | `vinai/phobert-base`   | Hugging Face model ID or local path for the encoder and tokenizer. Change this for XLM-R or mBERT.                             |
| `model.moe_layers`                 | `[8, 9, 10, 11]`       | Zero-based encoder layers whose FFNs are replaced by MoE. Supports lists and values such as `all`, `last_4`, or `alternating`. |
| `model.num_experts`                | `4`                    | Number of routed experts per MoE layer. Set to `0` for shared-expert-only controls.                                            |
| `model.top_k`                      | `1`                    | Number of experts selected for each token.                                                                                     |
| `model.upcycle`                    | `true`                 | Initializes GELU experts from the pretrained FFN weights.                                                                      |
| `model.shared_expert`              | `false`                | Adds an always-active expert alongside routed experts.                                                                         |
| `model.expert_type`                | `gelu`                 | Expert FFN type: `gelu`, `geglu`, or `swiglu`. Gated types require `upcycle=false`.                                            |
| `model.expert_hidden_size`         | `null`                 | Expert intermediate width; `null` uses the backbone FFN width.                                                                 |
| `model.expert_init_noise`          | `0.0`                  | Adds Gaussian noise to expert initialization after upcycling.                                                                  |
| `model.learnable_residual_scale`   | `false`                | Learns the MoE residual multiplier instead of keeping it fixed.                                                                |
| `model.residual_scale_init`        | `1.0`                  | Initial value of the MoE residual multiplier.                                                                                  |
| `model.freeze_attention`           | `false`                | Freezes attention modules and dense FFNs in non-MoE layers.                                                                    |
| `model.freeze_embeddings`          | `false`                | Freezes the backbone embedding parameters.                                                                                     |
| `model.freeze_backbone`            | `false`                | Freezes the complete encoder for dense or sentence-level classifiers.                                                          |
| `model.pooling`                    | `cls`                  | Sequence pooling: `cls` uses the first token; `mean` averages non-padding tokens.                                              |
| `model.dropout`                    | `0.1`                  | Dropout used by classifier and legacy MoE heads.                                                                               |
| `training.epochs`                  | `5`                    | Number of full training epochs.                                                                                                |
| `training.batch_size`              | `16`                   | Examples processed per optimization step.                                                                                      |
| `training.learning_rate`           | `0.00002`              | AdamW learning rate.                                                                                                           |
| `training.weight_decay`            | `0.01`                 | AdamW L2-style weight decay.                                                                                                   |
| `training.loss_type`               | `cross_entropy`        | Task loss: `cross_entropy`, weighted cross-entropy, or focal loss.                                                             |
| `routing.load_balance_loss_factor` | `0.01`                 | Weight of the auxiliary MoE load-balancing loss. Use `0.0` for dense controls.                                                 |
| `logging.use_wandb`                | `true`                 | Enables Weights & Biases logging; set `false` when W&B is unavailable.                                                         |

For example, switching the complete pipeline to XLM-R requires only:

```bash
python train.py --config configs/vihsd.yaml --no-smoke-test \
  --run-id xlmr-moe-s42 \
  --set model.pretrained_model_name=xlm-roberta-base
```

## Run an experiment

Training and evaluation are separate commands. Always use a unique `--run-id` for a new experiment.

### Dense ViANLI baselines

Use the unified YAML group runner to compare dense Hugging Face encoders and seeds. Each run
gets a unique generated ID and uses the same trainer, validation selection, checkpoint format,
and optional W&B logging.

```bash
python scripts/run_group.py \
  --group-config configs/groups/vianli_dense_backbones.yaml \
  --no-smoke-test
```

The runner reuses the shared trainer, selects the best checkpoint by validation macro F1,
and saves the resolved config, checkpoint, history, and metrics under `checkpoints/` and
`results/`. Prepare the local dataset first when needed:

```bash
python scripts/segment_vianli.py --output-dir data/vianli_segmented
```

### Train

```bash
python train.py \
  --config configs/vihsd.yaml \
  --no-smoke-test \
  --run-id phobert-moe-integrated \
  --set model.architecture=transformer_moe
```

For the dense PhoBERT baseline, use the integrated pipeline so the dataset,
seed, optimizer, loss, validation selection, checkpoint format, and result
schema are identical to the MoE runs:

```bash
python train.py \
  --config configs/vihsd.yaml \
  --no-smoke-test \
  --run-id dense-phobert-integrated \
  --set model.architecture=dense_phobert \
  --set model.pooling=mean \
  --set training.loss_type=cross_entropy \
  --set routing.load_balance_loss_factor=0.0
```

`routing.load_balance_loss_factor=0.0` is
explicit documentation that the dense model has no routing loss; it does not
change the result because the dense model returns no auxiliary loss.

Each new run stores its checkpoint using the architecture name, for example
`dense_phobert_best.safetensors` or `phobert_moe_best.safetensors`. Evaluation
also accepts the older `vihsd_moe_best.safetensors` name, so existing runs are
still usable. To debug a run, inspect these files in order:

1. `checkpoints/<run-id>/resolved_config.yaml` for the exact configuration.
2. `checkpoints/<run-id>/*_metadata.json` for the architecture and checkpoint.
3. `results/<run-id>/hyperparameters.json` and `run_metrics.json` for settings and metrics.

Evaluate either architecture with the same command:

```bash
python evaluate.py --config configs/vihsd.yaml --run-id dense-phobert-integrated
python evaluate.py --config configs/vihsd.yaml --run-id phobert-moe-integrated
```

### Evaluate

```bash
python evaluate.py \
  --config configs/vihsd.yaml \
  --run-id phobert-moe-integrated
```

Evaluation reloads the best checkpoint for that run. To evaluate a checkpoint directly:

```bash
python evaluate.py \
  --config configs/vihsd.yaml \
  --checkpoint checkpoints/<run-id>/<architecture>_best.safetensors
```

### Override settings for one run

Use repeated `--set section.key=value` options. The YAML file is not modified, and only existing keys can be overridden.

```bash
python train.py \
  --config configs/vihsd.yaml \
  --no-smoke-test \
  --run-id phobert-moe-experiment \
  --set model.architecture=transformer_moe \
  --set model.num_experts=4 \
  --set model.top_k=1 \
  --set training.loss_type=cross_entropy
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

| Priority | Settings                                                           | Compact guidance                                                                                                |
| -------- | ------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| 1        | `training.learning_rate`, `training.epochs`, `training.loss_type`  | Start with conservative learning rates and use a stable task loss; focal loss is available for class imbalance. |
| 2        | `model.architecture`, `model.num_experts`, `model.top_k`           | Compare dense PhoBERT with PhoBERT MoE using the same workflow. Keep `1 <= top_k <= num_experts`.               |
| 3        | `model.model_dim`, `model.expert_hidden_dim`, `dataset.max_length` | Increase capacity carefully; this raises capacity and memory use.                                               |
| 4        | `training.weight_decay`, `model.dropout`                           | Regularization tuning is helpful when validation performance starts to diverge from training performance.       |
| 5        | `model.expert_hidden_dim`, `training.batch_size`                   | Secondary capacity and optimization controls.                                                                   |

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
- `routing.load_balance_loss_factor` controls the weight of the auxiliary balance loss added to the task loss during training. Set it to `0.0` to disable balancing entirely.
