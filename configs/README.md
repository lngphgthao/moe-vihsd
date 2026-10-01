# Configuration Guide

This directory contains all experiment definitions for the repository. YAML is the source of
truth for dataset selection, model construction, training settings, routing, diagnostics,
output paths, and W&B logging.

The Python entry points do not silently invent experiment settings. A run is defined by:

1. a base YAML file;
2. optional `base_config` inheritance;
3. explicit YAML values in the selected file;
4. optional command-line `--set section.key=value` overrides.

The resolved configuration is saved with every run and is the authoritative record of what was
executed.

## Directory Structure

```text
configs/
  README.md                         This policy and file map
  vihsd.yaml                        ViHSD general-purpose base run
  vianli.yaml                       ViANLI general-purpose base run
  architectures/
    vihsd_classifier_moe.yaml       ViHSD Classifier-MoE overlay
  baselines/
    vianli_dense.yaml               One dense ViANLI baseline definition
  finalists/
    vianli_best.yaml                Frozen documented ViANLI finalist
    vihsd_best.yaml                 Frozen documented ViHSD finalist
  workflows/
    vianli_stages.yaml              Reserved ViANLI stage definitions
  groups/
    vianli_stage1_dense_backbones.yaml  Multi-backbone and multi-seed group
  stages/
    vihsd/                          ViHSD architecture/loss overlays
  diagnostics/                      Opt-in debugging and overfit configs
```

There is no active `configs/experiments/` or `configs/vihsd_experiments/` directory. ViHSD
overlays are under `configs/stages/vihsd/`.

## Which Config To Use

| Purpose                     | Use                                                 | When                                                                |
| --------------------------- | --------------------------------------------------- | ------------------------------------------------------------------- |
| ViHSD normal run            | `configs/vihsd.yaml`                                | Default ViHSD training or a new controlled variant                  |
| ViANLI normal run           | `configs/vianli.yaml`                               | Default ViANLI dense/MoE experiments                                |
| One dense ViANLI run        | `configs/baselines/vianli_dense.yaml`               | Debugging or running one explicitly chosen backbone                 |
| Dense backbone comparison   | `configs/groups/vianli_stage1_dense_backbones.yaml` | Run several backbones with common seeds                             |
| Frozen ViANLI finalist      | `configs/finalists/vianli_best.yaml`                | Reproduce the documented best ViANLI configuration                  |
| Frozen ViHSD finalist       | `configs/finalists/vihsd_best.yaml`                 | Reproduce the documented best ViHSD configuration                   |
| Classifier-level MoE        | `configs/architectures/vihsd_classifier_moe.yaml`   | Keep the encoder dense and route the pooled representation          |
| ViHSD architecture ablation | `configs/stages/vihsd/`                             | Compare one controlled architecture factor at a time                |
| Tiny overfit/debugging      | `configs/diagnostics/`                              | Investigate label, loss, routing, or collapse behavior              |
| Research stage definition   | `configs/workflows/vianli_stages.yaml`              | Record candidate values and stage gates; it is not a trainer config |

## Architecture Names

Use these canonical names in new files:

| Value               | Meaning                                                            |
| ------------------- | ------------------------------------------------------------------ |
| `dense_transformer` | Dense Hugging Face encoder and classifier                          |
| `transformer_moe`   | Selected internal Transformer FFNs replaced by token-level MoE     |
| `nlimoe`            | Dense encoder with sequence-level dynamic-threshold routing        |
| `classifier_moe`    | Dense encoder with a pooled-representation MoE classification head |
| `hybrid_moe`        | Transformer-MoE encoder followed by a Classifier-MoE head          |

Compatibility aliases retained for older runs:

- `dense_phobert` maps to `dense_transformer`.
- `phobert_moe` maps to `transformer_moe`.
- `classifier_moe_phobert` maps to `classifier_moe`.

Use the canonical names for new experiments even when the selected backbone is PhoBERT.

## Inheritance Rules

`base_config` is resolved by `src/utils/config.py`. The loader first resolves a path relative to
the child config and then falls back to the current working directory.

Use inheritance as follows:

- Base task files define complete runnable defaults: `vihsd.yaml` and `vianli.yaml`.
- Single-factor overlays inherit the nearest task base and override only the changed factor.
- Frozen `*_best.yaml` files may inherit a task base, but must preserve the exact documented
  finalist settings.
- Group configs inherit a task or single-run config and apply per-run overrides.
- Diagnostic configs inherit a task config and change only diagnostic controls such as sample
  count, epochs, or diagnostic collection flags.

Do not copy an entire base configuration into an overlay. Do not use `*_best.yaml` as a mutable
scratch file.

Example overlay:

```yaml
base_config: ../vihsd.yaml

model:
  architecture: transformer_moe
  top_k: 2

routing:
  load_balance_loss_factor: 0.01
```

## Configuration Rules

### Dataset and labels

- Keep dataset columns and split names explicit.
- For NLI, set `input_mode: nli`, `premise_column`, `hypothesis_column`, and an explicit
  `label_names` order.
- Use the same dataset preprocessing for every backbone in a controlled comparison, except when
  the backbone requires a documented tokenizer-specific preprocessing step such as PhoBERT word
  segmentation.
- Never change label ordering between comparable runs.

### Model comparisons

- Set `model.architecture` and `model.pretrained_model_name` explicitly in comparison configs.
- Change one experimental factor at a time.
- Keep seeds, epochs, batch size, learning rate, loss, and data split fixed for an ablation.
- Do not hardcode hidden size, layer count, or FFN size in YAML. The model reads these from the
  Hugging Face backbone configuration.
- For `transformer_moe`, specify `moe_layers`, `num_experts`, `top_k`, and routing settings.
- For `classifier_moe`, specify `num_experts`, `top_k`, `expert_hidden_dim`, and whether shared
  or residual paths are enabled.
- For `hybrid_moe`, use nested `model.transformer_moe` and `model.classifier_moe` sections so
  encoder and head routing settings cannot be confused. Set `model.hybrid_head_type` to
  `nlimoe` and use `model.nlimoe` to compare the reusable dynamic/top-k head.

### Training and evaluation

- Use validation macro F1 for checkpoint and hyperparameter selection.
- Do not use test metrics to choose configurations.
- Use common seeds for configurations being compared.
- Give every run a unique `--run-id`.
- Preserve the resolved YAML, checkpoint metadata, metrics, and history for every run.
- Enable W&B only when the run is authenticated and should be logged remotely.

### Diagnostics

Keep diagnostics opt-in in normal configs. Use `configs/diagnostics/` for tiny-set overfit
checks, per-class logits/probabilities, routing-by-class analysis, and separate task/auxiliary
loss logging. Diagnostic configs must state clearly which training settings they intentionally
change.

## Naming Rules

Use lowercase kebab-case for run IDs and include the important experimental factors:

```text
<dataset>-<backbone>-<architecture>-<variant>-s<seed>
```

Examples:

```text
vianli-phobert-dense-s42
vianli-xlmr-transformer-moe-last6-e4-top2-s42
vihsd-phobert-classifier-moe-e4-top2-s42
vianli-classifier-moe-overfit-s42
```

The run ID identifies an execution. It does not replace the resolved configuration.

## Group Configurations

Group files are for repeated execution, not for defining model internals. They contain:

- `base_config`;
- a default `seeds` list;
- a `runs` list;
- per-run `id` and dotted-key `overrides`.

Example:

```yaml
base_config: ../baselines/vianli_dense.yaml
seeds: [42, 43, 44]
runs:
  - id: vianli-dense-xlmr
    overrides:
      model.pretrained_model_name: xlm-roberta-base
      dataset.name: uitnlp/ViANLI
      dataset.tokenizer: xlm-roberta-base
```

Run a group with:

```bash
python scripts/run_group.py \
  --group-config configs/groups/vianli_stage1_dense_backbones.yaml \
  --no-smoke-test
```

The runner creates one training run per group entry and seed. It does not select a winner or
modify YAML files.

## Common Commands

Train one run:

```bash
python train.py \
  --config configs/vianli.yaml \
  --no-smoke-test \
  --run-id vianli-phobert-transformer-moe-s42 \
  --set seed=42
```

Override existing YAML keys without editing the file:

```bash
python train.py \
  --config configs/vianli.yaml \
  --no-smoke-test \
  --run-id vianli-xlmr-dense-s42 \
  --set model.architecture=dense_transformer \
  --set model.pretrained_model_name=xlm-roberta-base \
  --set dataset.name=uitnlp/ViANLI \
  --set dataset.tokenizer=xlm-roberta-base \
  --set seed=42
```

Run the diagnostic overfit profile:

```bash
python train.py \
  --config configs/diagnostics/vianli_classifier_moe_tiny_overfit.yaml \
  --no-smoke-test \
  --run-id vianli-classifier-moe-overfit-s42
```

Evaluate a saved checkpoint:

```bash
python evaluate.py \
  --config configs/vianli.yaml \
  --run-id vianli-phobert-transformer-moe-s42
```

Aggregate completed runs:

```bash
python scripts/aggregate_results.py --results-dir results
```

## Review Checklist Before Running

- Is the selected base config correct for the dataset?
- Is the architecture name canonical?
- Is the backbone and tokenizer pair explicit?
- Is the label order fixed and documented?
- Are train, validation, and test splits explicit?
- Is this a single run, a group, a finalist reproduction, or a diagnostic run?
- Are all comparison factors except the intended factor matched?
- Is the run ID unique?
- Is W&B authentication available if logging is enabled?
