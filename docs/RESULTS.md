# Experiment Results Ledger

This file records reported experiment results and their provenance. A configuration file,
checkpoint, or notebook is not evidence that a run completed successfully.

## Status and Entry Rules

- `reported`: values supplied from an external or local run; verification is pending.
- `verified`: values checked against the resolved config and saved metrics.
- `invalid`: run must not be used for claims; keep it with an explanation.
- `failed`: execution failed; record the failure reason.
- Use validation macro F1 for checkpoint and hyperparameter selection.
- Record test metrics only for frozen final candidates.
- Use a unique experiment ID for each configuration and seed.
- Use decimal points, not decimal commas: `45.82`, not `45,82`.
- Use `NA` when a metric was not recorded. Do not use `-` because it is ambiguous.

## Run Registry

This registry is the first place to add a new run. The metric tables below should reference the
same experiment ID.

| Experiment ID            | Dataset | Backbone                     | Architecture      | Variant  | Seed | Platform | Config path | Status   | Notes       |
| ------------------------ | ------- | ---------------------------- | ----------------- | -------- | ---: | -------- | ----------- | -------- | ----------- |
| VIANLI-DENSE-XLMR-S42    | ViANLI  | xlm-roberta-base             | dense_transformer | baseline |   42 | NA       | NA          | reported | Source ID 1 |
| VIANLI-DENSE-XLMR-S43    | ViANLI  | xlm-roberta-base             | dense_transformer | baseline |   43 | NA       | NA          | reported | Source ID 1 |
| VIANLI-DENSE-XLMR-S44    | ViANLI  | xlm-roberta-base             | dense_transformer | baseline |   44 | NA       | NA          | reported | Source ID 1 |
| VIANLI-DENSE-PHOBERT-S42 | ViANLI  | vinai/phobert-base           | dense_transformer | baseline |   42 | NA       | NA          | reported | Source ID 2 |
| VIANLI-DENSE-PHOBERT-S43 | ViANLI  | vinai/phobert-base           | dense_transformer | baseline |   43 | NA       | NA          | reported | Source ID 2 |
| VIANLI-DENSE-PHOBERT-S44 | ViANLI  | vinai/phobert-base           | dense_transformer | baseline |   44 | NA       | NA          | reported | Source ID 2 |
| VIANLI-DENSE-MBERT-S42   | ViANLI  | bert-base-multilingual-cased | dense_transformer | baseline |   42 | NA       | NA          | reported | Source ID 3 |
| VIANLI-DENSE-MBERT-S43   | ViANLI  | bert-base-multilingual-cased | dense_transformer | baseline |   43 | NA       | NA          | reported | Source ID 3 |
| VIANLI-DENSE-MBERT-S44   | ViANLI  | bert-base-multilingual-cased | dense_transformer | baseline |   44 | NA       | NA          | reported | Source ID 3 |

## ViANLI Results

### Dense Backbones

| Experiment ID            | Backbone                     | Seed | Val Macro F1 | Val Accuracy | Val Weighted F1 | Val Loss |
| ------------------------ | ---------------------------- | ---: | -----------: | -----------: | --------------: | -------: |
| VIANLI-DENSE-XLMR-S42    | xlm-roberta-base             |   42 |        45.82 |        46.30 |           45.82 |     1.43 |
| VIANLI-DENSE-XLMR-S43    | xlm-roberta-base             |   43 |        43.92 |        45.10 |           43.93 |     1.45 |
| VIANLI-DENSE-XLMR-S44    | xlm-roberta-base             |   44 |        41.82 |        44.90 |           41.83 |     1.07 |
| VIANLI-DENSE-PHOBERT-S42 | vinai/phobert-base           |   42 |        45.78 |        46.40 |           45.79 |     1.09 |
| VIANLI-DENSE-PHOBERT-S43 | vinai/phobert-base           |   43 |        46.29 |        46.90 |           46.30 |     1.07 |
| VIANLI-DENSE-PHOBERT-S44 | vinai/phobert-base           |   44 |        45.16 |        46.10 |           45.17 |     1.30 |
| VIANLI-DENSE-MBERT-S42   | bert-base-multilingual-cased |   42 |        45.00 |        44.50 |           44.51 |     1.80 |
| VIANLI-DENSE-MBERT-S43   | bert-base-multilingual-cased |   43 |        42.40 |        44.40 |           42.41 |     1.07 |
| VIANLI-DENSE-MBERT-S44   | bert-base-multilingual-cased |   44 |        40.61 |        44.40 |           40.62 |     1.07 |

### MoE Layer Placement

Common settings: `xlm-roberta-base`, `transformer_moe`, 4 experts, top-1 routing, seed 42.

| Experiment ID          | MoE Layers      | Val Macro F1 | Val Accuracy | Val Weighted F1 | Val Loss | Platform | Notes                                               |
| ---------------------- | --------------- | -----------: | -----------: | --------------: | -------: | -------- | --------------------------------------------------- |
| VIANLI-PLACE-LAST4-S42 | [8,9,10,11]     |        39.96 |        39.60 |           39.97 |     1.06 | Colab    | Source ID 4; reported note says last 6 was highest. |
| VIANLI-PLACE-LAST6-S42 | [6,7,8,9,10,11] |        41.27 |        42.50 |           41.28 |     1.08 | Kaggle   | Source ID 5                                         |
| VIANLI-PLACE-ALT-S42   | [1,3,5,7,9,11]  |        41.17 |        41.10 |           41.18 |     1.10 | Kaggle   | Source ID 6                                         |
| VIANLI-PLACE-ALL-S42   | all             |        41.09 |        42.30 |           41.09 |     1.06 | Colab    | Source ID 7                                         |

### Expert Count and Routing

Common settings: `xlm-roberta-base`, `transformer_moe`, layers `[6,7,8,9,10,11]`, seed 42.

| Experiment ID               | Experts | Top-k   | Val Macro F1 | Val Accuracy | Val Weighted F1 | Val Loss | Notes              |
| --------------------------- | ------: | ------- | -----------: | -----------: | --------------: | -------: | ------------------ |
| VIANLI-ROUTE-E2-T1-S42      |       2 | 1       |        41.60 |        42.00 |           41.61 |     1.09 |                    |
| VIANLI-ROUTE-E2-T2-S42      |       2 | 2       |        43.98 |        43.90 |           43.99 |     1.25 |                    |
| VIANLI-ROUTE-E2-DYNAMIC-S42 |       2 | dynamic |        43.70 |        43.60 |           43.70 |     1.18 |                    |
| VIANLI-ROUTE-E4-T1-S42      |       4 | 1       |        41.27 |        42.50 |           41.28 |     1.08 | Reuses Source ID 5 |
| VIANLI-ROUTE-E4-T2-S42      |       4 | 2       |        43.21 |        43.20 |           43.21 |     1.21 |                    |
| VIANLI-ROUTE-E4-T4-S42      |       4 | 4       |        40.36 |        40.90 |           40.38 |     1.12 |                    |
| VIANLI-ROUTE-E4-DYNAMIC-S42 |       4 | dynamic |        44.48 |        45.80 |           44.49 |     1.19 |                    |
| VIANLI-ROUTE-E6-T1-S42      |       6 | 1       |        37.98 |        43.00 |           37.99 |     1.09 |                    |
| VIANLI-ROUTE-E6-T2-S42      |       6 | 2       |        43.04 |        43.40 |           43.05 |     1.05 |                    |
| VIANLI-ROUTE-E6-T4-S42      |       6 | 4       |        44.20 |        44.40 |           44.20 |     1.35 |                    |
| VIANLI-ROUTE-E6-DYNAMIC-S42 |       6 | dynamic |        45.41 |        45.20 |           45.41 |     1.18 |                    |
| VIANLI-ROUTE-E8-T1-S42      |       8 | 1       |        36.84 |        43.20 |           36.86 |     1.06 |                    |
| VIANLI-ROUTE-E8-T2-S42      |       8 | 2       |        44.44 |        44.20 |           44.45 |     1.05 |                    |
| VIANLI-ROUTE-E8-T4-S42      |       8 | 4       |        43.72 |        45.00 |           43.73 |     1.08 |                    |
| VIANLI-ROUTE-E8-DYNAMIC-S42 |       8 | dynamic |        41.61 |        42.60 |           41.62 |     1.17 |                    |

### Shared Expert

Common settings: `xlm-roberta-base`, `transformer_moe`, layers `[6,7,8,9,10,11]`, seed 42.

| Experiment ID                | Shared Expert | Experts | Top-k   | Val Macro F1 | Val Accuracy | Val Weighted F1 | Val Loss |
| ---------------------------- | ------------- | ------: | ------- | -----------: | -----------: | --------------: | -------: |
| VIANLI-SHARED-E4-DYNAMIC-S42 | true          |       4 | dynamic |        38.27 |        45.20 |           38.29 |     1.04 |
| VIANLI-SHARED-E8-T2-S42      | true          |       8 | 2       |        37.71 |        41.70 |           37.72 |     1.07 |

### Cross-Backbone Architecture Check

Common seed: 42.

| Experiment ID                             | Architecture    | Backbone           | MoE Layers      | Experts | Top-k   | Val Macro F1 | Val Accuracy | Val Weighted F1 | Val Loss |
| ----------------------------------------- | --------------- | ------------------ | --------------- | ------: | ------- | -----------: | -----------: | --------------: | -------: |
| VIANLI-CROSS-XLMR-ALL-E4-T2-S42           | transformer_moe | xlm-roberta-base   | all             |       4 | 2       |        45.27 |        45.50 |           45.27 |     1.14 |
| VIANLI-CROSS-PHOBERT-LAST6-E6-T2-S42      | transformer_moe | vinai/phobert-base | [6,7,8,9,10,11] |       6 | 2       |        44.69 |        45.00 |           44.70 |     1.34 |
| VIANLI-CROSS-PHOBERT-LAST6-E4-T2-S42      | transformer_moe | vinai/phobert-base | [6,7,8,9,10,11] |       4 | 2       |        45.46 |        46.00 |           45.47 |     1.08 |
| VIANLI-CROSS-PHOBERT-ALL-E6-T2-S42        | transformer_moe | vinai/phobert-base | all             |       6 | 2       |        43.34 |        43.90 |           43.35 |     1.45 |
| VIANLI-CROSS-PHOBERT-LAST6-E6-DYNAMIC-S42 | transformer_moe | vinai/phobert-base | [6,7,8,9,10,11] |       6 | dynamic |        45.05 |        45.20 |           45.06 |     1.62 |

### Classifier-MoE

| Experiment ID                        | Architecture   | Backbone         | Experts | Top-k | Val Macro F1 | Val Accuracy | Val Weighted F1 | Val Loss |
| ------------------------------------ | -------------- | ---------------- | ------: | ----- | -----------: | -----------: | --------------: | -------: |
| VIANLI-CLASSIFIER-MOE-XLMR-E4-T2-S42 | classifier_moe | xlm-roberta-base |       4 | 2     |        34.00 |        42.80 |           34.02 |     1.08 |

## ViHSD Results

### Baseline and Basic Hyperparameters

Default when not otherwise stated: 4 experts, top-1 routing, cross-entropy, MoE layers `[8,9,10,11]`, load-balance coefficient `0.01`.

| Experiment ID                | Model   | Freeze | Pooling | Learning Rate | Best Epoch | Val Macro F1 | Test Macro F1 | Val Accuracy | Val Weighted F1 |
| ---------------------------- | ------- | ------ | ------- | ------------: | ---------: | -----------: | ------------: | -----------: | --------------: |
| VIHSD-DENSE-MEAN             | PhoBERT | false  | mean    |       0.00002 |         NA |           NA |            NA |           NA |              NA |
| VIHSD-DENSE-CLS              | PhoBERT | false  | CLS     |       0.00002 |          4 |        67.03 |         65.16 |        85.93 |           86.16 |
| VIHSD-TRANSFORMER-MOE-CLS    | PhoBERT | false  | CLS     |       0.00002 |         NA |           NA |            NA |           NA |              NA |
| VIHSD-TRANSFORMER-MOE-FROZEN | PhoBERT | true   | CLS     |       0.00002 |         NA |           NA |            NA |           NA |              NA |
| VIHSD-TRANSFORMER-MOE-MEAN   | PhoBERT | false  | mean    |       0.00002 |         NA |           NA |            NA |           NA |              NA |
| VIHSD-TRANSFORMER-MOE-LR1E5  | PhoBERT | false  | CLS     |       0.00001 |         NA |           NA |            NA |           NA |              NA |
| VIHSD-TRANSFORMER-MOE-LR3E5  | PhoBERT | false  | CLS     |       0.00003 |         NA |           NA |            NA |           NA |              NA |
| VIHSD-TRANSFORMER-MOE-LR5E6  | PhoBERT | false  | CLS     |      0.000005 |         NA |           NA |            NA |           NA |              NA |

### Expert Count and Routing

Common settings: PhoBERT, encoder unfrozen, CLS pooling, learning rate `0.00002`, shared expert false,
MoE layers `[8,9,10,11]`.

| Experiment ID          | Experts | Top-k | Loss | Residual Scale | Best Epoch | Val Macro F1 | Test Macro F1 | Val Accuracy | Val Weighted F1 |
| ---------------------- | ------: | ----- | ---- | -------------- | ---------: | -----------: | ------------: | -----------: | --------------: |
| VIHSD-ROUTE-E2-T1      |       2 | 1     | CE   | NA             |         NA |           NA |            NA |           NA |              NA |
| VIHSD-ROUTE-E4-T2      |       4 | 2     | CE   | NA             |          5 |        67.20 |         63.38 |        87.80 |           86.84 |
| VIHSD-ROUTE-E8-T1      |       8 | 1     | CE   | NA             |         NA |           NA |            NA |           NA |              NA |
| VIHSD-ROUTE-E8-T2-RES1 |       8 | 2     | CE   | 1.0            |          4 |        68.22 |         64.39 |        86.71 |           86.82 |
| VIHSD-ROUTE-E6-T2-WCE  |       6 | 2     | WCE  | NA             |          5 |        66.50 |         64.47 |        87.84 |           86.77 |

### Shared Expert and Residual Scale

Common settings: PhoBERT, encoder unfrozen, CLS pooling, learning rate `0.00002`, CE loss,
MoE layers `[8,9,10,11]`.

| Experiment ID            | Shared Expert | Experts | Top-k | Residual Scale Init | Best Epoch | Val Macro F1 | Test Macro F1 | Val Accuracy | Val Weighted F1 |
| ------------------------ | ------------- | ------: | ----- | ------------------: | ---------: | -----------: | ------------: | -----------: | --------------: |
| VIHSD-SHARED-E4-T1       | true          |       4 | 1     |                  NA |         NA |           NA |            NA |           NA |              NA |
| VIHSD-SHARED-ONLY        | true          |       0 | 1     |                  NA |         NA |           NA |            NA |           NA |              NA |
| VIHSD-SHARED-E4-T2       | true          |       4 | 2     |                  NA |          5 |        69.41 |         65.03 |        87.61 |           87.31 |
| VIHSD-SHARED-E8-T2-A     | true          |       8 | 2     |                  NA |         NA |           NA |            NA |           NA |              NA |
| VIHSD-SHARED-E8-T2-B     | true          |       8 | 2     |                  NA |         NA |           NA |            NA |           NA |              NA |
| VIHSD-SHARED-E4-T1-RES01 | true          |       4 | 1     |                 0.1 |         NA |           NA |            NA |           NA |              NA |
| VIHSD-SHARED-E4-T1-RES10 | true          |       4 | 1     |                 1.0 |          4 |        64.95 |         64.18 |        87.01 |           86.24 |

### Layer Placement and Other Ablations

Common settings: PhoBERT, encoder unfrozen, CLS pooling, learning rate `0.00002`, residual scale init `1.0`.

| Experiment ID            | MoE Layers  | Experts | Top-k | Shared Expert | Load Balance | Expert Init Noise | Best Epoch | Val Macro F1 | Test Macro F1 | Val Accuracy | Val Weighted F1 |
| ------------------------ | ----------- | ------: | ----: | ------------- | -----------: | ----------------: | ---------: | -----------: | ------------: | -----------: | --------------: |
| VIHSD-PLACE-LAST4-T2     | [8,9,10,11] |       4 |     2 | false         |         0.01 |                NA |          4 |        67.72 |         64.69 |        87.54 |           87.13 |
| VIHSD-PLACE-LAST4-T2-LB0 | [8,9,10,11] |       4 |     2 | false         |          0.0 |                NA |          5 |        67.51 |         64.05 |        87.61 |           86.94 |
| VIHSD-PLACE-LAYERS10-11  | [10,11]     |       4 |     2 | false         |         0.01 |                NA |         NA |           NA |            NA |           NA |              NA |
| VIHSD-PLACE-LAYER11      | [11]        |       4 |     2 | false         |         0.01 |                NA |          3 |        68.09 |         64.69 |        86.83 |           86.77 |
| VIHSD-PLACE-ALL          | all         |       4 |     2 | false         |         0.01 |                NA |          3 |        69.50 |         65.39 |        87.35 |           87.14 |
| VIHSD-PLACE-ALL-SHARED   | all         |       4 |     2 | true          |         0.01 |                NA |          5 |        48.60 |         47.30 |        85.07 |           81.21 |
| VIHSD-PLACE-ALL-E8-T2    | all         |       8 |     2 | false         |           NA |                NA |          3 |        65.36 |         62.51 |        87.09 |           86.27 |
| VIHSD-PLACE-ALL-NOISE01  | all         |       4 |     2 | false         |         0.01 |              0.01 |          5 |        66.79 |         62.76 |        86.90 |           86.54 |

## Aggregate Summary Template

Add aggregate rows only after the individual seed runs are recorded.

| Experiment Family | Seeds        | Mean Val Macro F1 | Std Val Macro F1 | Mean Test Macro F1 | Std Test Macro F1 | Selection Rule        | Status     |
| ----------------- | ------------ | ----------------: | ---------------: | -----------------: | ----------------: | --------------------- | ---------- |
| `<family-id>`     | `<42,43,44>` |                NA |               NA |                 NA |                NA | `validation_macro_f1` | `reported` |
