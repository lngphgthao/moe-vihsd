# Template ghi nhận experiment

Copy template này cho experiment mới trước khi chạy. Template không phải bằng chứng
experiment đã chạy.

## Metadata

| Field              | Value                                           |
| ------------------ | ----------------------------------------------- |
| Experiment ID      | `<dataset>-<architecture>-<variant>-s<seed>`    |
| Status             | `planned`                                       |
| Verification       | `unverified`                                    |
| Dataset            | `<ViANLI/ViHSD>`                                |
| Backbone           | `<model name>`                                  |
| Architecture       | `<dense/transformer_moe/classifier_moe/hybrid>` |
| Variant            | `<short description>`                           |
| Config path        | `<configs/...yaml>`                             |
| Seed               | `<42>`                                          |
| Platform           | `<local/Colab/Kaggle>`                          |
| Git commit         | `<commit>`                                      |
| Parent experiment  | `<ID or NA>`                                    |
| Research question  | `<what factor is being tested?>`                |
| Controlled factors | `<what remains fixed?>`                         |

## Config

| Parameter             | Value                            |
| --------------------- | -------------------------------- |
| MoE layers            | `<none/list/all>`                |
| Experts               | `<2/4/8/NA>`                     |
| Routing               | `<top-1/top-2/top-4/dynamic/NA>` |
| Shared expert         | `<true/false/NA>`                |
| Learning rate         | `<value>`                        |
| Weight decay          | `<value>`                        |
| Dropout               | `<value>`                        |
| Epochs                | `<value>`                        |
| Batch size            | `<value>`                        |
| Max length            | `<value>`                        |
| Loss                  | `<value>`                        |
| Auxiliary coefficient | `<value>`                        |

## Metric

| Split      | Loss | Accuracy | Macro F1 | Weighted F1 | Per-class F1 path |
| ---------- | ---: | -------: | -------: | ----------: | ----------------- |
| train      |   NA |       NA |       NA |          NA | `<path or NA>`    |
| validation |   NA |       NA |       NA |          NA | `<path or NA>`    |
| test       |   NA |       NA |       NA |          NA | `<path or NA>`    |

## Efficiency và MoE diagnostics

| Metric                  | Value | Artifact path  |
| ----------------------- | ----: | -------------- |
| Total parameters        |    NA | `<path or NA>` |
| Trainable parameters    |    NA | `<path or NA>` |
| Active parameters/token |    NA | `<path or NA>` |
| FLOPs                   |    NA | `<path or NA>` |
| Peak GPU memory         |    NA | `<path or NA>` |
| Training time           |    NA | `<path or NA>` |
| Inference time          |    NA | `<path or NA>` |
| Expert utilization      |    NA | `<path or NA>` |
| Routing entropy         |    NA | `<path or NA>` |
| Load imbalance          |    NA | `<path or NA>` |
| Auxiliary loss          |    NA | `<path or NA>` |

## Diễn giải

- **Observation:**
- **Interpretation:**
- **Hypothesis:**
- **Confirmed finding:**
- **Anomaly or failure notes:**
- **Next controlled comparison:**
