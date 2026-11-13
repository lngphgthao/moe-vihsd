# Kế hoạch experiment nghiên cứu

## Luận văn

**Vietnamese:** Ap dung kien truc Mixture of Experts cho bai toan phan lop

**English:** Applying Mixture of Experts Architectures to Text Classification

## Mục tiêu

Study how MoE architecture, expert count, routing, placement, backbone, training settings,
parameter efficiency, computational cost, and expert behavior affect Vietnamese text
classification on ViANLI and ViHSD.

The objective is not simply to maximize one score. Negative results, instability, efficiency,
and routing behavior are valid research outcomes.

## Biến nghiên cứu

- Dataset: ViANLI, ViHSD
- Backbone: XLM-R-base, PhoBERT-base, mBERT; optional XLM-R-large
- Architecture: dense, Transformer-MoE, Classifier-MoE, Hybrid-MoE
- MoE placement: none, last 4, last 6, alternating, all, or explicit layer lists
- Experts: 2, 4, 8; optional 5
- Routing: top-1, top-2, top-4, dynamic
- Shared expert: enabled or disabled
- Dynamic routing: rho, gamma, alpha, beta when implemented
- Training: learning rate, weight decay, dropout, epochs, random seed

## Metric

Primary development metric:

- Validation macro F1

Secondary development metrics:

- Validation accuracy
- Validation weighted F1
- Per-class F1 and recall
- Validation loss

Efficiency metrics:

- Total parameters
- Trainable parameters
- Active parameters per token
- FLOPs, when available
- Peak GPU memory
- Training time
- Inference time

MoE behavior metrics:

- Expert utilization
- Tokens per expert
- Routing entropy
- Load imbalance
- Active experts per token/example
- Auxiliary losses

## Policy cho test

Test results must not be used for architecture, hyperparameter, checkpoint, or finalist
selection. Evaluate the test split only after the configuration and seed set are locked for
final confirmation.

## Giai đoạn

### Stage 0: Kiểm tra protocol và implementation

Confirm dataset columns, label order, tokenizer behavior, split handling, model shapes,
losses, routing diagnostics, and artifact persistence. Use the diagnostic profiles for tiny-set
overfit checks when investigating a failure.

### Stage 1: Dense backbone baseline

Compare dense `xlm-roberta-base`, `vinai/phobert-base`, and
`bert-base-multilingual-cased` using matched seeds and training settings.

Configuration entry point: `configs/groups/vianli_stage1_dense_backbones.yaml`.

Gate: select development candidates using validation macro F1, variance, efficiency, and
preprocessing considerations. Do not use test performance.

### Stage 2: So sánh architecture family

Compare dense, Transformer-MoE, Classifier-MoE, and Hybrid-MoE where implemented. Keep the
backbone, data processing, loss, budget, and seed fixed.

Gate: identify stable candidates and record whether the result is a controlled comparison.

### Stage 3: Vị trí Transformer-MoE

Compare none, last 4, last 6, alternating, all, and explicit layer lists. Change placement
only; keep expert count and routing fixed.

### Stage 4: Số expert và routing

Compare expert counts 2, 4, and 8. Then compare top-1, top-2, top-4, and dynamic routing
under a declared parameter and compute budget.

### Stage 5: Shared path và hybrid path

Compare shared expert enabled/disabled and hybrid designs. Record active and total parameters
so added capacity is not confused with an architecture effect.

### Stage 6: Training và regularization

After architecture screening, compare learning rate, weight decay, dropout, epochs, and
auxiliary-loss coefficients one factor at a time.

### Stage 7: Xác nhận với nhiều seed

Repeat promising configurations with seeds 42, 43, and 44 at minimum. Use 3-5 seeds for final
candidates when resources allow. Report mean and standard deviation.

## Quy tắc thực thi

- A planned row is not evidence of execution.
- One row in `experiment_tracker.csv` represents one configuration and seed.
- Use a unique experiment ID for each run.
- Use controlled comparisons whenever possible.
- Record anomalies instead of deleting them.
- Update the tracker before writing interpretation.
- Preserve the resolved configuration and artifact paths.
