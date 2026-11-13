# Ghi chú phân tích nghiên cứu

Use this file for reasoning, not raw metric storage. Raw run metadata and metrics belong in
`experiment_tracker.csv`; concise result interpretation belongs in `RESULTS.md`.

## Kỷ luật bằng chứng

Every note must distinguish:

- **Observation:** directly recorded in the tracker.
- **Interpretation:** a plausible explanation of the observation.
- **Hypothesis:** an explanation that requires further testing.
- **Confirmed finding:** supported by controlled comparisons and repeated seeds.

Do not infer causality from a single seed or from experiments that change multiple factors.

## Điều tra hiện tại: Classifier-MoE collapse

### Quan sát

The tracker contains a single-seed Classifier-MoE ViANLI validation Macro-F1 of 34.00, close to
the three-class chance reference and below the reported dense baselines. The run remains
unverified because its resolved configuration and artifacts are unavailable.

### Kiểm tra trước mắt

- Verify the dataset label mapping and class counts.
- Inspect confusion matrix and per-class F1.
- Compare classification loss with auxiliary MoE loss.
- Inspect average logits and probabilities by true class.
- Inspect routing probabilities and assignments by true class.
- Check expert utilization, routing entropy, and load imbalance.
- Check NaN/Inf values and gradient behavior.
- Verify top-k weight normalization and auxiliary-loss coefficients.
- Confirm that the saved resolved configuration matches the intended run.

### Giả thuyết

The low score may arise from label mapping, classifier representation dynamics, routing
collapse, loss scaling, or an implementation mismatch. No explanation is confirmed yet.

## Điều tra hiện tại: Hybrid-MoE loss và performance anomaly

### Observation

The available `run_metrics.json` for
`vianli-hybrid-moe-xlmr-last6-e6-dynamic-head-e4-top2-s42` records validation Macro-F1 37.51,
training contradiction F1 0.079, task loss 1.01, balance loss 21.41, and dynamic loss 1.72. The
tracker also contains a test score from this historical run. The prior training implementation
evaluated test automatically after each run, so preserve that score as historical and do not use
it for selection. A resolved config and checkpoint are not present, so the run remains unverified.

### Diễn giải

The auxiliary terms dominate the task loss in the recorded metrics, and the training per-class
result indicates a likely failure. These are anomaly signals, not a confirmed cause or finding.

### Kiểm tra tiếp theo

- Recover the resolved config and checkpoint, if they exist outside this workspace.
- Verify the auxiliary-loss aggregation and coefficients against that resolved config.
- Inspect training history, predictions, label mapping, gradients, and routing diagnostics.
- If the implementation and configuration are valid, reproduce one controlled seed before
  spending compute on a multi-seed comparison.

## Câu hỏi mở

- Which dense backbone should be the primary MoE comparison under a predefined tie rule?
- Which architecture families can be compared at matched total and active parameter budgets?
- Does a routing behavior observed on ViANLI transfer to ViHSD?
- Which candidates are stable across seeds 42, 43, and 44?

## Nhật ký cập nhật

| Date       | Experiment IDs    | Type  | Note                                                    |
| ---------- | ----------------- | ----- | ------------------------------------------------------- |
| 2026-10-01 | initial migration | setup | Created structured tracker and interpretation workflow. |
| 2026-10-21 | VIANLI-CLASSIFIER-MOE-XLMR-E4-T2-S42; VIANLI-HYBRID-MOE-XLMR-LAST6-E6-HEAD-E4-T2-S42 | anomaly audit | Marked suspicious single-seed runs as anomalies; test metrics from the historical hybrid run are provenance only. |
