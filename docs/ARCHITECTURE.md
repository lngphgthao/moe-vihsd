# Architecture của repository

## Tổng quan

Repository chứa training pipeline dùng chung PyTorch và Hugging Face cho phân loại văn bản
tiếng Việt trên ViANLI và ViHSD. Load dataset, dựng model, training, checkpoint, evaluation
và aggregate result đều được điều khiển bằng config.

The active model families are registered in `src/models/factory.py`:

- `dense_transformer`: dense Hugging Face encoder with a sequence classification layer.
- `transformer_moe`: selected internal Transformer FFNs replaced by token-level MoE layers.
- `classifier_moe`: dense encoder with a top-k MoE classification head.
- `nlimoe`: dense encoder with a sequence-level dynamic-threshold or top-k MoE head.
- `hybrid_moe`: Transformer-MoE encoder followed by a Classifier-MoE head.

The names `dense_phobert` and `phobert_moe` are compatibility aliases. They resolve to the
generic dense and Transformer-MoE implementations respectively.

## Cấu trúc repository

```text
train.py                         Shared training entry point
evaluate.py                      Checkpoint evaluation entry point
configs/                         YAML configurations and experiment overlays
scripts/
  run_group.py                   Multi-run and multi-seed execution
  aggregate_results.py           Result aggregation entry point
  segment_vianli.py              ViANLI word segmentation
  preprocess_vihsd.py            ViHSD word segmentation
src/
  data/loader.py                 Dataset loading, tokenization, and DataLoaders
  models/                        Model registry and model implementations
  models/moe/                    Transformer MoE layers, experts, routers, and dynamic head
  training/losses.py             Task-loss implementations
  training/diagnostics.py        Opt-in per-class and routing diagnostics
  evaluation/metrics.py         Classification metrics and prediction diagnostics
  analysis/aggregate.py          CSV and Markdown result aggregation
  utils/config.py                YAML loading, overrides, seeds, and paths
tests/                           Unit tests, including Classifier-MoE tests
checkpoints/                     Legacy runtime checkpoints (ignored by Git)
results/                         Legacy runtime metrics (ignored by Git)
```

## Luồng dữ liệu

```text
Raw dataset or DatasetDict on disk
   |
   v
Optional VnCoreNLP preprocessing
   |
   v
Dataset loader validates columns and splits
   |
   v
AutoTokenizer: single text or premise/hypothesis pair
   |
   v
DataLoaders for train, validation, and test
   |
   v
Hugging Face backbone
   |
   +--> DenseTransformerClassifier
   +--> TransformerMoEClassifier
   +--> DynamicMoEClassifier
   +--> ClassifierMoEClassifier
   +--> HybridMoEClassifier
   |
   v
Logits and auxiliary routing losses
   |
   v
Validation checkpoint selection and result artifacts
```

## Xử lý dữ liệu

`src/data/loader.py` supports two input modes:

- `single`: tokenizes one configured text column.
- `nli`: passes premise and hypothesis as separate arguments to the Hugging Face tokenizer.

Datasets are loaded from a local `DatasetDict` directory when the configured path exists;
otherwise they are loaded from the Hugging Face Hub. The loader validates configured columns,
filters missing text, tokenizes with fixed maximum length, and creates train, validation, and
test DataLoaders. Train, validation, and test split names must be distinct. If validation is
missing, a deterministic 10% holdout is created from train; a missing test split is an error
rather than silently reusing validation, so model selection and final evaluation remain separate.

Label mapping follows this priority:

1. `dataset.label_names` when explicitly configured, such as ViANLI's
   `entailment`, `contradiction`, and `neutral` order.
2. Hugging Face `ClassLabel` metadata.
3. Sorted unique values from the training label column.

The resolved label names and mapping are exposed through `DatasetBundle` and saved by the
trainer. The optional tiny-overfit diagnostic profile uses stratified train-subset selection
and verifies that every configured class is present.

The loader retains tokenizer-produced `token_type_ids`. All model families, training,
evaluation, and diagnostics forward these optional sentence-segment IDs to the encoder.
Macro F1 averages over the complete configured label list, including absent classes.

PhoBERT ViANLI input is prepared by `scripts/segment_vianli.py`, which applies VnCoreNLP
segmentation independently to premise and hypothesis and saves the result under
`data/vianli_segmented`.

Both segmentation scripts propagate segmentation failures and close the Java service on
exit. Processing must finish successfully before a segmented dataset is saved.

## Model architecture

### Dense Transformer

`DenseTransformerClassifier` loads `AutoModel`, reads `hidden_size` from the Hugging Face
configuration, pools the final hidden states with CLS or masked mean pooling, and applies a
linear classifier. It can optionally freeze the full backbone.

### Transformer MoE

`TransformerMoEClassifier` loads a BERT/RoBERTa-family encoder and validates its
`encoder.layer` stack. Selected layers are replaced by `TransformerMoELayer` instances.
Each replacement preserves the original attention module and LayerNorm while replacing the
feed-forward computation with routed experts.

The layer supports:

- top-k routing through `TopKRouter`;
- dynamic-threshold routing through `DynamicThresholdRouter`;
- GELU, GEGLU, and SwiGLU expert modules;
- optional FFN upcycling for GELU experts;
- optional shared experts;
- fixed or learnable MoE residual scaling;
- auxiliary load-balancing loss.

`model.expert_inner_dropout` mặc định `null`, dùng dropout của Transformer-MoE
như trước. Giá trị `0.0` tắt dropout bên trong expert routed và shared, giữ chỉ số
Linear trong state dict để nạp checkpoint cũ. Tùy chọn cũng đi qua encoder Hybrid.
`configs/ablations/vianli_moe_no_inner_dropout.yaml` đối chứng với YAML finalist
hiện tại bằng cách chỉ thay biến này. Expert và output residual có dropout riêng.

Backbone hidden size, layer count, and original FFN dimensions are read from the loaded model
modules rather than hardcoded for a specific backbone.

Transformers is pinned to 5.5.0 for the tensor-returning encoder-layer contract. Offline
integration tests use small random BERT, RoBERTa, and XLM-R encoders across all five model
families to check batch dimensions, segment IDs, and backward passes.

### Dynamic MoE

`DynamicMoEClassifier` keeps the encoder dense, mean-pools the sequence, and routes the pooled
representation through a set of single-layer square FFN experts (Linear → Dropout → ReLU, weight
matrix d×d). The routing strategy is selected by `model.routing_type`:

- `"dynamic"` (default) — NLIMoEDynamic-style: a learned complexity gate adjusts a per-example
  threshold; experts whose softmax probability exceeds the threshold are active, with a fallback
  to the highest-probability expert. Returns both `dynamic_loss` (entropy regulariser) and
  `balance_loss`.
- `"top_k"` — NLIMoETopK-style: exactly `model.dynamic_top_k` experts (highest softmax
  probability) are selected per example. The complexity gate is absent. Returns only
  `balance_loss` (`dynamic_loss` is zero).

After aggregating active expert outputs the combined representation passes through LayerNorm
before the final linear classifier.

### Classifier-MoE

`ClassifierMoEClassifier` keeps the encoder completely dense, pools its output, and sends the
pooled representation through `ClassifierMoEHead`. The head uses top-k routing over independent
two-layer MLP experts, optionally adds a shared expert or residual connection, and applies a
final linear classifier.

### Hybrid-MoE

`HybridMoEClassifier` composes the existing Transformer-MoE encoder with a selectable pooled
representation head. The default `hybrid_head_type=classifier_moe` uses `ClassifierMoEHead`;
`hybrid_head_type=dynamic_moe` uses the reusable `DynamicMoEHead`. In both cases the backbone
first replaces the configured Transformer FFNs with token-level MoE layers, then pools the
resulting sequence representation before head routing. The model returns one combined
auxiliary loss for the trainer while retaining separate routing records for each
Transformer-MoE layer and the selected head.

The first Classifier-MoE-head experiment is configured in
`configs/architectures/vianli_hybrid_moe_xlmr.yaml`: XLM-R-base, Transformer-MoE in layers
6-11 with six dynamic-threshold experts, followed by four classifier experts with top-2
routing.

The comparable Dynamic-MoE-head configuration is
`configs/architectures/vianli_hybrid_moe_xlmr_dynamic_head.yaml`.

## Trạng thái migration

The target layout in `AGENTS.md` is an approved migration design, not the current
filesystem contract. The current entry points still use `configs/vianli.yaml` and
`configs/vihsd.yaml`, write legacy runtime artifacts under `checkpoints/` and
`results/`, and keep the Transformer-MoE classifier in `src/models/moe/layer.py`.
Moving these responsibilities to `configs/base/`, `configs/experiments/`, `runs/`,
and `src/training/engine.py` requires coordinated caller, notebook, checkpoint, and
artifact migration; it must be handled as a separate refactor. The manifests in
`data/manifests/` are the implemented provenance boundary for data that remains
outside Git.

## Training

`train.py` loads a base YAML configuration, applies repeated `--set section.key=value`
overrides, sets random seeds, prepares data, builds the selected model, and creates an AdamW
optimizer over trainable parameters.

`training.head_learning_rate` và `training.router_learning_rate` mặc định `null`, dùng
`training.learning_rate`. Nhóm router gồm thành phần tên `router` và được ưu tiên
trước head. Head gồm các tiền tố `classifier_moe.`, `nlimoe.`, `classifier.`.
Tham số trainable còn lại dùng nhóm backbone. Mỗi tham số dùng chung chỉ xuất hiện
một lần. Console và `hyperparameters.json` ghi tên nhóm, số phần tử tham số và lr.

For each epoch it records task classification loss, balance loss, dynamic loss, their combined
auxiliary value, accuracy, macro F1, weighted F1, and validation metrics. The best checkpoint
is selected using validation macro F1. W&B logging is optional and is controlled by the
`logging` section.

### Lịch validation cho NLIMoE

Hai cấu hình `configs/architectures/vianli_nlimoe_dynamic.yaml` và
`vianli_nlimoe_topk.yaml` đặt `training.eval_frequency=400`. Đơn vị triển khai
là optimizer step toàn cục, được đếm liên tục qua các epoch. Trainer đánh giá
toàn bộ validation tại mốc và tại bước cuối, sau đó chọn checkpoint theo macro F1.
Cấu hình thiếu frequency hoặc đặt `null` dùng lịch mỗi epoch hiện có.

`training_history.json` được cập nhật tại mỗi lần validation. Mỗi bản ghi có
`epoch`, `global_step`, các train metric tích lũy trong epoch đang chạy và
validation metric. Summary có `best_global_step`. Loss được ghi ở cả dạng thô
`auxiliary_loss` và dạng đã nhân hệ số `weighted_auxiliary_loss`.

NLIMoE đọc riêng `classifier_dropout`, có thể tắt dropout sau LayerNorm trong
khi giữ expert dropout 0.4. Diagnostics chỉ đọc complexity/threshold khi router
cung cấp các trường này, nên Dynamic và Top-K cùng hoàn tất qua entry point train.

Hai YAML cố định revision dữ liệu, encoder và tokenizer. Loader kiểm tra các
split chính thức và số mẫu sau lọc khi `dataset.expected_split_sizes` được đặt.
`model.pretrained_model_revision` được dùng bởi NLIMoE và truyền sang tokenizer
ở cả train và evaluate. Các giả định từ bài được ghi tại D005 trong
`docs/DECISIONS.md`.

Each run stores:

- resolved configuration;
- hyperparameters;
- label mapping;
- training history;
- best checkpoint and metadata;
- run metrics;
- optional per-class logits, probabilities, and routing diagnostics.

The diagnostic profile at `configs/diagnostics/vianli_classifier_moe_tiny_overfit.yaml` enables a
150-sample, 75-epoch stratified overfit check and does not change the model implementation.

## Evaluation

`evaluate.py` restores a checkpoint using its resolved configuration and evaluates the test
DataLoader. It writes classification metrics, per-class F1, classification reports, individual
predictions, confusion-matrix counts, actual class distribution, predicted class distribution,
and a labeled confusion-matrix CSV.

The evaluator writes these prediction artifacts into the run's results directory. Training writes
train and validation metrics only and restores the best-validation checkpoint for diagnostics. Run
`evaluate.py` explicitly only after locking the finalist configuration and seed set; it is the final
test evaluation path.

## Configuration

`src/utils/config.py` loads YAML files and recursively merges an optional `base_config`. CLI
overrides modify the in-memory resolved configuration without editing the source YAML.

Important configuration sections are:

- `dataset`: source, columns, label order, tokenizer, length, and split names;
- `model`: architecture, backbone, pooling, MoE placement, expert, and routing settings;
- `training`: epochs, batch size, optimizer settings, task loss, and sample limits;
- `routing`: auxiliary-loss coefficients;
- `diagnostics`: opt-in tiny-overfit and per-class routing diagnostics;
- `paths`: checkpoint and result roots;
- `logging`: optional W&B project and run settings.

The reserved ViANLI stage workflow is in `configs/workflows/vianli_stages.yaml`. Concrete dense
backbone group execution is configured separately under `configs/groups/`.

The group runner retains the original `base_config` reference and resolves it relative to
the group file before executing each seed. This preserves the selected baseline settings.

Result aggregation reads training metrics together with the separate test prediction
artifact. Missing scores remain empty in CSV and display as a dash in tables. Recorded
zero scores and settings remain zero. Hyperparameters come from saved metadata, with
unknown fields left empty. Nested discovery supports mixed historical artifact formats.

## Thực thi

Install dependencies and authenticate with Hugging Face when required:

```bash
pip install -r requirements.txt
hf auth login
wandb login
```

Prepare segmented ViANLI data for PhoBERT:

```bash
python scripts/segment_vianli.py --output-dir data/vianli_segmented
```

Run one training experiment:

```bash
python train.py \
  --config configs/vianli.yaml \
  --no-smoke-test \
  --run-id vianli-transformer-moe-s42 \
  --set seed=42
```

Run a configured experiment group:

```bash
python scripts/run_group.py \
   --group-config configs/groups/vianli_stage1_dense_backbones.yaml \
  --no-smoke-test
```

Evaluate a saved run:

```bash
python evaluate.py \
  --config configs/vianli.yaml \
  --run-id vianli-transformer-moe-s42
```

Aggregate completed result directories:

```bash
python scripts/aggregate_results.py --results-dir results
```
