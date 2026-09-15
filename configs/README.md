# Configuration

`vihsd.yaml` controls dataset columns and splits, tokenizer settings, model architecture selection, model dimensions, routing (`num_experts`, `top_k`), load-balancing loss, training hyperparameters, W&B, and output paths.

The config includes a `model.architecture` field so the same workflow can run multiple variants with the same pipeline:

```yaml
model:
  architecture: phobert_moe
```

Use `phobert_moe` (default) for the true token-level MoE Transformer with PhoBERT,
`pretrained_backbone` for PhoBERT followed by a sentence-level MoE head, or
`dense_phobert` for the dense PhoBERT baseline.

With the default settings, `phobert_moe` loads the full pretrained
`vinai/phobert-base` encoder (12 layers, hidden size 768, 12 attention heads),
keeps its embeddings and self-attention, and replaces the feed-forward network in
layers 8-11 (zero-based) with a token-level Sparse MoE block. Each replaced layer
has four 768 -> 3072 -> 768 experts, top-1 routing, the original PhoBERT FFN
weights copied into every expert, and the original residual dropout and LayerNorm.
The other eight encoder layers remain standard PhoBERT layers. Therefore, the
flowchart's attention, router, experts, weighted combination, and residual path
describe each converted layer; they do not describe the entire encoder as one
attention-plus-MoE block.

The architecture experiments are configurable without changing the model code:

```yaml
model:
  learnable_residual_scale: true
  residual_scale_init: 0.1
  expert_init_noise: 0.001
```

Set `expert_type` to `geglu` or `swiglu` for a gated expert and use an
`expert_hidden_size` of 2048 for a roughly parameter-matched comparison. Gated
experts should be run with `upcycle: false`, because the original two-projection
PhoBERT FFN cannot be copied directly into a three-projection gated expert.
The default `expert_type: gelu`, `expert_init_noise: 0.0`, fixed residual scale,
and `upcycle: true` preserve the original experiment.

This is different from `pretrained_backbone`, which runs the complete PhoBERT
encoder unchanged, pools its sequence output to one sentence vector, and sends
that single vector through one sentence-level MoE head.

## Experiment Overlays

Preset experiment configurations are provided in `configs/experiments/`:
- `stage_a_dense.yaml`: Dense PhoBERT baseline (mean pooling, cross-entropy)
- `stage_a_moe.yaml`: Standard PhoBERT-MoE control (4 experts, top-1, layers 8-11, GELU, upcycle)
- `stage_b_shared_expert.yaml`: Shared-plus-routed experts (always-active shared expert + routed experts)
- `stage_d_geglu.yaml`: GEGLU gated experts
- `stage_d_swiglu.yaml`: SwiGLU gated experts

Each overlay specifies `base_config: ../vihsd.yaml` to inherit default hyperparameters and paths. Run them directly without lengthy CLI overrides:

```bash
python train.py --config configs/experiments/stage_a_dense.yaml --run-id dense-s42 --set seed=42
python train.py --config configs/experiments/stage_d_geglu.yaml --run-id geglu-s42 --set seed=42
```

