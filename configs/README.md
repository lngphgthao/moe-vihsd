# Configuration

`vihsd.yaml` controls dataset columns and splits, tokenizer settings, model architecture selection, model dimensions, routing (`num_experts`, `top_k`), load-balancing loss, training hyperparameters, W&B, and output paths.

The config includes a `model.architecture` field so the same workflow can run multiple variants with the same pipeline:

```yaml
model:
  architecture: phobert_moe
```

Use `phobert_moe` (default) for the true token-level MoE Transformer with PhoBERT,
`current_moe` for the scratch baseline, `stronger_moe` for the stronger multi-expert
scratch variant, or `pretrained_backbone` for PhoBERT followed by a sentence-level MoE head.
