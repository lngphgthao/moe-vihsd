# Configuration

`vihsd.yaml` controls dataset columns and splits, tokenizer settings, model architecture selection, model dimensions, routing (`num_experts`, `top_k`), load-balancing loss, training hyperparameters, W&B, and output paths.

The config includes a `model.architecture` field so the same workflow can run multiple variants with the same pipeline:

```yaml
model:
  architecture: current_moe
```

Use `current_moe` for the baseline, `stronger_moe` for the stronger variant, or
`pretrained_backbone` for PhoBERT contextual features followed by an MoE block.
All registered architectures include MoE; the last option does not represent a
standalone PhoBERT classifier.
