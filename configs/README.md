# Configuration

`vihsd.yaml` controls dataset columns and splits, tokenizer settings, model architecture selection, model dimensions, routing (`num_experts`, `top_k`), load-balancing loss, training hyperparameters, W&B, and output paths.

The config includes a `model.architecture` field so the same workflow can run multiple variants with the same pipeline:

```yaml
model:
  architecture: current_moe
```

Use `current_moe` for the baseline architecture or `stronger_moe` for the stronger variant in [models/moe_v2.py](models/moe_v2.py).
