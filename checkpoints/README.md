# Checkpoints

Training checkpoints are written to the Google Drive path configured in `configs/vihsd.yaml`:

`/content/drive/MyDrive/ViHSD-MoE/checkpoints`

The scripts save model weights as architecture-specific `.safetensors` files,
such as `dense_phobert_best.safetensors` and `phobert_moe_best.safetensors`,
with matching metadata files. Older `vihsd_moe_best.safetensors` checkpoints
remain supported by `evaluate.py`.
