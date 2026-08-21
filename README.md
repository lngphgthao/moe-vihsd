# ViHSD Mixture of Experts experiment

## Colab

Open `main.ipynb` in Google Colab. The notebook mounts Drive, installs `requirements.txt`, runs `train.py`, and runs `evaluate.py`.

## Local commands

```bash
pip install -r requirements.txt
python train.py --config configs/vihsd.yaml
python evaluate.py --config configs/vihsd.yaml
```

The default checkpoint path is `/content/drive/MyDrive/ViHSD-MoE/checkpoints`, which is intentionally a Google Colab Drive path. Change `paths.checkpoint_dir` for local execution.

Set `logging.use_wandb: true` in the YAML and authenticate with W&B before training to enable logging.
