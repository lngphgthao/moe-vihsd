# ViHSD Mixture of Experts experiment

## Colab

Open `main.ipynb` in Google Colab. The notebook mounts Drive, installs `requirements.txt`, runs `train.py`, and runs `evaluate.py`.

## Local commands

```bash
pip install -r requirements.txt
python train.py --config configs/vihsd.yaml
python evaluate.py --config configs/vihsd.yaml
```

The YAML defaults to full training. Run a quick smoke test without editing the YAML:

```bash
python train.py --config configs/vihsd.yaml --smoke-test
```

To explicitly force the full profile, use:

```bash
python train.py --config configs/vihsd.yaml --no-smoke-test
```

The smoke profile is defined under `training.smoke` in the YAML. The normal profile uses the top-level `training.epochs` and `training.max_train_samples` values.

The default checkpoint path is `/content/drive/MyDrive/ViHSD-MoE/checkpoints`, which is intentionally a Google Colab Drive path. Change `paths.checkpoint_dir` for local execution.

Set `logging.use_wandb: true` in the YAML and authenticate with W&B before training to enable logging.
