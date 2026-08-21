# ViHSD Mixture of Experts experiment

## Colab

Open `main.ipynb` in Google Colab. The notebook mounts Drive, installs `requirements.txt`, runs `train.py`, and runs `evaluate.py`.

## Local commands

```bash
pip install -r requirements.txt
python train.py --config configs/vihsd.yaml
python evaluate.py --config configs/vihsd.yaml
```

Each training run receives a unique UTC timestamp and profile identifier, for example:

```text
20260822T143015Z-full
20260822T143420Z-smoke
```

Checkpoints and results are stored in matching run folders. The latest run is recorded in `checkpoints/latest_run.json` and `results/latest_run.json`, so evaluation without extra options uses the newest run.

To evaluate an older run, pass its identifier:

```bash
python evaluate.py --config configs/vihsd.yaml --run-id 20260822T143015Z-full
```

You can also provide a direct checkpoint path with `--checkpoint`.

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

Hugging Face authentication is loaded from the local `.env` file using the `HF_TOKEN` variable. Keep `.env` private and create it with:

```text
HF_TOKEN=your_hugging_face_token
```

Tokenization uses the configured `dataset.tokenization_num_proc` workers and Hugging Face's cache. Set it to `1` if multiprocessing is unavailable in your environment. Already-tokenized data is reused from cache on later runs.

Set `logging.use_wandb: true` in the YAML and authenticate with W&B before training to enable logging.
