# ViHSD Mixture of Experts experiment

## Colab

Open `main.ipynb` in Google Colab. The notebook mounts Drive, installs `requirements.txt`, runs `train.py`, and runs `evaluate.py`.

For repeated experiments, edit the `EXPERIMENT_OVERRIDES` cell in the notebook instead of editing and pushing `configs/vihsd.yaml`. It passes only the changed values to the training command, so the versioned YAML remains the shared baseline. Each checkpoint folder stores the exact `resolved_config.yaml`, and evaluation automatically uses it.

## Experiment overrides

An override uses the same path as a value in `configs/vihsd.yaml`, with sections separated by dots. The YAML file is never modified: omitted values keep their defaults.

For example, this Colab configuration compares an 8-expert, top-2 router against the default 4-expert, top-1 model:

```python
EXPERIMENT_OVERRIDES = {
    "training.epochs": 10,
    "training.learning_rate": 0.0001,
    "model.num_experts": 8,
    "model.top_k": 2,
}
RUN_ID = "experts-8-topk-2-lr-1e-4"
SMOKE_TEST = False
```

Set `EXPERIMENT_OVERRIDES = {}` to run the unmodified YAML defaults. Use a unique `RUN_ID` for a readable experiment folder, or set it to `None` for an automatic UTC timestamp. Keep `model.top_k` no greater than `model.num_experts`.

## Local commands

```bash
pip install -r requirements.txt
python train.py --config configs/vihsd.yaml
python evaluate.py --config configs/vihsd.yaml
```

To override values for one run without changing the file, repeat `--set` with a dotted YAML key:

```bash
python train.py --config configs/vihsd.yaml --set training.epochs=10 --set training.learning_rate=0.0001 --set model.num_experts=8
```

Values are parsed as YAML, so use `true`, `false`, `null`, numbers, quoted strings, or YAML lists as appropriate. Only existing keys can be overridden; this catches misspellings before training begins.

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

The default checkpoint path is the local `checkpoints` folder. In Colab, set `CHECKPOINT_DIR` to a Google Drive folder so the same scripts persist runs outside the temporary runtime. `RESULTS_DIR` can override the results location in the same way.

Hugging Face authentication is loaded from the local `.env` file using the `HF_TOKEN` variable. Keep `.env` private and create it with:

```text
HF_TOKEN=your_hugging_face_token
```

Tokenization uses the configured `dataset.tokenization_num_proc` workers and Hugging Face's cache. Set it to `1` if multiprocessing is unavailable in your environment. Already-tokenized data is reused from cache on later runs.

Set `logging.use_wandb: true` in the YAML and authenticate with W&B before training to enable logging.
