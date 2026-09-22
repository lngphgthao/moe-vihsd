"""Preprocess ViANLI dataset with VnCoreNLP word segmentation and cache to disk.

Usage (Kaggle or local, requires Java >= 8):
    python scripts/segment_vianli.py --output-dir data/vianli_segmented

This script:
1. Ensures VnCoreNLP model files exist (downloads automatically if needed).
2. Loads uitnlp/ViANLI from HuggingFace Hub (train=8012, val=1000, test=1000).
3. Applies word segmentation to BOTH the 'premise' and 'hypothesis' fields
   independently (joining compound words with underscores, e.g. học_sinh).
4. Saves the preprocessed DatasetDict to disk so training runs load in <1 second.
5. Fully compatible with local environments and Kaggle.

Note: The label column ('label') contains string values:
  'entailment', 'contradiction', 'neutral'
These are preserved as-is; the dataset loader maps them to int IDs at runtime.
"""

import argparse
import os
import urllib.request
from datasets import load_dataset, DatasetDict
from vncorenlp import VnCoreNLP
from tqdm import tqdm


def ensure_vncorenlp(save_dir: str = "vncorenlp") -> str:
    """Ensure VnCoreNLP jar and word segmenter model files exist."""
    os.makedirs(os.path.join(save_dir, "models", "wordsegmenter"), exist_ok=True)
    base_url = "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/"
    files = [
        ("VnCoreNLP-1.2.jar", os.path.join(save_dir, "VnCoreNLP-1.2.jar")),
        ("models/wordsegmenter/vi-vocab", os.path.join(save_dir, "models", "wordsegmenter", "vi-vocab")),
        ("models/wordsegmenter/wordsegmenter.rdr", os.path.join(save_dir, "models", "wordsegmenter", "wordsegmenter.rdr")),
    ]
    for remote, local in files:
        if not os.path.exists(local):
            print(f"Downloading {remote} to {local}...")
            urllib.request.urlretrieve(base_url + remote, local)
    return os.path.join(save_dir, "VnCoreNLP-1.2.jar")


def segment_text(rdr: VnCoreNLP, text: str) -> str:
    """Segment Vietnamese text into compound words separated by underscores."""
    if not text or not str(text).strip():
        return ""
    try:
        sentences = rdr.tokenize(str(text))
        return " ".join([" ".join(sent) for sent in sentences])
    except Exception:
        # Fallback to original text if tokenizer encounters abnormal input
        return str(text)


def process_split(rdr: VnCoreNLP, split_data, split_name: str):
    """Segment both premise and hypothesis for one dataset split."""
    print(f"Segmenting '{split_name}' split ({len(split_data)} samples)...")

    segmented_premises = []
    segmented_hypotheses = []

    for premise, hypothesis in tqdm(
        zip(split_data["premise"], split_data["hypothesis"]),
        total=len(split_data),
        desc=f"Segmenting {split_name}",
    ):
        segmented_premises.append(segment_text(rdr, premise))
        segmented_hypotheses.append(segment_text(rdr, hypothesis))

    # Replace both text columns with their segmented versions; preserve label and uid
    result = split_data.remove_columns(["premise", "hypothesis"])
    result = result.add_column("premise", segmented_premises)
    result = result.add_column("hypothesis", segmented_hypotheses)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess ViANLI with VnCoreNLP word segmentation"
    )
    parser.add_argument("--dataset-name", default="uitnlp/ViANLI",
                        help="HuggingFace dataset name")
    parser.add_argument("--output-dir", default="data/vianli_segmented",
                        help="Directory to save segmented dataset")
    parser.add_argument("--vncorenlp-dir", default="vncorenlp",
                        help="Directory where VnCoreNLP model files are stored")
    args = parser.parse_args()

    print("=== ViANLI Word Segmentation Preprocessing ===")
    print(f"Dataset: {args.dataset_name}")
    print(f"Output:  {args.output_dir}")

    jar_path = ensure_vncorenlp(args.vncorenlp_dir)
    print("Initializing VnCoreNLP word segmenter...")
    rdr = VnCoreNLP(jar_path, annotators="wseg", max_heap_size="-Xmx2g")

    print(f"Loading raw dataset '{args.dataset_name}' from HuggingFace Hub...")
    raw = load_dataset(args.dataset_name)
    print(f"Splits: { {k: len(v) for k, v in raw.items()} }")

    processed_dict = {}
    for split_name in raw.keys():
        processed_dict[split_name] = process_split(rdr, raw[split_name], split_name)

    processed_dataset = DatasetDict(processed_dict)

    print(f"\nSaving preprocessed dataset to '{args.output_dir}'...")
    os.makedirs(args.output_dir, exist_ok=True)
    processed_dataset.save_to_disk(args.output_dir)

    print("\n--- Verification Samples (train split) ---")
    for i in range(min(2, len(raw["train"]))):
        print(f"\nSample {i + 1}:")
        print(f"  Premise (raw):        {raw['train']['premise'][i]}")
        print(f"  Premise (segmented):  {processed_dataset['train']['premise'][i]}")
        print(f"  Hypothesis (raw):     {raw['train']['hypothesis'][i]}")
        print(f"  Hypothesis (seg):     {processed_dataset['train']['hypothesis'][i]}")
        print(f"  Label:                {processed_dataset['train']['label'][i]}")

    rdr.close()
    print(f"\n[DONE] Successfully segmented and cached ViANLI to '{args.output_dir}'.")
    print("Run training with: python train.py --config configs/vianli.yaml --no-smoke-test ...")


if __name__ == "__main__":
    main()
