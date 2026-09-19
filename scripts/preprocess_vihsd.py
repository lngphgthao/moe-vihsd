"""Preprocess ViHSD dataset with VnCoreNLP word segmentation and cache to disk.

Usage:
    python scripts/preprocess_vihsd.py --output-dir data/vihsd_segmented

This script:
1. Ensures VnCoreNLP model files exist (downloads automatically if needed).
2. Loads the ViHSD dataset (uitnlp/vihsd).
3. Applies word segmentation (joining compound words with underscores, e.g. học_sinh).
4. Saves the preprocessed DatasetDict to disk so training runs load in <1 second.
5. Fully compatible with local environments and Kaggle.
"""

import argparse
import os
import urllib.request
from datasets import load_dataset
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
    jar_path = os.path.join(save_dir, "VnCoreNLP-1.2.jar")
    return jar_path


def segment_text(rdr: VnCoreNLP, text: str) -> str:
    """Segment Vietnamese text into compound words separated by underscores."""
    if not text or not str(text).strip():
        return ""
    try:
        sentences = rdr.tokenize(str(text))
        return " ".join([" ".join(sent) for sent in sentences])
    except Exception:
        # Fallback to original text if tokenizer encounters abnormal string
        return str(text)


def main():
    parser = argparse.ArgumentParser(description="Preprocess ViHSD with VnCoreNLP word segmentation")
    parser.add_argument("--dataset-name", default="uitnlp/vihsd", help="Hugging Face dataset name or path")
    parser.add_argument("--output-dir", default="data/vihsd_segmented", help="Directory to save segmented dataset")
    parser.add_argument("--vncorenlp-dir", default="vncorenlp", help="Directory where VnCoreNLP model is stored")
    args = parser.parse_args()

    print(f"=== ViHSD Word Segmentation Preprocessing ===")
    print(f"Dataset: {args.dataset_name}")
    print(f"Output:  {args.output_dir}")

    jar_path = ensure_vncorenlp(args.vncorenlp_dir)
    print("Initializing VnCoreNLP word segmenter...")
    rdr = VnCoreNLP(jar_path, annotators="wseg", max_heap_size="-Xmx2g")

    print(f"Loading raw dataset '{args.dataset_name}'...")
    raw = load_dataset(args.dataset_name)

    def process_split(split_data, split_name):
        print(f"Segmenting '{split_name}' split ({len(split_data)} samples)...")
        segmented_texts = []
        for text in tqdm(split_data["free_text"], desc=f"Segmenting {split_name}"):
            segmented_texts.append(segment_text(rdr, text))
        
        # Replace free_text column with segmented text, preserve all other columns
        return split_data.remove_columns(["free_text"]).add_column("free_text", segmented_texts)

    processed_dict = {}
    for split_name in raw.keys():
        processed_dict[split_name] = process_split(raw[split_name], split_name)

    from datasets import DatasetDict
    processed_dataset = DatasetDict(processed_dict)

    print(f"Saving preprocessed dataset to '{args.output_dir}'...")
    os.makedirs(args.output_dir, exist_ok=True)
    processed_dataset.save_to_disk(args.output_dir)

    print("\n--- Verification Sample ---")
    for i in range(min(3, len(raw["train"]))):
        print(f"Sample {i+1}:")
        print(f"  Raw:       {raw['train']['free_text'][i]}")
        print(f"  Segmented: {processed_dataset['train']['free_text'][i]}")
        print(f"  Label:     {processed_dataset['train']['label_id'][i]}")

    rdr.close()
    print(f"\n[DONE] Successfully segmented and cached ViHSD to '{args.output_dir}'.")


if __name__ == "__main__":
    main()
