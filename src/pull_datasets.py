"""
Pull verified dataset sources for the cross-lingual eval-awareness project.
Run this first, inside the persistent Jupyter/ipython session on the GPU instance.

Fix log:
- CMMLU: datasets>=4.0.0 removed loading-script support entirely, so
  load_dataset('haonan-li/cmmlu', ...) fails no matter which mirror is used.
  The actual repo (verified via the HF file browser) ships the whole dataset
  as a single archive, cmmlu_v1_0_1.zip, NOT loose CSVs at the repo root.
  Fix: download that zip with hf_hub_download, extract it locally, then read
  the per-subject CSVs from the extracted folder with pandas. This never
  touches the removed builder mechanism.
- ArabicMMLU: config name is 'All' (capital A), not 'all'. This repo is
  parquet-native so load_dataset works fine once the config name is right.
"""
import os
import zipfile
import pandas as pd
from huggingface_hub import hf_hub_download
from datasets import load_dataset

CACHE_DIR = os.path.expanduser("~/.cache/eval_awareness_data")

def pull_cmmlu(subject=None, split="test"):
    """
    CMMLU ships as a single zip archive on the Hub. Download + extract it
    once, then read whichever split's CSVs are inside.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    zip_path = hf_hub_download(
        "haonan-li/cmmlu", "cmmlu_v1_0_1.zip", repo_type="dataset"
    )
    extract_dir = os.path.join(CACHE_DIR, "cmmlu_v1_0_1")
    if not os.path.exists(extract_dir):
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

    # Walk the extracted tree and print it once so we can SEE the real layout
    # instead of assuming it — CMMLU's internal folder structure has varied
    # across versions (some releases nest an extra 'cmmlu' folder).
    found_csvs = []
    for root, _, files in os.walk(extract_dir):
        for fn in files:
            if fn.endswith(".csv") and split in root.lower():
                found_csvs.append(os.path.join(root, fn))

    if not found_csvs:
        print("DEBUG: no CSVs matched split filter, dumping full extracted tree:")
        for root, _, files in os.walk(extract_dir):
            for fn in files:
                print(os.path.join(root, fn))
        raise RuntimeError(
            f"No CSVs found under an implied '{split}' folder inside the "
            f"extracted CMMLU zip. See the DEBUG tree printed above, then "
            f"adjust the `split` match logic to the real folder name."
        )

    if subject is not None:
        found_csvs = [f for f in found_csvs if subject in f]

    frames = []
    for f in found_csvs:
        df = pd.read_csv(f)
        df["subject"] = os.path.basename(f).replace(".csv", "")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

def pull_arabicmmlu():
    return load_dataset("MBZUAI/ArabicMMLU", "All")

def pull_afrimmlu(lang_code="yor"):
    # masakhane IrokoBench AfriMMLU; lang_code e.g. 'yor' (Yoruba), 'hau' (Hausa), 'swa' (Swahili)
    # If this hits the same script error, inspect the repo's file browser on HF
    # directly (huggingface.co/datasets/masakhane/afrimmlu/tree/main) before
    # guessing the fix — do not assume it's the same zip structure as CMMLU.
    return load_dataset("masakhane/afrimmlu", lang_code)

def pull_wildchat(language_filter=None, streaming=True, n=2000):
    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=streaming)
    if language_filter:
        ds = ds.filter(lambda x: x.get("language") == language_filter)
    out = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        out.append(row)
    return out

if __name__ == "__main__":
    print("Verify each pull individually before trusting volume/quality for your target language.")

    cmmlu_df = pull_cmmlu()
    print("CMMLU rows:", len(cmmlu_df))
    print("CMMLU subjects found:", cmmlu_df["subject"].nunique())
    print(cmmlu_df.head(3))

    arabicmmlu = pull_arabicmmlu()
    print("ArabicMMLU splits:", list(arabicmmlu.keys()))
    print("ArabicMMLU rows (first split):", len(arabicmmlu[list(arabicmmlu.keys())[0]]))
