"""
Pull verified dataset sources for the cross-lingual eval-awareness project.
Run this first, inside the persistent Jupyter/ipython session on the GPU instance.

Fix log:
- CMMLU: datasets>=4.0.0 (July 2025) REMOVED loading-script support entirely.
  Every repo still shipping a cmmlu.py script (haonan-li/cmmlu, lmlmcat/cmmlu,
  and most other mirrors) will raise 'Dataset scripts are no longer supported'
  no matter which mirror you pick. Fix: bypass load_dataset() for CMMLU and
  pull the raw per-subject CSVs directly from the HF repo via hf_hub_download
  or pandas.read_csv against the resolve URL. This sidesteps the script issue
  completely since we never invoke the builder.
- ArabicMMLU: config name is 'All' (capital A), not 'all'. This one DOES use
  a parquet-native repo so load_dataset works fine once the config name is right.
"""
import pandas as pd
from huggingface_hub import list_repo_files, hf_hub_download
from datasets import load_dataset

def pull_cmmlu(subject=None):
    """
    Loads CMMLU by reading raw CSVs directly, avoiding load_dataset() entirely.
    CMMLU's original repo structure has per-subject CSVs under test/ (and
    dev/few-shot examples under dev/). If subject is None, concatenates all
    available test CSVs into one dataframe with a 'subject' column.
    """
    repo_id = "haonan-li/cmmlu"
    files = list_repo_files(repo_id, repo_type="dataset")
    test_csvs = [f for f in files if f.startswith("test/") and f.endswith(".csv")]
    if not test_csvs:
        raise RuntimeError(
            f"No test/*.csv files found in {repo_id}. Repo layout may have "
            f"changed — run list_repo_files('{repo_id}', repo_type='dataset') "
            f"yourself and inspect the actual paths before assuming this structure."
        )
    if subject is not None:
        test_csvs = [f for f in test_csvs if subject in f]
    frames = []
    for f in test_csvs:
        local_path = hf_hub_download(repo_id, f, repo_type="dataset")
        df = pd.read_csv(local_path)
        df["subject"] = f.split("/")[-1].replace(".csv", "")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

def pull_arabicmmlu():
    # Config name is capitalized 'All'. This repo is parquet-native so
    # load_dataset works directly — no script bypass needed here.
    return load_dataset("MBZUAI/ArabicMMLU", "All")

def pull_afrimmlu(lang_code="yor"):
    # masakhane IrokoBench AfriMMLU; lang_code e.g. 'yor' (Yoruba), 'hau' (Hausa), 'swa' (Swahili)
    # Verify this exact dataset id/config on HF before relying on it — IrokoBench
    # subsets are sometimes split across multiple repo names. If load_dataset
    # fails here with the same script error, apply the same raw-file bypass
    # used in pull_cmmlu() above: list_repo_files + hf_hub_download + read_csv.
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
