"""
Pull verified dataset sources for the cross-lingual eval-awareness project.
Run this first, inside the persistent Jupyter/ipython session on the GPU instance.

Fix log:
- CMMLU: haonan-li/cmmlu uses a deprecated loading script (datasets no longer
  supports arbitrary code execution by default). Use the auto-converted parquet
  branch instead via the refs/convert/parquet ref, or a script-free mirror.
- ArabicMMLU: config name is 'All' (capital A), not 'all'.
"""
from datasets import load_dataset

def pull_cmmlu():
    # Script-free parquet mirror (auto-converted by HF datasets-server).
    # If this specific mirror ever goes stale, fall back to loading the
    # refs/convert/parquet branch of haonan-li/cmmlu directly:
    #   load_dataset("haonan-li/cmmlu", "all", revision="refs/convert/parquet")
    try:
        return load_dataset("haonan-li/cmmlu", "all", revision="refs/convert/parquet")
    except Exception as e:
        print(f"Primary CMMLU parquet ref failed ({e}), trying mirror lmlmcat/cmmlu")
        return load_dataset("lmlmcat/cmmlu", "all")

def pull_arabicmmlu():
    # NOTE: config name is capitalized 'All', not 'all' — this was the likely
    # second landmine after the CMMLU fix. Verify the exact split name printed
    # by dataset.keys() before assuming 'test' exists.
    return load_dataset("MBZUAI/ArabicMMLU", "All")

def pull_afrimmlu(lang_code="yor"):
    # masakhane IrokoBench AfriMMLU; lang_code e.g. 'yor' (Yoruba), 'hau' (Hausa), 'swa' (Swahili)
    # Verify this exact dataset id/config on HF before relying on it — IrokoBench
    # subsets are sometimes split across multiple repo names. If this fails,
    # search "masakhane afrimmlu" or "masakhane/irokobench" on HF directly.
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

    cmmlu = pull_cmmlu()
    print("CMMLU splits:", list(cmmlu.keys()))
    print("CMMLU rows (first split):", len(cmmlu[list(cmmlu.keys())[0]]))

    arabicmmlu = pull_arabicmmlu()
    print("ArabicMMLU splits:", list(arabicmmlu.keys()))
    print("ArabicMMLU rows (first split):", len(arabicmmlu[list(arabicmmlu.keys())[0]]))
