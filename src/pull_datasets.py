"""
Pull verified dataset sources for the cross-lingual eval-awareness project.
Run this first, inside the persistent Jupyter/ipython session on the GPU instance.
"""
from datasets import load_dataset

def pull_cmmlu():
    return load_dataset("haonan-li/cmmlu", "all")

def pull_arabicmmlu():
    return load_dataset("MBZUAI/ArabicMMLU")

def pull_afrimmlu(lang_code="yor"):
    # masakhane IrokoBench AfriMMLU; lang_code e.g. 'yor' (Yoruba), 'hau' (Hausa), 'swa' (Swahili)
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
    print("CMMLU rows:", len(pull_cmmlu()["test"]))
    print("ArabicMMLU rows:", len(pull_arabicmmlu()["test"]))
