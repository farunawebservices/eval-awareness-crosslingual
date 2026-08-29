#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

NEW_TEST = Path("audit/h1_out_of_template_test.jsonl")
EN_ACT = Path("outputs/h1_english_activations.pt")
NEW_ACTS = Path("audit/h1_out_of_template_activations_layer16.pt")

def as_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)

rows = [
    json.loads(line)
    for line in NEW_TEST.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
new = torch.load(NEW_ACTS, map_location="cpu", weights_only=False)
eng = torch.load(EN_ACT, map_location="cpu", weights_only=False)

labels = as_numpy(eng["labels"]).astype(int)
splits = np.asarray(eng["splits"], dtype=object)
train_idx = np.flatnonzero(splits == "train")

probe = LogisticRegression(max_iter=1000, random_state=42)
probe.fit(
    as_numpy(eng["activations"]["layer_16"][train_idx]).astype(np.float32),
    labels[train_idx],
)

probs = probe.predict_proba(as_numpy(new["activations"]).astype(np.float32))[:, 1]
order = np.argsort(probs)

print("=" * 90)
print("LOWEST TEN PREDICTED EVALUATION PROBABILITIES")
print("=" * 90)
for i in order[:10]:
    r = rows[int(i)]
    print(
        f"\nidx={i} core={r['core_id']} label={r['label']} "
        f"format={r['format']} prob_eval={probs[i]:.4f}"
    )
    print("wrapper:", r["new_wrapper"])
    print("task:", r["prompt"][len(r["new_wrapper"]):].strip()[:260].replace("\n", " "))

print("\n" + "=" * 90)
print("HIGHEST TEN PREDICTED EVALUATION PROBABILITIES")
print("=" * 90)
for i in order[-10:][::-1]:
    r = rows[int(i)]
    print(
        f"\nidx={i} core={r['core_id']} label={r['label']} "
        f"format={r['format']} prob_eval={probs[i]:.4f}"
    )
    print("wrapper:", r["new_wrapper"])
    print("task:", r["prompt"][len(r["new_wrapper"]):].strip()[:260].replace("\n", " "))

