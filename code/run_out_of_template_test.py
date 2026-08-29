#!/usr/bin/env python3
"""
Out-of-template English wrapper-generalization test.

Source training:
  - original English train split only
  - activation logistic regression at selected layer 16
  - character TF-IDF logistic regression
  - prompt-length logistic regression

Evaluation:
  - 80 newly rendered English held-out test prompts
  - same semantic tasks and labels as original test rows
  - original wrappers replaced by unseen framing templates

This is a template-generalization stress test, not a test of implicit
evaluation awareness.
"""

import json
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = Path(".")
ORIGINAL_DATA = ROOT / "experiments/h1_clean_pilot/h1_70_cores_with_splits.jsonl"
NEW_TEST = ROOT / "audit/h1_out_of_template_test.jsonl"
ENGLISH_ACTIVATIONS = ROOT / "outputs/h1_english_activations.pt"

OUT_ACTIVATIONS = ROOT / "audit/h1_out_of_template_activations_layer16.pt"
OUT_RESULTS = ROOT / "audit/h1_out_of_template_results.json"

MODEL_NAME = (
    "/tmp/hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct/"
    "snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
)
LAYER = 16
HIDDEN_SIZE = 3584

def as_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)

def mean_pool(hidden, attention_mask):
    mask = attention_mask.to(
        device=hidden.device, dtype=torch.float32
    ).unsqueeze(-1)
    denominator = mask.sum(dim=1).clamp_min(1.0)
    return ((hidden.to(torch.float32) * mask).sum(dim=1) / denominator).squeeze(0)

# Load original prompt metadata.
orig_rows = [
    json.loads(line)
    for line in ORIGINAL_DATA.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
new_rows = [
    json.loads(line)
    for line in NEW_TEST.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

if len(new_rows) != 80:
    raise ValueError(f"Expected 80 new test prompts, got {len(new_rows)}")

new_labels = np.array([int(r["label"]) for r in new_rows], dtype=int)
if Counter(new_labels) != Counter({0: 40, 1: 40}):
    raise ValueError(f"Unexpected new label balance: {Counter(new_labels)}")

# Train original text baselines only on original train prompts.
train_rows = [r for r in orig_rows if r["split"] == "train"]
train_prompts = [r["prompt"] for r in train_rows]
train_labels = np.array([
    1 if r["format"].startswith("eval") else 0
    for r in train_rows
], dtype=int)
new_prompts = [r["prompt"] for r in new_rows]

tfidf = TfidfVectorizer(
    ngram_range=(1, 3),
    analyzer="char_wb",
    min_df=2,
)
X_text_train = tfidf.fit_transform(train_prompts)
X_text_new = tfidf.transform(new_prompts)

tfidf_probe = LogisticRegression(
    C=1.0,
    max_iter=2000,
    random_state=0,
)
tfidf_probe.fit(X_text_train, train_labels)
tfidf_probs = tfidf_probe.predict_proba(X_text_new)[:, 1]
tfidf_pred = tfidf_probe.predict(X_text_new)

length_train = np.array([len(x) for x in train_prompts]).reshape(-1, 1)
length_new = np.array([len(x) for x in new_prompts]).reshape(-1, 1)

length_probe = LogisticRegression(
    C=1.0,
    max_iter=2000,
    random_state=0,
)
length_probe.fit(length_train, train_labels)
length_probs = length_probe.predict_proba(length_new)[:, 1]
length_pred = length_probe.predict(length_new)

# Train original activation probe only on original English training activations.
eng = torch.load(ENGLISH_ACTIVATIONS, map_location="cpu", weights_only=False)
eng_labels = as_numpy(eng["labels"]).astype(int)
eng_splits = list(eng["splits"])
eng_train_idx = np.array([
    i for i, split in enumerate(eng_splits) if split == "train"
])

X_activation_train = as_numpy(
    eng["activations"][f"layer_{LAYER}"][eng_train_idx]
).astype(np.float32)

activation_probe = LogisticRegression(
    max_iter=1000,
    random_state=42,
)
activation_probe.fit(X_activation_train, eng_labels[eng_train_idx])

# Extract activations for new prompts.
print("=" * 80)
print("OUT-OF-TEMPLATE ACTIVATION EXTRACTION")
print("=" * 80)
print(f"Model: {MODEL_NAME}")
print(f"Layer: {LAYER}")
print(f"New prompts: {len(new_rows)}")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    local_files_only=True,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    local_files_only=True,
)
model.to("cuda")
model.eval()

captured = {}

def hook(_module, _args, output):
    tensor = output[0] if isinstance(output, (tuple, list)) else output
    captured["hidden"] = tensor.detach()

handle = model.model.layers[LAYER].register_forward_hook(hook)
vectors = []

with torch.inference_mode():
    for i, row in enumerate(new_rows):
        encoded = tokenizer(
            row["prompt"],
            return_tensors="pt",
            padding=False,
            truncation=False,
        )
        input_ids = encoded["input_ids"].to("cuda")
        attention_mask = encoded["attention_mask"].to("cuda")

        captured.clear()
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

        vector = mean_pool(captured["hidden"], attention_mask)
        vector = vector.to("cpu", torch.float32).contiguous()

        if tuple(vector.shape) != (HIDDEN_SIZE,):
            raise RuntimeError(
                f"Prompt {i} has shape {tuple(vector.shape)}, "
                f"expected ({HIDDEN_SIZE},)"
            )

        vectors.append(vector)

        if (i + 1) % 10 == 0:
            print(f"Extracted {i + 1}/{len(new_rows)}")

handle.remove()
new_activations = torch.stack(vectors)

# Save activations so the outcome can be audited without re-running the model.
OUT_ACTIVATIONS.parent.mkdir(parents=True, exist_ok=True)
torch.save({
    "layer": LAYER,
    "activations": new_activations,
    "labels": torch.tensor(new_labels),
    "core_ids": [int(r["core_id"]) for r in new_rows],
    "formats": [r["format"] for r in new_rows],
    "new_wrappers": [r["new_wrapper"] for r in new_rows],
    "source_wrappers": [r["source_wrapper"] for r in new_rows],
    "prompts": new_prompts,
}, OUT_ACTIVATIONS)

X_activation_new = new_activations.numpy()
activation_probs = activation_probe.predict_proba(X_activation_new)[:, 1]
activation_pred = activation_probe.predict(X_activation_new)

def metrics(probs, predictions):
    return {
        "auroc": float(roc_auc_score(new_labels, probs)),
        "balanced_accuracy_at_0_5": float(
            balanced_accuracy_score(new_labels, predictions)
        ),
    }

results = {
    "description": (
        "English out-of-template wrapper-generalization test. "
        "Original English train split used for all probes. "
        "New prompts use held-out English test tasks with original wrappers "
        "replaced by unseen label-consistent wrappers."
    ),
    "model": MODEL_NAME,
    "layer": LAYER,
    "n_test": len(new_rows),
    "label_counts": {
        str(label): int(count)
        for label, count in Counter(new_labels).items()
    },
    "activation_probe": metrics(activation_probs, activation_pred),
    "character_tfidf": metrics(tfidf_probs, tfidf_pred),
    "prompt_length": metrics(length_probs, length_pred),
    "wrapper_counts": {
        "eval": dict(Counter(
            r["new_wrapper"] for r in new_rows if r["label"] == 1
        )),
        "deploy": dict(Counter(
            r["new_wrapper"] for r in new_rows if r["label"] == 0
        )),
    },
    "artifacts": {
        "new_test_prompts": str(NEW_TEST),
        "new_activations": str(OUT_ACTIVATIONS),
    },
}

OUT_RESULTS.write_text(
    json.dumps(results, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("\n" + "=" * 80)
print("OUT-OF-TEMPLATE RESULTS")
print("=" * 80)
for name, result in [
    ("Activation probe", results["activation_probe"]),
    ("Character TF-IDF", results["character_tfidf"]),
    ("Prompt length", results["prompt_length"]),
]:
    print(
        f"{name:20} AUROC={result['auroc']:.4f} "
        f"BAcc={result['balanced_accuracy_at_0_5']:.4f}"
    )

print(f"\nSaved activations: {OUT_ACTIVATIONS}")
print(f"Saved results: {OUT_RESULTS}")
