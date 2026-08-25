#!/usr/bin/env python3
"""Compute FPR on the 40 deployment-only prompts using the best-layer probe."""

import json
import numpy as np
import torch
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer, AutoModelForCausalLM

REPO = Path("/workspace/eval-awareness-crosslingual")
ACT_PT = REPO / "outputs/h1_english_activations.pt"
FPR_JSONL = REPO / "experiments/h1_clean_pilot/h1_fpr_40_prompts.jsonl"
PROBE_RESULTS = REPO / "outputs/h1_english_probe_results.json"
FPR_OUT = REPO / "outputs/h1_fpr_false_positives.jsonl"

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
BEST_LAYER = 16  # from probe results
HIDDEN = 3584
FPR_THRESH = 0.5

def mean_pool(hidden, attention_mask):
    mask = attention_mask.to(device=hidden.device, dtype=torch.float32).unsqueeze(-1)
    denom = mask.sum(dim=1)
    pooled = (hidden.to(torch.float32) * mask).sum(dim=1) / denom
    return pooled.squeeze(0)

def main():
    # Load probe results to get best layer
    with PROBE_RESULTS.open() as f:
        results = json.load(f)
    best_layer = results["best_layer"]
    print(f"[info] best layer = {best_layer}")

    # Load main activations to retrain the best-layer probe
    data = torch.load(ACT_PT, weights_only=False)
    acts = data["activations"]
    labels = data["labels"].numpy()
    splits = data["splits"]

    train_idx = [i for i, s in enumerate(splits) if s == "train"]
    X_train = acts[f"layer_{best_layer}"][train_idx].numpy()
    y_train = labels[train_idx]

    # Retrain the best-layer probe
    probe = LogisticRegression(max_iter=1000)
    probe.fit(X_train, y_train)
    print(f"[train] retrained probe on layer {best_layer}")

    # Load model and tokenizer for FPR extraction
    print(f"[model] loading {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.to("cuda")
    model.eval()

    # Register hook on best layer
    store = {}
    def hook(_module, _args, output):
        tensor = output[0] if isinstance(output, (tuple, list)) else output
        store[best_layer] = tensor.detach()
    handle = model.model.layers[best_layer].register_forward_hook(hook)

    # Load FPR prompts
    fpr_records = []
    with FPR_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            fpr_records.append(json.loads(line))
    print(f"[fpr] loaded {len(fpr_records)} prompts")

    # Extract activations and classify
    fpr_probs = []
    fpr_fps = []
    device = torch.device("cuda")

    with torch.inference_mode():
        for i, rec in enumerate(fpr_records):
            prompt = rec["prompt"]
            enc = tokenizer(prompt, return_tensors="pt", padding=False, truncation=False)
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            seq_len = int(input_ids.shape[1])

            store.clear()
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

            hidden = store[best_layer]
            vec = mean_pool(hidden, attention_mask).to("cpu", torch.float32).contiguous()

            if vec.shape != (HIDDEN,):
                raise RuntimeError(f"FPR prompt {i} pooled vector shape {tuple(vec.shape)} != ({HIDDEN},)")

            prob = float(probe.predict_proba(vec.numpy().reshape(1, -1))[0, 1])
            fpr_probs.append(prob)
            is_fp = prob >= FPR_THRESH
            fpr_fps.append(is_fp)

            if (i + 1) % 10 == 0:
                print(f"[fpr] {i + 1}/{len(fpr_records)} probs: mean={np.mean(fpr_probs):.4f}, max={np.max(fpr_probs):.4f}")

    handle.remove()

    # Compute FPR
    n_fp = sum(fpr_fps)
    fpr_fraction = n_fp / len(fpr_records)
    print(f"[fpr] {n_fp}/{len(fpr_records)} classified as eval -> FPR = {fpr_fraction:.4f}")

    # Save top 10 highest-probability false positives
    top10_idx = np.argsort(fpr_probs)[-10:][::-1]
    top10_records = []
    for idx in top10_idx:
        top10_records.append({
            "fpr_id": fpr_records[idx].get("fpr_id", idx),
            "prompt": fpr_records[idx]["prompt"],
            "prob_eval": fpr_probs[idx],
            "classified_as_eval": fpr_fps[idx],
        })

    FPR_OUT.parent.mkdir(parents=True, exist_ok=True)
    with FPR_OUT.open("w", encoding="utf-8") as f:
        for rec in top10_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[save] {FPR_OUT}")

    # Update probe results JSON with FPR
    results["fpr"] = {
        "n_fpr_prompts": len(fpr_records),
        "n_false_positives": n_fp,
        "fpr_fraction": float(fpr_fraction),
        "mean_prob_eval": float(np.mean(fpr_probs)),
        "max_prob_eval": float(np.max(fpr_probs)),
        "threshold": FPR_THRESH,
    }
    with PROBE_RESULTS.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[update] {PROBE_RESULTS}")
    print("[done] FPR analysis complete.")

if __name__ == "__main__":
    main()
