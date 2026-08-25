#!/usr/bin/env python3
"""H2 Chinese transfer analysis: English probe zero-shot vs. native Chinese probes."""

import json
import numpy as np
import torch
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

REPO = Path("/workspace/eval-awareness-crosslingual")
ENG_ACT_PT = REPO / "outputs/h1_english_activations.pt"
CHN_ACT_PT = REPO / "outputs/h2_chinese_activations_4layers.pt"
ENG_PROBE_RESULTS = REPO / "outputs/h1_english_probe_results.json"
OUT_JSON = REPO / "outputs/h2_chinese_transfer_results.json"

LAYERS = [12, 16, 20, 24]
HIDDEN = 3584
BOOTSTRAP_N = 1000

def bootstrap_ci(y_true, y_prob, metric_fn, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n):
        idx = rng.choice(n, n, replace=True)
        yt, yp = y_true[idx], y_prob[idx]
        try:
            scores.append(metric_fn(yt, yp))
        except:
            scores.append(np.nan)
    scores = np.array(scores)
    lo, hi = np.nanpercentile(scores, 2.5), np.nanpercentile(scores, 97.5)
    return float(lo), float(hi)

def main():
    print("[load] English activations")
    eng_data = torch.load(ENG_ACT_PT, weights_only=False)
    eng_acts = eng_data["activations"]
    eng_labels = eng_data["labels"].numpy()
    eng_splits = eng_data["splits"]

    print("[load] Chinese activations")
    chn_data = torch.load(CHN_ACT_PT, weights_only=False)
    chn_acts = chn_data["activations"]
    chn_labels = chn_data["labels"].numpy()
    chn_splits = chn_data["splits"]

    # English train indices (for retraining the English probe)
    eng_train_idx = [i for i, s in enumerate(eng_splits) if s == "train"]
    eng_val_idx = [i for i, s in enumerate(eng_splits) if s == "val"]
    eng_test_idx = [i for i, s in enumerate(eng_splits) if s == "test"]

    # Chinese indices
    chn_train_idx = [i for i, s in enumerate(chn_splits) if s == "train"]
    chn_val_idx = [i for i, s in enumerate(chn_splits) if s == "val"]
    chn_test_idx = [i for i, s in enumerate(chn_splits) if s == "test"]

    print(f"[splits] English train={len(eng_train_idx)}, val={len(eng_val_idx)}, test={len(eng_test_idx)}")
    print(f"[splits] Chinese train={len(chn_train_idx)}, val={len(chn_val_idx)}, test={len(chn_test_idx)}")

    results = {
        "layers": LAYERS,
        "english_probe_transfer": {},  # English probe applied to Chinese at each layer
        "native_chinese_probes": {},   # Native Chinese probes at each layer
        "best_transfer_layer": None,
        "best_native_layer": None,
    }

    # Retrain English probe at layer 16 (from H1 results)
    print("\n[english] retraining probe at layer 16")
    eng_layer = 16
    X_eng_train = eng_acts[f"layer_{eng_layer}"][eng_train_idx].numpy()
    y_eng_train = eng_labels[eng_train_idx]
    
    eng_probe = LogisticRegression(max_iter=1000)
    eng_probe.fit(X_eng_train, y_eng_train)
    print(f"  English layer {eng_layer} probe trained on {len(eng_train_idx)} samples")

    # Apply English probe to Chinese at all four layers (zero-shot transfer)
    print("\n[transfer] English probe → Chinese (zero-shot)")
    best_transfer_auc = -1
    best_transfer_layer = None

    for layer in LAYERS:
        X_chn_test = chn_acts[f"layer_{layer}"][chn_test_idx].numpy()
        y_chn_test = chn_labels[chn_test_idx]
        
        # English probe on Chinese test
        p_chn = eng_probe.predict_proba(X_chn_test)[:, 1]
        auc = roc_auc_score(y_chn_test, p_chn)
        bal = balanced_accuracy_score(y_chn_test, (p_chn >= 0.5).astype(int))
        
        results["english_probe_transfer"][f"layer_{layer}"] = {
            "test_auroc": float(auc),
            "test_balanced_acc": float(bal),
        }
        print(f"  layer {layer}: test_auroc={auc:.4f}, test_bal={bal:.4f}")
        
        if auc > best_transfer_auc:
            best_transfer_auc = auc
            best_transfer_layer = layer

    results["best_transfer_layer"] = int(best_transfer_layer)
    results["best_transfer_auroc"] = float(best_transfer_auc)
    print(f"\n[transfer] Best: layer {best_transfer_layer} with AUROC={best_transfer_auc:.4f}")

    # Train native Chinese probes at all four layers
    print("\n[native] training Chinese probes at all layers")
    best_native_auc = -1
    best_native_layer = None

    for layer in LAYERS:
        X_chn_train = chn_acts[f"layer_{layer}"][chn_train_idx].numpy()
        y_chn_train = chn_labels[chn_train_idx]
        X_chn_val = chn_acts[f"layer_{layer}"][chn_val_idx].numpy()
        y_chn_val = chn_labels[chn_val_idx]
        X_chn_test = chn_acts[f"layer_{layer}"][chn_test_idx].numpy()
        y_chn_test = chn_labels[chn_test_idx]

        probe = LogisticRegression(max_iter=1000)
        probe.fit(X_chn_train, y_chn_train)
        
        p_val = probe.predict_proba(X_chn_val)[:, 1]
        p_test = probe.predict_proba(X_chn_test)[:, 1]
        
        auc_val = roc_auc_score(y_chn_val, p_val)
        auc_test = roc_auc_score(y_chn_test, p_test)
        bal_test = balanced_accuracy_score(y_chn_test, (p_test >= 0.5).astype(int))
        
        lo, hi = bootstrap_ci(y_chn_test, p_test, lambda yt, yp: roc_auc_score(yt, yp), n=BOOTSTRAP_N)
        
        results["native_chinese_probes"][f"layer_{layer}"] = {
            "val_auroc": float(auc_val),
            "test_auroc": float(auc_test),
            "test_balanced_acc": float(bal_test),
            "test_auroc_95ci": [lo, hi],
        }
        print(f"  layer {layer}: val_auroc={auc_val:.4f}, test_auroc={auc_test:.4f} [{lo:.4f}, {hi:.4f}]")
        
        if auc_test > best_native_auc:
            best_native_auc = auc_test
            best_native_layer = layer

    results["best_native_layer"] = int(best_native_layer)
    results["best_native_auroc"] = float(best_native_auc)
    print(f"\n[native] Best: layer {best_native_layer} with AUROC={best_native_auc:.4f}")

    # Transfer gap
    transfer_gap = best_native_auc - best_transfer_auc
    results["transfer_gap"] = float(transfer_gap)
    print(f"\n[gap] Transfer gap: {transfer_gap:.4f} (native {best_native_auc:.4f} - transfer {best_transfer_auc:.4f})")

    # Save results
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[save] {OUT_JSON}")
    print("[done] Chinese transfer analysis complete.")

if __name__ == "__main__":
    main()
