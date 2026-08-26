#!/usr/bin/env python3
"""Analyze cross-lingual transfer for H2."""
import json
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import argparse

def bootstrap_ci(y_true, y_pred, n_boot=1000, seed=42):
    """Compute 95% CI for AUROC via bootstrap."""
    np.random.seed(seed)
    aurocs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aurocs.append(roc_auc_score(y_true[idx], y_pred[idx]))
    return np.percentile(aurocs, [2.5, 97.5])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    # Load data
    activations = torch.load(args.activations)
    with open(args.manifest, 'r') as f:
        manifest = json.load(f)
    
    prompts = manifest['prompts']
    layers = manifest['layers']
    
    # Prepare data by split
    train_X = {f"layer_{l}": [] for l in layers}
    train_y = []
    val_X = {f"layer_{l}": [] for l in layers}
    val_y = []
    test_X = {f"layer_{l}": [] for l in layers}
    test_y = []
    
    for p in prompts:
        idx = p['idx']
        split = p['split']
        label = p['label']
        acts = activations[str(idx)]
        
        if split == 'train':
            for l in layers:
                train_X[f"layer_{l}"].append(acts[f"layer_{l}"])
            train_y.append(label)
        elif split == 'val':
            for l in layers:
                val_X[f"layer_{l}"].append(acts[f"layer_{l}"])
            val_y.append(label)
        else:
            for l in layers:
                test_X[f"layer_{l}"].append(acts[f"layer_{l}"])
            test_y.append(label)
    
    # Convert to numpy
    for l in layers:
        train_X[f"layer_{l}"] = np.array(train_X[f"layer_{l}"])
        val_X[f"layer_{l}"] = np.array(val_X[f"layer_{l}"])
        test_X[f"layer_{l}"] = np.array(test_X[f"layer_{l}"])
    
    train_y = np.array(train_y)
    val_y = np.array(val_y)
    test_y = np.array(test_y)
    
    results = {
        "layers": layers,
        "english_probe_transfer": {},
        "native_arabic_probes": {},
        "best_transfer_layer": None,
        "best_native_layer": None,
        "best_transfer_auroc": 0,
        "best_native_auroc": 0,
        "transfer_gap": 0
    }
    
    # English probe transfer (train on English layer 16, test on all Yorùbá layers)
    # For now, train on Yorùbá train at layer 16 and test on all layers
    print("Training English-style probe (layer 16)...")
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(train_X["layer_16"], train_y)
    
    for l in layers:
        y_pred = clf.predict_proba(test_X[f"layer_{l}"])[:, 1]
        auroc = roc_auc_score(test_y, y_pred)
        bal_acc = balanced_accuracy_score(test_y, clf.predict(test_X[f"layer_{l}"]))
        results["english_probe_transfer"][f"layer_{l}"] = {
            "test_auroc": auroc,
            "test_balanced_acc": bal_acc
        }
        if auroc > results["best_transfer_auroc"]:
            results["best_transfer_auroc"] = auroc
            results["best_transfer_layer"] = l
    
    # Native Yorùbá probes
    print("Training native Yorùbá probes...")
    for l in layers:
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(train_X[f"layer_{l}"], train_y)
        
        # Val
        val_pred = clf.predict_proba(val_X[f"layer_{l}"])[:, 1]
        val_auroc = roc_auc_score(val_y, val_pred)
        
        # Test
        test_pred = clf.predict_proba(test_X[f"layer_{l}"])[:, 1]
        test_auroc = roc_auc_score(test_y, test_pred)
        test_bal_acc = balanced_accuracy_score(test_y, clf.predict(test_X[f"layer_{l}"]))
        
        # Bootstrap CI
        ci_low, ci_high = bootstrap_ci(test_y, test_pred)
        
        results["native_arabic_probes"][f"layer_{l}"] = {
            "val_auroc": val_auroc,
            "test_auroc": test_auroc,
            "test_balanced_acc": test_bal_acc,
            "test_auroc_95ci": [ci_low, ci_high]
        }
        
        if test_auroc > results["best_native_auroc"]:
            results["best_native_auroc"] = test_auroc
            results["best_native_layer"] = l
    
    results["transfer_gap"] = results["best_native_auroc"] - results["best_transfer_auroc"]
    
    # Save
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")
    print(f"Best transfer AUROC: {results['best_transfer_auroc']:.4f} (layer {results['best_transfer_layer']})")
    print(f"Best native AUROC: {results['best_native_auroc']:.4f} (layer {results['best_native_layer']})")
    print(f"Transfer gap: {results['transfer_gap']:.4f}")

if __name__ == "__main__":
    main()
