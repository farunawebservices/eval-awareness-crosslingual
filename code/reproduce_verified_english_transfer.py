#!/usr/bin/env python3
"""
Independently reproduce English -> target zero-shot transfer.

The English source probe is fit once on:
    outputs/h1_english_activations.pt
    English TRAIN rows only
    layer 16 only.

Target evaluation uses:
    - Chinese/Arabic: split metadata embedded in target .pt artifact.
    - Yorùbá: per-index target activations plus separate manifest.

This script never writes to outputs/*.json; it writes only to audit/.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

ROOT = Path(".")
ENGLISH_ACTIVATIONS = ROOT / "outputs/h1_english_activations.pt"
ENGLISH_PROBE_LAYER = 16
EXPECTED_LAYERS = [12, 16, 20, 24]
ALLOWED_SPLITS = {"train", "val", "test"}


def to_numpy(x):
    """Convert torch / NumPy-like object to a NumPy array."""
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


def binary_counts(y):
    values, counts = np.unique(np.asarray(y, dtype=int), return_counts=True)
    return {str(int(v)): int(n) for v, n in zip(values, counts)}


def core_split_audit(core_ids, splits):
    """Reject target data if a semantic core appears in multiple splits."""
    by_core = defaultdict(set)
    for core, split in zip(core_ids, splits):
        by_core[int(core)].add(str(split))

    bad = {
        int(core): sorted(values)
        for core, values in by_core.items()
        if len(values) != 1
    }
    if bad:
        preview = list(bad.items())[:10]
        raise ValueError(f"Core leakage: {len(bad)} cores occur in >1 split: {preview}")

    return {
        "n_unique_cores": len(by_core),
        "core_split_counts": dict(Counter(
            next(iter(values)) for values in by_core.values()
        )),
        "n_cores_with_multiple_splits": 0,
    }


def load_english_probe():
    """Train exactly one source probe: English train split, layer 16."""
    obj = torch.load(
        ENGLISH_ACTIVATIONS,
        map_location="cpu",
        weights_only=False,
    )

    required = {"activations", "labels", "splits", "core_ids"}
    missing = required - set(obj.keys())
    if missing:
        raise KeyError(f"English artifact missing fields: {missing}")

    layer_key = f"layer_{ENGLISH_PROBE_LAYER}"
    if layer_key not in obj["activations"]:
        raise KeyError(f"English artifact lacks {layer_key}")

    labels = to_numpy(obj["labels"]).astype(int)
    splits = np.asarray(obj["splits"], dtype=object)
    core_ids = np.asarray(obj["core_ids"])

    unexpected = sorted(set(splits) - ALLOWED_SPLITS)
    if unexpected:
        raise ValueError(f"English has unexpected split names: {unexpected}")

    split_audit = core_split_audit(core_ids, splits)
    train_mask = splits == "train"

    X_train = to_numpy(obj["activations"][layer_key][train_mask]).astype(np.float32)
    y_train = labels[train_mask]

    if set(np.unique(y_train)) != {0, 1}:
        raise ValueError("English training split is not binary.")

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    metadata = {
        "source_artifact": str(ENGLISH_ACTIVATIONS),
        "source_layer": ENGLISH_PROBE_LAYER,
        "n_rows": int(len(labels)),
        "split_counts": dict(Counter(map(str, splits))),
        "train_label_counts": binary_counts(y_train),
        "core_split_audit": split_audit,
        "coefficient_shape": list(clf.coef_.shape),
    }
    return clf, metadata


def load_stacked_target(target_path):
    """
    Load Chinese/Arabic style artifacts:
    obj['activations']['layer_16'][row_idx]
    and parallel labels/splits/core_ids arrays.
    """
    obj = torch.load(target_path, map_location="cpu", weights_only=False)

    required = {"activations", "labels", "splits", "core_ids"}
    missing = required - set(obj.keys())
    if missing:
        raise KeyError(f"{target_path} missing fields: {missing}")

    activations = obj["activations"]
    labels = to_numpy(obj["labels"]).astype(int)
    splits = np.asarray(obj["splits"], dtype=object)
    core_ids = np.asarray(obj["core_ids"])

    if len(labels) != len(splits) or len(labels) != len(core_ids):
        raise ValueError(
            f"{target_path}: labels/splits/core_ids have inconsistent lengths"
        )

    missing_layers = [
        f"layer_{layer}" for layer in EXPECTED_LAYERS
        if f"layer_{layer}" not in activations
    ]
    if missing_layers:
        raise KeyError(f"{target_path}: missing activation layers {missing_layers}")

    for layer in EXPECTED_LAYERS:
        n = len(activations[f"layer_{layer}"])
        if n != len(labels):
            raise ValueError(
                f"{target_path}: layer_{layer} has {n} rows, expected {len(labels)}"
            )

    unexpected = sorted(set(splits) - ALLOWED_SPLITS)
    if unexpected:
        raise ValueError(f"{target_path}: unexpected splits {unexpected}")

    return {
        "layout": "stacked_by_layer",
        "path": str(target_path),
        "activations": activations,
        "labels": labels,
        "splits": splits,
        "core_ids": core_ids,
        "formats": list(obj.get("formats", [])),
        "n_rows": int(len(labels)),
    }


def load_indexed_target(target_pt, target_manifest):
    """
    Load Yorùbá style artifacts:
    obj[row_idx]['layer_16']
    with row metadata in manifest['prompts'].
    """
    acts = torch.load(target_pt, map_location="cpu", weights_only=False)
    manifest = json.loads(Path(target_manifest).read_text(encoding="utf-8"))

    if "prompts" not in manifest or "layers" not in manifest:
        raise KeyError(f"{target_manifest} must contain prompts and layers")

    prompts = manifest["prompts"]
    layers = list(manifest["layers"])

    if layers != EXPECTED_LAYERS:
        raise ValueError(f"{target_manifest}: unexpected layer list {layers}")

    if len(acts) != len(prompts):
        raise ValueError(
            f"Activation/manifest mismatch: {len(acts)} activation entries vs "
            f"{len(prompts)} manifest records"
        )

    expected_indices = set(range(len(prompts)))
    actual_indices = set(acts.keys())
    if actual_indices != expected_indices:
        raise ValueError("Indexed activation keys are not contiguous 0..N-1.")

    indices = np.asarray([int(p["idx"]) for p in prompts])
    labels = np.asarray([int(p["label"]) for p in prompts])
    splits = np.asarray([p["split"] for p in prompts], dtype=object)
    core_ids = np.asarray([int(p["core_id"]) for p in prompts])
    formats = [p.get("format") for p in prompts]

    if sorted(indices.tolist()) != list(range(len(prompts))):
        raise ValueError("Manifest indices are not a permutation of 0..N-1.")

    unexpected = sorted(set(splits) - ALLOWED_SPLITS)
    if unexpected:
        raise ValueError(f"{target_manifest}: unexpected splits {unexpected}")

    for i in [0, len(prompts) - 1]:
        for layer in EXPECTED_LAYERS:
            if f"layer_{layer}" not in acts[i]:
                raise KeyError(f"Activation {i} lacks layer_{layer}")

    return {
        "layout": "indexed_by_row",
        "path": str(target_pt),
        "manifest": str(target_manifest),
        "activations": acts,
        "indices": indices,
        "labels": labels,
        "splits": splits,
        "core_ids": core_ids,
        "formats": formats,
        "n_rows": int(len(labels)),
    }


def target_matrix(target, layer, indices):
    """Return an N x hidden_size matrix for target rows at a layer."""
    key = f"layer_{layer}"

    if target["layout"] == "stacked_by_layer":
        return to_numpy(target["activations"][key][indices]).astype(np.float32)

    if target["layout"] == "indexed_by_row":
        return np.stack([
            to_numpy(target["activations"][int(idx)][key]).astype(np.float32)
            for idx in indices
        ])

    raise ValueError(f"Unknown target layout: {target['layout']}")


def run_target(target_name, target, english_probe, english_metadata, output_path):
    labels = target["labels"]
    splits = target["splits"]
    core_ids = target["core_ids"]

    split_audit = core_split_audit(core_ids, splits)

    train_idx = np.flatnonzero(splits == "train")
    val_idx = np.flatnonzero(splits == "val")
    test_idx = np.flatnonzero(splits == "test")

    if min(len(train_idx), len(val_idx), len(test_idx)) == 0:
        raise ValueError(f"{target_name}: an expected split is empty")

    y_train = labels[train_idx]
    y_val = labels[val_idx]
    y_test = labels[test_idx]

    for split_name, y in [
        ("train", y_train), ("val", y_val), ("test", y_test)
    ]:
        if set(np.unique(y)) != {0, 1}:
            raise ValueError(
                f"{target_name}: {split_name} is not binary: {binary_counts(y)}"
            )

    print("=" * 88)
    print(f"INDEPENDENT ENGLISH → {target_name.upper()} TRANSFER REPRODUCTION")
    print("=" * 88)
    print(
        f"English training: layer={ENGLISH_PROBE_LAYER}, "
        f"n={english_metadata['split_counts']['train']}, "
        f"labels={english_metadata['train_label_counts']}"
    )
    print(
        f"{target_name} artifact: {target['path']} "
        f"({target['layout']}, n={target['n_rows']})"
    )
    print(
        f"{target_name} rows: train={len(train_idx)}, val={len(val_idx)}, "
        f"test={len(test_idx)}"
    )
    print(
        f"{target_name} test labels: {binary_counts(y_test)}; "
        f"core-disjoint split: yes"
    )

    result = {
        "description": (
            "Independent zero-shot transfer reproduction. The source "
            "logistic-regression probe is trained only on English training "
            f"activations at layer {ENGLISH_PROBE_LAYER}; no target-language "
            "examples are used by the source probe."
        ),
        "target_name": target_name,
        "english_training": english_metadata,
        "target_artifact": {
            "path": target["path"],
            "manifest": target.get("manifest"),
            "layout": target["layout"],
            "n_rows": target["n_rows"],
            "split_row_counts": {
                "train": int(len(train_idx)),
                "val": int(len(val_idx)),
                "test": int(len(test_idx)),
            },
            "test_label_counts": binary_counts(y_test),
            "core_split_audit": split_audit,
        },
        "english_to_target": {},
        "native_target": {},
    }

    for layer in EXPECTED_LAYERS:
        X_train = target_matrix(target, layer, train_idx)
        X_val = target_matrix(target, layer, val_idx)
        X_test = target_matrix(target, layer, test_idx)

        transfer_probs = english_probe.predict_proba(X_test)[:, 1]
        transfer_pred = english_probe.predict(X_test)

        transfer_auc = float(roc_auc_score(y_test, transfer_probs))
        transfer_bacc = float(balanced_accuracy_score(y_test, transfer_pred))

        native_probe = LogisticRegression(max_iter=1000, random_state=42)
        native_probe.fit(X_train, y_train)
        native_val_probs = native_probe.predict_proba(X_val)[:, 1]
        native_test_probs = native_probe.predict_proba(X_test)[:, 1]

        native_val_auc = float(roc_auc_score(y_val, native_val_probs))
        native_test_auc = float(roc_auc_score(y_test, native_test_probs))
        native_test_bacc = float(
            balanced_accuracy_score(y_test, native_probe.predict(X_test))
        )

        result["english_to_target"][f"layer_{layer}"] = {
            "test_auroc": transfer_auc,
            "test_balanced_acc_at_0_5": transfer_bacc,
        }
        result["native_target"][f"layer_{layer}"] = {
            "val_auroc": native_val_auc,
            "test_auroc": native_test_auc,
            "test_balanced_acc_at_0_5": native_test_bacc,
        }

        print(
            f"Layer {layer:>2}: "
            f"English→{target_name} AUROC={transfer_auc:.4f}, "
            f"BAcc={transfer_bacc:.4f}; "
            f"native AUROC={native_test_auc:.4f}, "
            f"BAcc={native_test_bacc:.4f}"
        )

    best_transfer_layer = max(
        EXPECTED_LAYERS,
        key=lambda layer: result["english_to_target"][f"layer_{layer}"]["test_auroc"],
    )
    best_native_layer = max(
        EXPECTED_LAYERS,
        key=lambda layer: result["native_target"][f"layer_{layer}"]["test_auroc"],
    )

    result["best_transfer_layer"] = int(best_transfer_layer)
    result["best_transfer_auroc"] = result["english_to_target"][
        f"layer_{best_transfer_layer}"
    ]["test_auroc"]
    result["best_native_layer"] = int(best_native_layer)
    result["best_native_auroc"] = result["native_target"][
        f"layer_{best_native_layer}"
    ]["test_auroc"]
    result["transfer_gap"] = (
        result["best_native_auroc"] - result["best_transfer_auroc"]
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("-" * 88)
    print(
        f"Best English→{target_name}: "
        f"{result['best_transfer_auroc']:.4f} at L{best_transfer_layer}"
    )
    print(
        f"Best native {target_name}: "
        f"{result['best_native_auroc']:.4f} at L{best_native_layer}"
    )
    print(f"Transfer gap: {result['transfer_gap']:.4f}")
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-name",
        choices=["chinese", "arabic", "yoruba"],
        required=True,
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    english_probe, english_metadata = load_english_probe()

    if args.target_name == "chinese":
        target = load_stacked_target(
            ROOT / "outputs/h2_chinese_activations_4layers.pt"
        )
    elif args.target_name == "arabic":
        target = load_stacked_target(
            ROOT / "outputs/h2_arabic_activations_4layers.pt"
        )
    else:
        target = load_indexed_target(
            ROOT / "outputs/h2_yoruba_activations_4layers.pt",
            ROOT / "outputs/h2_yoruba_activation_manifest_4layers.json",
        )

    run_target(
        args.target_name,
        target,
        english_probe,
        english_metadata,
        args.output,
    )


if __name__ == "__main__":
    main()
