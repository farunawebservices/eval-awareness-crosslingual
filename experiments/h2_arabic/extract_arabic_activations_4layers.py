#!/usr/bin/env python3
"""Extract Arabic H2 activations at layers 12, 16, 20, 24 (parallel to Chinese)."""

import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

REPO = Path("/workspace/eval-awareness-crosslingual")
INPUT_JSONL = REPO / "experiments/h2_arabic/h2_arabic_70_cores_with_splits.jsonl"
OUT_PT = REPO / "outputs/h2_arabic_activations_4layers.pt"
OUT_MANIFEST = REPO / "outputs/h2_arabic_activation_manifest_4layers.json"

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
LAYERS = [12, 16, 20, 24]
HIDDEN = 3584

def mean_pool(hidden, attention_mask):
    mask = attention_mask.to(device=hidden.device, dtype=torch.float32).unsqueeze(-1)
    denom = mask.sum(dim=1)
    pooled = (hidden.to(torch.float32) * mask).sum(dim=1) / denom
    return pooled.squeeze(0)

def main():
    print(f"[model] loading {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.to("cuda")
    model.eval()

    # Register hooks on all four layers
    store = {}
    handles = []
    for layer in LAYERS:
        def make_hook(l):
            def hook(_module, _args, output):
                tensor = output[0] if isinstance(output, (tuple, list)) else output
                store[l] = tensor.detach()
            return hook
        handle = model.model.layers[layer].register_forward_hook(make_hook(layer))
        handles.append(handle)

    # Load prompts
    records = []
    with INPUT_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    print(f"[input] loaded {len(records)} Arabic prompts")

    # Extract activations
    pooled = {l: [] for l in LAYERS}
    labels = []
    core_ids = []
    families = []
    formats = []
    splits = []
    prompts = []
    domains = []

    device = torch.device("cuda")

    with torch.inference_mode():
        for i, rec in enumerate(records):
            prompt = rec["prompt"]
            enc = tokenizer(prompt, return_tensors="pt", padding=False, truncation=False)
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            seq_len = int(input_ids.shape[1])

            store.clear()
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

            missing = [l for l in LAYERS if l not in store]
            if missing:
                raise RuntimeError(f"Hooks failed for layers {missing} at record {i}")

            for layer in LAYERS:
                hidden = store[layer]
                vec = mean_pool(hidden, attention_mask).to("cpu", torch.float32).contiguous()
                if vec.shape != (HIDDEN,):
                    raise RuntimeError(f"Prompt {i} layer {layer} pooled vector shape {tuple(vec.shape)} != ({HIDDEN},)")
                pooled[layer].append(vec)

            labels.append(1 if rec["format"] in ["eval_prose", "eval_mcq"] else 0)
            core_ids.append(rec["core_id"])
            families.append(rec["family"])
            formats.append(rec["format"])
            splits.append(rec["split"])
            prompts.append(prompt)
            domains.append(rec.get("domain", ""))

            if (i + 1) % 40 == 0 or (i + 1) == len(records):
                print(f"[progress] {i + 1}/{len(records)}")

    for handle in handles:
        handle.remove()

    # Stack activations
    activations = {}
    for layer in LAYERS:
        mat = torch.stack(pooled[layer], dim=0)
        if mat.shape != (len(records), HIDDEN):
            raise RuntimeError(f"Layer {layer} final shape {tuple(mat.shape)} wrong")
        activations[f"layer_{layer}"] = mat
        print(f"[final] layer_{layer}: shape={tuple(mat.shape)}, mean_row_norm={mat.norm(dim=1).mean().item():.4f}")

    # Save
    payload = {
        "activations": activations,
        "labels": torch.tensor(labels, dtype=torch.long),
        "core_ids": core_ids,
        "families": families,
        "formats": formats,
        "splits": splits,
        "prompts": prompts,
        "domains": domains,
        "record_indices": list(range(len(records))),
    }

    OUT_PT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, OUT_PT)
    print(f"[save] {OUT_PT}")

    # Manifest
    manifest = {
        "model_name": MODEL_NAME,
        "layers": LAYERS,
        "hidden_size": HIDDEN,
        "record_count": len(records),
        "label_counts": {"1": sum(labels), "0": len(labels) - sum(labels)},
        "split_counts": {
            "train": splits.count("train"),
            "val": splits.count("val"),
            "test": splits.count("test"),
        },
    }
    with OUT_MANIFEST.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[save] {OUT_MANIFEST}")
    print("[done] Arabic 4-layer activation extraction complete.")

if __name__ == "__main__":
    main()
