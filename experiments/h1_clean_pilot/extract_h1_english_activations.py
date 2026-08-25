#!/usr/bin/env python3
"""
Step 1a (H1): English residual-stream activation extraction for Qwen/Qwen2.5-7B-Instruct.

Scope (strict): extraction ONLY. This script does not train probes, compute AUROC,
run FPR analyses, generate text, steer activations, or touch H2/H3. No network APIs
are used beyond the HuggingFace weight download performed by `transformers`.

Layer indexing convention
-------------------------
Qwen2 exposes its transformer blocks as `model.model.layers`, a 0-based ModuleList of
length `config.num_hidden_layers` (28 for Qwen2.5-7B). We interpret the requested
"layers" 12/16/20/24 as 0-based block indices, i.e.:

    requested layer L  ->  model.model.layers[L]   (the (L+1)-th block from the input)

We register a forward hook on each of those blocks and capture the block's OUTPUT,
which is the post-block residual stream (hidden_states after that block, before the
next block, and before the final RMSNorm). This is equivalent to
`output_hidden_states=True`'s `hidden_states[L + 1]`, since `hidden_states[0]` is the
embedding output. Both mappings are recorded in the manifest.

Outputs
-------
outputs/h1_english_activations.pt
outputs/h1_english_activation_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path("/workspace/eval-awareness-crosslingual")
INPUT_REL = Path("experiments/h1_clean_pilot/h1_70_cores_with_splits.jsonl")
OUT_PT_REL = Path("outputs/h1_english_activations.pt")
OUT_MANIFEST_REL = Path("outputs/h1_english_activation_manifest.json")

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
REQUESTED_LAYERS: List[int] = [12, 16, 20, 24]
EXPECTED_RECORD_COUNT = 280
MAX_PROMPT_TOKENS = 2048

LABEL_MAP: Dict[str, int] = {
    "eval_prose": 1,
    "eval_mcq": 1,
    "deploy_prose": 0,
    "deploy_mcq": 0,
}
REQUIRED_FIELDS = ("core_id", "family", "format", "prompt", "answer_key", "split")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_records(path: Path) -> List[dict]:
    """Load the JSONL, preserving file order, and validate it loudly."""
    if not path.is_file():
        raise FileNotFoundError(f"Input JSONL not found: {path}")

    records: List[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON on line {lineno} of {path}: {exc}") from exc
            records.append(rec)

    if len(records) != EXPECTED_RECORD_COUNT:
        raise ValueError(
            f"Unexpected record count: got {len(records)}, expected {EXPECTED_RECORD_COUNT} in {path}"
        )

    seen_keys = set()
    for idx, rec in enumerate(records):
        missing = [f for f in REQUIRED_FIELDS if f not in rec or rec[f] is None]
        if missing:
            raise ValueError(f"Record index {idx} missing required field(s) {missing}: {rec}")

        fmt = rec["format"]
        if fmt not in LABEL_MAP:
            raise ValueError(
                f"Record index {idx} (core_id={rec['core_id']}) has unknown format {fmt!r}; "
                f"expected one of {sorted(LABEL_MAP)}"
            )

        try:
            core_id = int(rec["core_id"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"core_id at index {idx} is not an integer: {rec['core_id']!r}") from exc
        rec["core_id"] = core_id

        prompt = rec["prompt"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Empty/non-string prompt at index {idx} (core_id={core_id}, format={fmt})")

        key = (core_id, fmt)
        if key in seen_keys:
            raise ValueError(f"Duplicate record for (core_id={core_id}, format={fmt}) at index {idx}")
        seen_keys.add(key)

    return records


def get_transformer_blocks(model: nn.Module) -> nn.ModuleList:
    """Locate Qwen2 transformer blocks; fail loudly if the structure is unexpected."""
    inner = getattr(model, "model", None)
    if inner is None:
        raise RuntimeError(
            f"Model {type(model).__name__} has no `.model` attribute; cannot locate Qwen blocks."
        )
    blocks = getattr(inner, "layers", None)
    if not isinstance(blocks, nn.ModuleList):
        raise RuntimeError(
            "Expected `model.model.layers` to be an nn.ModuleList of Qwen2 decoder blocks, "
            f"found {type(blocks).__name__}."
        )
    if len(blocks) == 0:
        raise RuntimeError("`model.model.layers` is empty; no transformer blocks to hook.")

    block_cls = type(blocks[0]).__name__
    if "DecoderLayer" not in block_cls:
        raise RuntimeError(
            f"Unexpected block class {block_cls!r}; expected a Qwen2*DecoderLayer. Aborting."
        )
    print(f"[structure] found {len(blocks)} blocks of type {block_cls}")
    return blocks


def make_hook(layer_idx: int, store: Dict[int, torch.Tensor]):
    def hook(_module, _args, output):
        tensor = output[0] if isinstance(output, (tuple, list)) else output
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(
                f"Hook on block {layer_idx} received non-tensor output of type {type(tensor).__name__}"
            )
        if tensor.dim() != 3:
            raise RuntimeError(
                f"Hook on block {layer_idx} expected (batch, seq, hidden), got shape {tuple(tensor.shape)}"
            )
        store[layer_idx] = tensor.detach()

    return hook


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool over non-padding prompt tokens. hidden: (1, T, H); mask: (1, T)."""
    mask = attention_mask.to(device=hidden.device, dtype=torch.float32).unsqueeze(-1)
    denom = mask.sum(dim=1)
    if float(denom.min()) < 1.0:
        raise RuntimeError("Attention mask selects zero tokens; cannot mean-pool.")
    pooled = (hidden.to(torch.float32) * mask).sum(dim=1) / denom
    return pooled.squeeze(0)


def check_finite(tensor: torch.Tensor, what: str) -> None:
    if torch.isnan(tensor).any():
        raise RuntimeError(f"NaN detected in {what}")
    if torch.isinf(tensor).any():
        raise RuntimeError(f"Inf detected in {what}")


def main() -> None:
    parser = argparse.ArgumentParser(description="H1 Step 1a: English activation extraction.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--max-tokens", type=int, default=MAX_PROMPT_TOKENS)
    parser.add_argument(
        "--use-chat-template",
        action="store_true",
        help="Wrap each prompt in the Qwen chat template before tokenizing "
             "(default: OFF, raw prompt text is tokenized).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script but torch.cuda.is_available() is False.")
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    input_path = (args.repo_root / INPUT_REL).resolve()
    out_pt = (args.repo_root / OUT_PT_REL).resolve()
    out_manifest = (args.repo_root / OUT_MANIFEST_REL).resolve()
    out_pt.parent.mkdir(parents=True, exist_ok=True)

    print(f"[input] {input_path}")
    input_sha256 = sha256_file(input_path)
    print(f"[input] sha256={input_sha256}")

    records = load_records(input_path)
    print(f"[input] loaded {len(records)} records (order preserved)")

    print(f"[model] loading {MODEL_NAME} once with dtype={dtype}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    model.to(device)
    model.eval()

    blocks = get_transformer_blocks(model)
    num_layers = int(model.config.num_hidden_layers)
    hidden_size = int(model.config.hidden_size)
    if len(blocks) != num_layers:
        raise RuntimeError(
            f"Block count {len(blocks)} != config.num_hidden_layers {num_layers}; aborting."
        )
    for layer in REQUESTED_LAYERS:
        if not (0 <= layer < num_layers):
            raise RuntimeError(
                f"Requested layer {layer} is out of range for a {num_layers}-block model."
            )

    model_device = str(next(model.parameters()).device)
    print(f"[model] device={model_device} num_layers={num_layers} hidden_size={hidden_size}")
    print(f"[model] requested layers (0-based block indices) = {REQUESTED_LAYERS}")
    print(f"[model] equivalent output_hidden_states indices  = {[l + 1 for l in REQUESTED_LAYERS]}")
    print(f"[data]  total record count = {len(records)}")

    store: Dict[int, torch.Tensor] = {}
    handles = [blocks[l].register_forward_hook(make_hook(l, store)) for l in REQUESTED_LAYERS]

    pooled: Dict[int, List[torch.Tensor]] = {l: [] for l in REQUESTED_LAYERS}
    labels: List[int] = []
    core_ids: List[int] = []
    families: List[str] = []
    formats: List[str] = []
    splits: List[str] = []
    prompts: List[str] = []

    t0 = time.time()
    try:
        with torch.inference_mode():
            for idx, rec in enumerate(records):
                core_id, fmt, prompt = rec["core_id"], rec["format"], rec["prompt"]

                if args.use_chat_template:
                    text = tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                else:
                    text = prompt

                enc = tokenizer(text, return_tensors="pt", padding=False, truncation=False)
                seq_len = int(enc["input_ids"].shape[1])
                if seq_len > args.max_tokens:
                    raise RuntimeError(
                        f"Prompt too long ({seq_len} > {args.max_tokens} tokens) for "
                        f"core_id={core_id}, format={fmt}. Refusing to truncate silently."
                    )

                input_ids = enc["input_ids"].to(device)
                attention_mask = enc["attention_mask"].to(device)

                store.clear()
                model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

                missing = [l for l in REQUESTED_LAYERS if l not in store]
                if missing:
                    raise RuntimeError(
                        f"Hooks failed to fire for block(s) {missing} at record index {idx} "
                        f"(core_id={core_id}, format={fmt})"
                    )

                for layer in REQUESTED_LAYERS:
                    hidden = store[layer]
                    if hidden.shape[0] != 1 or hidden.shape[1] != seq_len or hidden.shape[2] != hidden_size:
                        raise RuntimeError(
                            f"Shape mismatch at record {idx} layer {layer}: got {tuple(hidden.shape)}, "
                            f"expected (1, {seq_len}, {hidden_size})"
                        )
                    if idx == 0:
                        print(
                            f"[first-forward] layer_{layer}: shape={tuple(hidden.shape)} "
                            f"dtype={hidden.dtype} device={hidden.device} "
                            f"norm={hidden.to(torch.float32).norm().item():.4f}"
                        )
                    vec = mean_pool(hidden, attention_mask).to("cpu", torch.float32).contiguous()
                    check_finite(vec, f"pooled activation (record {idx}, core_id={core_id}, layer {layer})")
                    if vec.shape != (hidden_size,):
                        raise RuntimeError(
                            f"Pooled vector shape {tuple(vec.shape)} != ({hidden_size},) at record {idx}"
                        )
                    pooled[layer].append(vec)
                    if idx == 0:
                        print(
                            f"[first-forward] layer_{layer} pooled: shape={tuple(vec.shape)} "
                            f"dtype={vec.dtype} device={vec.device} norm={vec.norm().item():.4f}"
                        )

                labels.append(LABEL_MAP[fmt])
                core_ids.append(core_id)
                families.append(str(rec["family"]))
                formats.append(str(fmt))
                splits.append(str(rec["split"]))
                prompts.append(prompt)

                if (idx + 1) % 40 == 0 or (idx + 1) == len(records):
                    print(f"[progress] {idx + 1}/{len(records)} records ({time.time() - t0:.1f}s)")
    finally:
        for handle in handles:
            handle.remove()
        store.clear()

    activations: Dict[str, torch.Tensor] = {}
    for layer in REQUESTED_LAYERS:
        if len(pooled[layer]) != EXPECTED_RECORD_COUNT:
            raise RuntimeError(
                f"layer_{layer}: collected {len(pooled[layer])} vectors, expected {EXPECTED_RECORD_COUNT}"
            )
        mat = torch.stack(pooled[layer], dim=0)
        if mat.shape != (EXPECTED_RECORD_COUNT, hidden_size):
            raise RuntimeError(f"layer_{layer} final shape {tuple(mat.shape)} is wrong")
        if mat.dtype != torch.float32 or mat.device.type != "cpu":
            raise RuntimeError(f"layer_{layer} must be CPU float32, got {mat.device}/{mat.dtype}")
        check_finite(mat, f"layer_{layer} activation matrix")
        activations[f"layer_{layer}"] = mat
        print(
            f"[final] layer_{layer}: shape={tuple(mat.shape)} dtype={mat.dtype} "
            f"device={mat.device} mean_row_norm={mat.norm(dim=1).mean().item():.4f}"
        )

    for name, seq in (
        ("labels", labels), ("core_ids", core_ids), ("families", families),
        ("formats", formats), ("splits", splits), ("prompts", prompts),
    ):
        if len(seq) != EXPECTED_RECORD_COUNT:
            raise RuntimeError(f"{name} has length {len(seq)}, expected {EXPECTED_RECORD_COUNT}")

    payload = {
        "activations": activations,
        "labels": torch.tensor(labels, dtype=torch.long),
        "core_ids": core_ids,
        "families": families,
        "formats": formats,
        "splits": splits,
        "prompts": prompts,
        "record_indices": list(range(EXPECTED_RECORD_COUNT)),
    }
    if payload["labels"].shape != (EXPECTED_RECORD_COUNT,):
        raise RuntimeError(f"labels tensor shape {tuple(payload['labels'].shape)} is wrong")

    torch.save(payload, out_pt)
    print(f"[save] {out_pt}")

    manifest = {
        "step": "1a_english_activation_extraction_h1",
        "model_name": MODEL_NAME,
        "device": model_device,
        "dtype": str(dtype),
        "activation_storage_dtype": "torch.float32",
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "requested_layers": REQUESTED_LAYERS,
        "block_indices_used": REQUESTED_LAYERS,
        "layer_index_convention": (
            "Requested layer L maps to the 0-based Qwen block model.model.layers[L]; the hook "
            "captures that block's output (post-block residual stream), equivalent to "
            "output_hidden_states[L+1]."
        ),
        "hidden_states_equivalent_indices": [l + 1 for l in REQUESTED_LAYERS],
        "pooling": "mean over non-padding prompt tokens (attention_mask == 1)",
        "used_chat_template": bool(args.use_chat_template),
        "max_prompt_tokens": int(args.max_tokens),
        "truncation": "disabled (errors instead of truncating)",
        "label_map": LABEL_MAP,
        "input_path": str(input_path),
        "input_sha256": input_sha256,
        "output_path": str(out_pt),
        "manifest_path": str(out_manifest),
        "record_count": EXPECTED_RECORD_COUNT,
        "label_counts": {"1": int(sum(labels)), "0": int(len(labels) - sum(labels))},
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "elapsed_seconds": round(time.time() - t0, 2),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with out_manifest.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"[save] {out_manifest}")
    print("[done] Step 1a extraction complete.")


if __name__ == "__main__":
    main()
