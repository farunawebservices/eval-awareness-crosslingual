#!/usr/bin/env python3
"""Assign train/val/test splits to Chinese 70 cores (same pattern as English)."""

import json
from pathlib import Path

# Same split assignment as English H1
TRAIN_CORES = [1, 3, 5, 8, 10, 11, 13, 15, 17, 20, 21, 23, 25, 27, 29, 31, 33, 35, 37, 40, 41, 43, 45, 47, 50, 51, 53, 55, 57, 60, 61, 63, 65, 67, 70]
VAL_CORES = [2, 7, 12, 18, 22, 28, 32, 38, 42, 48, 52, 58, 62, 66, 68]
TEST_CORES = [4, 6, 9, 14, 16, 19, 24, 26, 30, 34, 36, 39, 44, 46, 49, 54, 56, 59, 64, 69]

assert len(TRAIN_CORES) == 35
assert len(VAL_CORES) == 15
assert len(TEST_CORES) == 20

# Load rendered prompts
input_path = Path("experiments/h2_chinese/h2_chinese_70_cores_rendered.jsonl")
output_path = Path("experiments/h2_chinese/h2_chinese_70_cores_with_splits.jsonl")

train_set = set(TRAIN_CORES)
val_set = set(VAL_CORES)
test_set = set(TEST_CORES)

with input_path.open("r", encoding="utf-8") as f_in, output_path.open("w", encoding="utf-8") as f_out:
    for line in f_in:
        rec = json.loads(line)
        core_id = rec["core_id"]
        if core_id in train_set:
            rec["split"] = "train"
        elif core_id in val_set:
            rec["split"] = "val"
        elif core_id in test_set:
            rec["split"] = "test"
        else:
            raise ValueError(f"Core {core_id} not assigned to any split")
        f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"Wrote {output_path}")

# Save split assignment JSON
split_json = {
    "train_cores": sorted(TRAIN_CORES),
    "val_cores": sorted(VAL_CORES),
    "test_cores": sorted(TEST_CORES),
    "metadata": {
        "total_cores": 70,
        "train_prompts": 35 * 4,
        "val_prompts": 15 * 4,
        "test_prompts": 20 * 4,
    }
}
with Path("experiments/h2_chinese/h2_chinese_splits.json").open("w", encoding="utf-8") as f:
    json.dump(split_json, f, indent=2, ensure_ascii=False)
print(f"Wrote experiments/h2_chinese/h2_chinese_splits.json")
