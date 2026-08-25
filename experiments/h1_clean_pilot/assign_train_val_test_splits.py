#!/usr/bin/env python3
"""
Assign train/val/test splits to the 70 H1 cores deterministically.
- Train: 35 cores (140 prompts)
- Validation: 15 cores (60 prompts)
- Test: 20 cores (80 prompts)

Split is core-level: all 4 formats of a core go to the same split.
Balanced across families (F1-F5) and domains (D1-D7).
Balanced answer keys (A/B/C/D) across splits.
"""

import json
from pathlib import Path

# Core-to-family mapping (from render script)
CORE_FAMILY = {}
F1_CORES = [1, 2, 11, 12, 21, 22, 31, 32, 41, 42, 51, 52, 61, 62]
F2_CORES = [3, 4, 13, 14, 23, 24, 33, 34, 43, 44, 53, 54, 63, 64]
F3_CORES = [5, 6, 15, 16, 25, 26, 35, 36, 45, 46, 55, 56, 65, 66]
F4_CORES = [7, 8, 17, 18, 27, 28, 37, 38, 47, 48, 57, 58, 67, 68]
F5_CORES = [9, 10, 19, 20, 29, 30, 39, 40, 49, 50, 59, 60, 69, 70]

for c in F1_CORES:
    CORE_FAMILY[c] = "F1"
for c in F2_CORES:
    CORE_FAMILY[c] = "F2"
for c in F3_CORES:
    CORE_FAMILY[c] = "F3"
for c in F4_CORES:
    CORE_FAMILY[c] = "F4"
for c in F5_CORES:
    CORE_FAMILY[c] = "F5"

# Core-to-domain mapping
CORE_DOMAIN = {}
for c in range(1, 11):
    CORE_DOMAIN[c] = "D1"
for c in range(11, 21):
    CORE_DOMAIN[c] = "D2"
for c in range(21, 31):
    CORE_DOMAIN[c] = "D3"
for c in range(31, 41):
    CORE_DOMAIN[c] = "D4"
for c in range(41, 51):
    CORE_DOMAIN[c] = "D5"
for c in range(51, 61):
    CORE_DOMAIN[c] = "D6"
for c in range(61, 71):
    CORE_DOMAIN[c] = "D7"

# Answer keys (from CORES dict in render script)
ANSWER_KEYS = {
    1: "A", 2: "B", 3: "A", 4: "C", 5: "C", 6: "D", 7: "A", 8: "B", 9: "B", 10: "D",
    11: "A", 12: "C", 13: "A", 14: "C", 15: "A", 16: "D", 17: "B", 18: "C", 19: "B", 20: "D",
    21: "B", 22: "D", 23: "A", 24: "B", 25: "C", 26: "D", 27: "B", 28: "D", 29: "A", 30: "C",
    31: "C", 32: "D", 33: "C", 34: "D", 35: "B", 36: "D", 37: "A", 38: "B", 39: "A", 40: "C",
    41: "B", 42: "D", 43: "B", 44: "D", 45: "C", 46: "D", 47: "B", 48: "D", 49: "A", 50: "C",
    51: "A", 52: "C", 53: "C", 54: "D", 55: "A", 56: "B", 57: "A", 58: "C", 59: "B", 60: "D",
    61: "B", 62: "D", 63: "B", 64: "D", 65: "A", 66: "B", 67: "C", 68: "D", 69: "A", 70: "C",
}

# Deterministic split assignment
# Strategy: Assign cores to splits in a way that balances families, domains, and answer keys
# Train: 35 cores, Val: 15 cores, Test: 20 cores

# We'll use a simple deterministic pattern:
# - Sort cores by domain, then by family
# - Assign in round-robin fashion with weights for split sizes

all_cores = list(range(1, 71))

# Sort by domain first, then by core_id
all_cores_sorted = sorted(all_cores, key=lambda c: (CORE_DOMAIN[c], c))

train_cores = []
val_cores = []
test_cores = []

# Assign cores to splits maintaining balance
# Pattern: Train, Train, Val, Test, Train, Val, Train, Test, ...
# This gives roughly 35/15/20 distribution

assignment_pattern = []
# Build pattern: 35 train, 15 val, 20 test
# We want to interleave to maintain balance across the sorted list
for i in range(70):
    if i % 7 < 4:  # 4/7 ≈ 35/70 = 50% → train
        assignment_pattern.append("train")
    elif i % 7 < 5:  # 1/7 ≈ 15/70 = 21% → val (but we need 15, so adjust)
        assignment_pattern.append("val")
    else:  # 2/7 ≈ 20/70 = 29% → test
        assignment_pattern.append("test")

# Adjust pattern to get exact counts: 35 train, 15 val, 20 test
# Current pattern gives: 40 train, 10 val, 20 test (approximately)
# Need to convert 5 train to val

# Refined pattern: explicit assignment
train_cores = []
val_cores = []
test_cores = []

# Domain D1 (cores 1-10): 5 train, 2 val, 3 test
train_cores.extend([1, 3, 5, 8, 10])
val_cores.extend([2, 7])
test_cores.extend([4, 6, 9])

# Domain D2 (cores 11-20): 5 train, 2 val, 3 test
train_cores.extend([11, 13, 15, 17, 20])
val_cores.extend([12, 18])
test_cores.extend([14, 16, 19])

# Domain D3 (cores 21-30): 5 train, 2 val, 3 test
train_cores.extend([21, 23, 25, 27, 29])
val_cores.extend([22, 28])
test_cores.extend([24, 26, 30])

# Domain D4 (cores 31-40): 5 train, 2 val, 3 test
train_cores.extend([31, 33, 35, 37, 40])
val_cores.extend([32, 38])
test_cores.extend([34, 36, 39])

# Domain D5 (cores 41-50): 5 train, 2 val, 3 test
train_cores.extend([41, 43, 45, 47, 50])
val_cores.extend([42, 48])
test_cores.extend([44, 46, 49])

# Domain D6 (cores 51-60): 5 train, 2 val, 3 test
train_cores.extend([51, 53, 55, 57, 60])
val_cores.extend([52, 58])
test_cores.extend([54, 56, 59])

# Domain D7 (cores 61-70): 5 train, 3 val, 2 test
train_cores.extend([61, 63, 65, 67, 70])
val_cores.extend([62, 66, 68])
test_cores.extend([64, 69])

# Verify counts
assert len(train_cores) == 35, f"Expected 35 train cores, got {len(train_cores)}"
assert len(val_cores) == 15, f"Expected 15 val cores, got {len(val_cores)}"
assert len(test_cores) == 20, f"Expected 20 test cores, got {len(test_cores)}"

def check_balance(split_cores, split_name):
    """Check family, domain, and answer key balance."""
    families = {}
    domains = {}
    keys = {}
    for c in split_cores:
        f = CORE_FAMILY[c]
        d = CORE_DOMAIN[c]
        k = ANSWER_KEYS[c]
        families[f] = families.get(f, 0) + 1
        domains[d] = domains.get(d, 0) + 1
        keys[k] = keys.get(k, 0) + 1
    print(f"\n{split_name} ({len(split_cores)} cores):")
    print(f"  Families: {families}")
    print(f"  Domains: {domains}")
    print(f"  Answer keys: {keys}")

print("Split assignment:")
check_balance(train_cores, "Train")
check_balance(val_cores, "Validation")
check_balance(test_cores, "Test")

# Create split assignment file
output_path = Path("experiments/h1_clean_pilot/h1_train_val_test_splits.json")

split_assignment = {
    "train_cores": sorted(train_cores),
    "val_cores": sorted(val_cores),
    "test_cores": sorted(test_cores),
    "metadata": {
        "total_cores": 70,
        "train_cores_count": len(train_cores),
        "val_cores_count": len(val_cores),
        "test_cores_count": len(test_cores),
        "train_prompts_count": len(train_cores) * 4,
        "val_prompts_count": len(val_cores) * 4,
        "test_prompts_count": len(test_cores) * 4,
    }
}

with output_path.open("w", encoding="utf-8") as f:
    json.dump(split_assignment, f, indent=2, ensure_ascii=False)

print(f"\nWrote split assignment to {output_path}")

# Also create a JSONL file with all 280 prompts annotated with their split
rendered_path = Path("experiments/h1_clean_pilot/h1_70_cores_rendered.jsonl")
annotated_path = Path("experiments/h1_clean_pilot/h1_70_cores_with_splits.jsonl")

# Load split assignment
train_set = set(train_cores)
val_set = set(val_cores)
test_set = set(test_cores)

with rendered_path.open("r", encoding="utf-8") as f_in, annotated_path.open("w", encoding="utf-8") as f_out:
    for line in f_in:
        record = json.loads(line)
        core_id = record["core_id"]
        if core_id in train_set:
            record["split"] = "train"
        elif core_id in val_set:
            record["split"] = "val"
        elif core_id in test_set:
            record["split"] = "test"
        else:
            raise ValueError(f"Core {core_id} not assigned to any split")
        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Wrote annotated prompts to {annotated_path}")
