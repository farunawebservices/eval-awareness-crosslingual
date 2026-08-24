"""Audit the tokenizer-only controlled prompt corpus (data/prompts.jsonl).

Checks (see docs/decision_log.md and docs/dataset_protocol.md purity check):
1. Condition coverage — every (pair_id, language) has exactly the 4 conditions.
2. Eval-marker leakage — no benchmark/eval vocabulary in deploy_plain/deploy_framed.
3. Length parity — per (pair_id, language), every condition's n_tokens is within
   LENGTH_TOLERANCE of that pair's deploy_plain n_tokens (the null-condition anchor).
4. Token-length statistics — reported alongside the parity check, for visibility.
5. Byte-identical rerun — rebuilding the corpus twice (and against disk) produces
   byte-identical output.

This script does not test H1, H2, or H3. It is a data-quality gate: a failure here
means the corpus is invalid and any downstream probe result would be
uninterpretable, not that a hypothesis has been confirmed or disconfirmed.

Exit code 0 iff checks 1, 2, 3, and 5 all pass. Rows with gold_answer=None are
reported by count, not treated as a failure.
"""

import json
import re
import statistics
import sys

from build_prompts import (
    CONDITIONS,
    LANGUAGES,
    OUTPUT_PATH,
    TASKS_PATH,
    TOKENIZER_NAME,
    build_rows,
    load_tasks,
    load_wrappers,
    serialize_rows,
)
from transformers import AutoTokenizer

DEPLOYMENT_CONDITIONS = {"deploy_plain", "deploy_framed"}

# Per-pair, per-language length-parity gate. Anchored to deploy_plain (the null
# condition) rather than the four-condition mean, and checked per pair rather
# than on aggregate condition means — a wrapper that's fine on average could
# still blow the budget on one specific task, and per-pair is what
# docs/dataset_protocol.md actually specifies ("within each pair, preserve...
# approximate token length").
LENGTH_TOLERANCE = 0.15

BANNED_WORDS = [
    "test", "tested", "testing",
    "evaluate", "evaluated", "evaluation", "evaluating",
    "benchmark", "benchmarked",
    "grade", "graded", "grading",
    "score", "scored", "scoring",
    "assess", "assessment",
    "exam", "examination",
    "quiz",
]
BANNED_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in BANNED_WORDS) + r")\b",
    re.IGNORECASE,
)


def load_disk_rows() -> list[dict]:
    rows = []
    with open(OUTPUT_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def check_condition_coverage(rows: list[dict]) -> list[str]:
    failures = []
    by_key = {}
    for row in rows:
        key = (row["pair_id"], row["language"])
        by_key.setdefault(key, []).append(row["condition"])

    tasks = load_tasks()
    expected_keys = {(t["pair_id"], lang) for t in tasks for lang in LANGUAGES}

    for key in sorted(expected_keys):
        conditions = by_key.get(key, [])
        missing = set(CONDITIONS) - set(conditions)
        extra = set(conditions) - set(CONDITIONS)
        duplicates = {c for c in conditions if conditions.count(c) > 1}
        if missing:
            failures.append(f"{key}: missing conditions {sorted(missing)}")
        if extra:
            failures.append(f"{key}: unexpected conditions {sorted(extra)}")
        if duplicates:
            failures.append(f"{key}: duplicate conditions {sorted(duplicates)}")

    for key in sorted(set(by_key) - expected_keys):
        failures.append(f"{key}: present in corpus but not in seed tasks / LANGUAGES")

    return failures


def check_leakage(rows: list[dict]) -> list[str]:
    failures = []
    for row in rows:
        if row["condition"] not in DEPLOYMENT_CONDITIONS:
            continue
        match = BANNED_PATTERN.search(row["user_text"])
        if match:
            failures.append(
                f"{row['pair_id']}/{row['language']}/{row['condition']}: "
                f"banned word {match.group(0)!r} found in user_text: {row['user_text']!r}"
            )
    return failures


def check_length_parity(rows: list[dict]) -> list[str]:
    failures = []
    by_pair = {}
    for row in rows:
        by_pair.setdefault((row["pair_id"], row["language"]), []).append(row)

    for (pair_id, language), pair_rows in sorted(by_pair.items()):
        anchor_rows = [r for r in pair_rows if r["condition"] == "deploy_plain"]
        if not anchor_rows:
            failures.append(f"{pair_id}/{language}: no deploy_plain row to anchor length parity on")
            continue
        anchor = anchor_rows[0]["n_tokens"]
        for row in pair_rows:
            rel = (row["n_tokens"] - anchor) / anchor
            if abs(rel) > LENGTH_TOLERANCE:
                failures.append(
                    f"{pair_id}/{language}/{row['condition']}: n_tokens={row['n_tokens']} "
                    f"vs deploy_plain={anchor} ({rel:+.1%}, limit ±{LENGTH_TOLERANCE:.0%})"
                )
    return failures


def report_token_length_stats(rows: list[dict]) -> None:
    all_lengths = [r["n_tokens"] for r in rows]
    print("\nToken-length statistics (n_tokens = full rendered chat-template string):")
    print(
        f"  overall: n={len(all_lengths)} min={min(all_lengths)} "
        f"mean={statistics.mean(all_lengths):.1f} max={max(all_lengths)} "
        f"stdev={statistics.stdev(all_lengths):.1f}"
    )
    for condition in CONDITIONS:
        lengths = [r["n_tokens"] for r in rows if r["condition"] == condition]
        if not lengths:
            continue
        stdev = statistics.stdev(lengths) if len(lengths) > 1 else 0.0
        print(
            f"  {condition:>15}: n={len(lengths)} min={min(lengths)} "
            f"mean={statistics.mean(lengths):.1f} max={max(lengths)} stdev={stdev:.1f}"
        )


def report_null_gold_answers(rows: list[dict]) -> None:
    null_rows = [r for r in rows if r["gold_answer"] is None]
    pair_ids = sorted({r["pair_id"] for r in null_rows})
    print(f"\nRows with gold_answer=None: {len(null_rows)} (pair_ids: {pair_ids})")


def check_byte_identical_rerun(disk_rows: list[dict]) -> list[str]:
    failures = []
    tasks = load_tasks()
    wrappers = load_wrappers()

    print(f"\nLoading tokenizer {TOKENIZER_NAME} to verify byte-identical rebuild...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    build_a = serialize_rows(build_rows(tasks, wrappers, tokenizer))
    build_b = serialize_rows(build_rows(tasks, wrappers, tokenizer))

    if build_a != build_b:
        failures.append("two in-process rebuilds were NOT byte-identical to each other")

    disk_text = serialize_rows(disk_rows)
    if build_a != disk_text:
        failures.append(
            f"in-process rebuild does NOT match {OUTPUT_PATH} on disk "
            f"(disk file is stale — rerun src/build_prompts.py)"
        )

    return failures


def main() -> int:
    if not OUTPUT_PATH.exists():
        print(f"ERROR: {OUTPUT_PATH} does not exist. Run src/build_prompts.py first.")
        return 1

    disk_rows = load_disk_rows()
    print(f"Loaded {len(disk_rows)} rows from {OUTPUT_PATH}")

    coverage_failures = check_condition_coverage(disk_rows)
    leakage_failures = check_leakage(disk_rows)
    parity_failures = check_length_parity(disk_rows)
    report_token_length_stats(disk_rows)
    report_null_gold_answers(disk_rows)
    rerun_failures = check_byte_identical_rerun(disk_rows)

    print("\n=== Condition coverage ===")
    if coverage_failures:
        for f in coverage_failures:
            print(f"  FAIL: {f}")
    else:
        print("  OK")

    print("\n=== Eval-marker leakage (deploy_plain, deploy_framed) ===")
    if leakage_failures:
        for f in leakage_failures:
            print(f"  FAIL: {f}")
    else:
        print("  OK")

    print(f"\n=== Length parity (per pair, anchored to deploy_plain, ±{LENGTH_TOLERANCE:.0%}) ===")
    if parity_failures:
        for f in parity_failures:
            print(f"  FAIL: {f}")
    else:
        print("  OK")

    print("\n=== Byte-identical rerun ===")
    if rerun_failures:
        for f in rerun_failures:
            print(f"  FAIL: {f}")
    else:
        print("  OK")

    all_failures = coverage_failures + leakage_failures + parity_failures + rerun_failures
    if all_failures:
        print(f"\nAUDIT FAILED: {len(all_failures)} failure(s)")
        return 1

    print("\nAUDIT PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
