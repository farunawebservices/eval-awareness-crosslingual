"""Audit the tokenizer-only controlled prompt corpus (data/prompts.jsonl).

Checks (see docs/decision_log.md and docs/dataset_protocol.md purity check):
1. Condition coverage — every (pair_id, language) has exactly the 4 conditions.
2. Eval-marker leakage — no benchmark/eval vocabulary in deploy_plain/deploy_framed,
   checked against that row's own language's banned-word list (BANNED_WORDS_BY_LANGUAGE).
   A row whose language has no list yet is a hard failure, not a skipped check.
3. Length parity — per (pair_id, language), every condition's n_tokens is within
   LENGTH_TOLERANCE of that pair's deploy_plain n_tokens (the null-condition anchor).
4. Token-length statistics — reported alongside the parity check, for visibility.
5. Byte-identical rerun — rebuilding the corpus twice (and against disk) produces
   byte-identical output.
6. Split integrity — data/splits.json holds exactly the 30 expected pair_ids,
   20 train / 10 held_out, 4/2 per domain, and re-deriving the blake2b hash
   procedure reproduces the committed file byte-for-byte. Also cross-checks
   data/splits.json against data/prompts.jsonl: every split pair_id must appear
   in the prompt corpus, no stray pair_id may exist in the corpus beyond the 30
   in the split, and the train/held_out counts must be 20/10.

This script does not test H1, H2, or H3. It is a data-quality gate: a failure here
means the corpus is invalid and any downstream probe result would be
uninterpretable, not that a hypothesis has been confirmed or disconfirmed.

Exit code 0 iff checks 1, 2, 3, 5, and 6 all pass. Rows with gold_answer=None are
reported by count, not treated as a failure.

Note: the data/splits.json <-> data/prompts.jsonl cross-check (point 6 above) was
deliberately deferred until the seed file's pair_id space matched the split (the
pilot corpus used coding_001/writing_001 while the split already covered the
30-task space of code_*/text_*). Now that data/controlled_tasks_seed.json has
been expanded to the full 30 tasks, the cross-check is implemented below.
"""

import hashlib
import json
import re
import statistics
import sys

from build_prompts import (
    CONDITIONS,
    LANGUAGES,
    OUTPUT_PATH,
    REPO_ROOT,
    TASKS_PATH,
    TOKENIZER_NAME,
    build_rows,
    load_tasks,
    load_wrappers,
    serialize_rows,
)
from transformers import AutoTokenizer

DEPLOYMENT_CONDITIONS = {"deploy_plain", "deploy_framed"}

# --- Locked 30-task train/held-out split (docs/decision_log.md) ---------------
# The split is fixed by hashing pair_ids BEFORE any task text is authored, so
# item difficulty / test-likeness cannot be steered into a fold. blake2b (not
# Python's built-in hash(), which is per-process salted) so the assignment is
# reproducible across processes, Python versions, and platforms, and so a human
# can recompute any single row by hand from hash_hex (CLAUDE.md rule 8).
SPLITS_PATH = REPO_ROOT / "data" / "splits.json"
SPLIT_DOMAINS = ["arith", "code", "logic", "fact", "text"]
TASKS_PER_DOMAIN = 6
TRAIN_PER_DOMAIN = 4
HELD_OUT_PER_DOMAIN = 2
SPLIT_FOLDS = ("train", "held_out")

# Per-pair, per-language length-parity gate. Anchored to deploy_plain (the null
# condition) rather than the four-condition mean, and checked per pair rather
# than on aggregate condition means — a wrapper that's fine on average could
# still blow the budget on one specific task, and per-pair is what
# docs/dataset_protocol.md actually specifies ("within each pair, preserve...
# approximate token length").
LENGTH_TOLERANCE = 0.15

# Per-language banned-word lists for the eval-marker leakage check. Only "en" is
# populated today (the corpus is English-only). Add "zh"/"ar"/"yo" entries here
# — each a native list, not a machine translation of the English one, since a
# translated list is produced by the same process being audited and cannot
# catch a mistranslation of its own terms — before build_prompts.py emits any
# row in that language. check_leakage() below fails loudly (a reported failure,
# not a silent skip) for any language it encounters with no entry here, so a
# missing list cannot pass the audit by omission.
BANNED_WORDS_BY_LANGUAGE = {
    "en": [
        "test", "tested", "testing",
        "evaluate", "evaluated", "evaluation", "evaluating",
        "benchmark", "benchmarked",
        "grade", "graded", "grading",
        "score", "scored", "scoring",
        "assess", "assessment",
        "exam", "examination",
        "quiz",
    ],
    # Yoruba. Checked by hand against all 30 task_yo strings in
    # data/controlled_tasks_seed.json for false-positive collisions (e.g. "àmì",
    # used legitimately in fact_002/fact_006 for "chemical symbol", was
    # deliberately left OFF this list). Each entry appears with and without
    # tone/underdot diacritics, so a degraded (no-diacritics) rendering of the
    # same word is still caught. The four loanword-transliteration entries
    # (bẹ́ńṣímáàkì/benchmark, kúísì/quiz, gírédì/grade, síkọ̀/score) are a
    # best-effort AI guess at how those borrowed terms would be spelled, not a
    # confirmed native-speaker spelling -- revisit once real Yorùbá wrapper text
    # exists and we see what spelling it actually uses (see report to researcher).
    # Machine-translated Chinese terms; structurally audited, not native-speaker certified.
    "zh": [
        "测试", "测验", "考试", "考核", "评估",
        "基准", "benchmark", "quiz", "grade", "score",
    ],
    # Machine-translated Arabic terms; structurally audited, not native-speaker certified.
    "ar": [
        "اختبار", "اختبارات",
        "امتحان", "امتحانات",
        "تقييم", "تقويم",
        "benchmark", "quiz", "grade", "score",
    ],
    "yo": [
        "ìdánwò", "idanwo",
        "àdánwò", "adanwo",
        "àyẹ̀wò", "ayewo",
        "ṣe àyẹ̀wò", "se ayewo",
        "bẹ́ńṣímáàkì", "bensimaki",
        "kúísì", "kuisi",
        "gírédì", "giredi",
        "síkọ̀", "siko",
    ],
}
BANNED_PATTERNS = {
    language: re.compile(
        r"\b(" + "|".join(re.escape(w) for w in words) + r")\b",
        re.IGNORECASE,
    )
    for language, words in BANNED_WORDS_BY_LANGUAGE.items()
}


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
    languages_missing_list = {}
    for row in rows:
        if row["condition"] not in DEPLOYMENT_CONDITIONS:
            continue
        language = row["language"]
        pattern = BANNED_PATTERNS.get(language)
        if pattern is None:
            languages_missing_list.setdefault(language, set()).add(row["pair_id"])
            continue
        match = pattern.search(row["user_text"])
        if match:
            failures.append(
                f"{row['pair_id']}/{language}/{row['condition']}: "
                f"banned word {match.group(0)!r} found in user_text: {row['user_text']!r}"
            )

    # A missing list is reported as its own hard failure — never treated as "no
    # match found" — so a new language cannot pass this check by omission.
    for language, pair_ids in sorted(languages_missing_list.items()):
        failures.append(
            f"language {language!r}: no entry in BANNED_WORDS_BY_LANGUAGE, cannot check "
            f"leakage for {len(pair_ids)} pair_id(s) ({sorted(pair_ids)}) — add a native "
            f"banned-word list for this language before auditing it"
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


def split_hash(pair_id: str) -> tuple[int, str]:
    """Return (sort_key, hash_hex) for one pair_id under the locked procedure."""
    digest = hashlib.blake2b(pair_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big"), digest.hex()


def derive_split() -> list[dict]:
    """Re-derive the 30-task split from pair_ids alone.

    Within each domain: sort the 6 ids ascending by their blake2b integer, first
    4 -> train, last 2 -> held_out. Rows are emitted in (domain order, id order)
    rather than hash order, so the file is readable; the fold assignment itself
    comes only from the hash ranking.
    """
    rows = []
    for domain in SPLIT_DOMAINS:
        pair_ids = [f"{domain}_{i:03d}" for i in range(1, TASKS_PER_DOMAIN + 1)]
        ranked = sorted(pair_ids, key=lambda pid: split_hash(pid)[0])
        fold_of = {
            pid: ("train" if rank < TRAIN_PER_DOMAIN else "held_out")
            for rank, pid in enumerate(ranked)
        }
        for pair_id in pair_ids:
            rows.append(
                {
                    "pair_id": pair_id,
                    "domain": domain,
                    "fold": fold_of[pair_id],
                    "hash_hex": split_hash(pair_id)[1],
                }
            )
    return rows


def serialize_split(rows: list[dict]) -> str:
    """Canonical on-disk form of data/splits.json (key order fixed by derive_split)."""
    return json.dumps(rows, indent=2, ensure_ascii=False) + "\n"


def check_split_integrity(prompt_pair_ids: set[str]) -> list[str]:
    failures = []

    if not SPLITS_PATH.exists():
        return [f"{SPLITS_PATH} does not exist"]

    disk_text = SPLITS_PATH.read_text(encoding="utf-8")
    try:
        disk_rows = json.loads(disk_text)
    except json.JSONDecodeError as exc:
        return [f"{SPLITS_PATH} is not valid JSON: {exc}"]

    if not isinstance(disk_rows, list):
        return [f"{SPLITS_PATH} must be a JSON list, got {type(disk_rows).__name__}"]

    expected_total = len(SPLIT_DOMAINS) * TASKS_PER_DOMAIN
    if len(disk_rows) != expected_total:
        failures.append(f"expected {expected_total} rows, found {len(disk_rows)}")

    # Structural checks run against the committed file, not the re-derivation, so
    # a hand-edited file fails here rather than only on the byte comparison.
    required_keys = {"pair_id", "domain", "fold", "hash_hex"}
    seen_ids = []
    for i, row in enumerate(disk_rows):
        if not isinstance(row, dict) or set(row) != required_keys:
            failures.append(f"row {i}: keys {sorted(row) if isinstance(row, dict) else row!r} != {sorted(required_keys)}")
            continue
        seen_ids.append(row["pair_id"])
        if row["fold"] not in SPLIT_FOLDS:
            failures.append(f"{row['pair_id']}: fold {row['fold']!r} not in {SPLIT_FOLDS}")
        if row["domain"] not in SPLIT_DOMAINS:
            failures.append(f"{row['pair_id']}: domain {row['domain']!r} not in {SPLIT_DOMAINS}")

    duplicates = sorted({pid for pid in seen_ids if seen_ids.count(pid) > 1})
    if duplicates:
        failures.append(f"duplicate pair_ids: {duplicates}")

    expected_ids = {row["pair_id"] for row in derive_split()}
    missing = sorted(expected_ids - set(seen_ids))
    unexpected = sorted(set(seen_ids) - expected_ids)
    if missing:
        failures.append(f"missing pair_ids: {missing}")
    if unexpected:
        failures.append(f"unexpected pair_ids: {unexpected}")

    fold_counts = {fold: sum(1 for r in disk_rows if r.get("fold") == fold) for fold in SPLIT_FOLDS}
    expected_train = len(SPLIT_DOMAINS) * TRAIN_PER_DOMAIN
    expected_held_out = len(SPLIT_DOMAINS) * HELD_OUT_PER_DOMAIN
    if fold_counts["train"] != expected_train:
        failures.append(f"expected {expected_train} train rows, found {fold_counts['train']}")
    if fold_counts["held_out"] != expected_held_out:
        failures.append(f"expected {expected_held_out} held_out rows, found {fold_counts['held_out']}")

    for domain in SPLIT_DOMAINS:
        domain_rows = [r for r in disk_rows if r.get("domain") == domain]
        n_train = sum(1 for r in domain_rows if r.get("fold") == "train")
        n_held_out = sum(1 for r in domain_rows if r.get("fold") == "held_out")
        if (n_train, n_held_out) != (TRAIN_PER_DOMAIN, HELD_OUT_PER_DOMAIN):
            failures.append(
                f"{domain}: expected {TRAIN_PER_DOMAIN} train / {HELD_OUT_PER_DOMAIN} held_out, "
                f"found {n_train} train / {n_held_out} held_out"
            )

    # Byte-for-byte re-derivation. This detects hand-editing, ID drift, and any
    # divergence between the committed file and the procedure that produced it.
    # It does not, and cannot, verify that the procedure itself is the intended
    # one — that is what the locked spec in docs/decision_log.md is for.
    derived_text = serialize_split(derive_split())
    if derived_text != disk_text:
        failures.append(
            f"re-derived split is NOT byte-identical to {SPLITS_PATH} "
            f"(derived {len(derived_text)} bytes vs disk {len(disk_text)} bytes)"
        )

    # Cross-check against the actual prompt corpus (data/prompts.jsonl), not just
    # the split file's own internal structure. splits.json can be internally
    # consistent (30 correct ids, correct 20/10 fold counts) while still having
    # drifted from the seed file the corpus was actually built from — e.g. a
    # typo'd pair_id in data/controlled_tasks_seed.json, or a task dropped from
    # the seed but left in the split. Both directions are checked so neither a
    # missing task nor a stray one can pass silently.
    split_ids = set(seen_ids)
    missing_from_corpus = sorted(split_ids - prompt_pair_ids)
    if missing_from_corpus:
        failures.append(
            f"pair_id(s) in {SPLITS_PATH.name} but missing from {OUTPUT_PATH.name}: "
            f"{missing_from_corpus}"
        )
    stray_in_corpus = sorted(prompt_pair_ids - split_ids)
    if stray_in_corpus:
        failures.append(
            f"pair_id(s) in {OUTPUT_PATH.name} but not present in {SPLITS_PATH.name}: "
            f"{stray_in_corpus}"
        )

    return failures


def report_split_summary() -> None:
    if not SPLITS_PATH.exists():
        return
    try:
        rows = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    if not isinstance(rows, list):
        return
    print(f"\nLocked split ({SPLITS_PATH.name}), fold assignment by domain:")
    for domain in SPLIT_DOMAINS:
        train = sorted(r["pair_id"] for r in rows if r.get("domain") == domain and r.get("fold") == "train")
        held = sorted(r["pair_id"] for r in rows if r.get("domain") == domain and r.get("fold") == "held_out")
        print(f"  {domain:>6}: train={train} held_out={held}")


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
    prompt_pair_ids = {row["pair_id"] for row in disk_rows}
    split_failures = check_split_integrity(prompt_pair_ids)
    report_token_length_stats(disk_rows)
    report_null_gold_answers(disk_rows)
    report_split_summary()
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

    print(
        f"\n=== Split integrity ({len(SPLIT_DOMAINS) * TASKS_PER_DOMAIN} ids, "
        f"{len(SPLIT_DOMAINS) * TRAIN_PER_DOMAIN} train / "
        f"{len(SPLIT_DOMAINS) * HELD_OUT_PER_DOMAIN} held_out, "
        f"{TRAIN_PER_DOMAIN}/{HELD_OUT_PER_DOMAIN} per domain, blake2b re-derived, "
        f"cross-checked against {OUTPUT_PATH.name}) ==="
    )
    if split_failures:
        for f in split_failures:
            print(f"  FAIL: {f}")
    else:
        print("  OK")

    print("\n=== Byte-identical rerun ===")
    if rerun_failures:
        for f in rerun_failures:
            print(f"  FAIL: {f}")
    else:
        print("  OK")

    all_failures = (
        coverage_failures + leakage_failures + parity_failures + split_failures + rerun_failures
    )
    if all_failures:
        print(f"\nAUDIT FAILED: {len(all_failures)} failure(s)")
        return 1

    print("\nAUDIT PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
