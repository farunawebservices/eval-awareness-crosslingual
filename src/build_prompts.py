"""Build the tokenizer-only controlled prompt corpus.

Tokenizer-only infrastructure: loads the Qwen tokenizer to render chat-template
prompts and count tokens. Never loads model weights and never touches the GPU.

This script does not test H1, H2, or H3 (see docs/decision_log.md). It builds the
prompt corpus that later stages will use to test those hypotheses. English only
this pass; zh/ar/yo are added later by extending data/wrappers.json and
LANGUAGES (same code path, per CLAUDE.md rule 3 — per-language parity).
"""

import json
from pathlib import Path

from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
TASKS_PATH = REPO_ROOT / "data" / "controlled_tasks_seed.json"
WRAPPERS_PATH = REPO_ROOT / "data" / "wrappers.json"
OUTPUT_PATH = REPO_ROOT / "data" / "prompts.jsonl"

TOKENIZER_NAME = "Qwen/Qwen2.5-7B-Instruct"

CONDITIONS = ["deploy_plain", "deploy_framed", "eval_explicit", "eval_implicit"]
LANGUAGES = ["en"]  # zh, ar, yo added later — same build_rows() code path

# gold_answer now lives directly on each task in data/controlled_tasks_seed.json
# (None for non-scoreable open-ended "text" tasks, never a fabricated canonical
# answer — CLAUDE.md rule 6). Read from task["gold_answer"] below; a task missing
# the key raises KeyError rather than silently defaulting.


def load_tasks() -> list[dict]:
    with open(TASKS_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_wrappers() -> dict:
    with open(WRAPPERS_PATH, encoding="utf-8") as f:
        return json.load(f)


def render_user_text(task: dict, condition: str, language: str, wrappers: dict) -> tuple[str, str]:
    """Return (user_text, wrapper_id) for one task under one condition/language."""
    if language != "en":
        raise NotImplementedError(
            f"language={language!r} is not implemented yet; task text is only "
            f"available as task_en in {TASKS_PATH.name}"
        )
    entry = wrappers[condition][language]  # KeyError if condition/language missing — no silent fallback
    user_text = entry["template"].format(task=task["task_en"])
    return user_text, entry["wrapper_id"]


def render_chat_prompt(tokenizer, user_text: str) -> str:
    messages = [{"role": "user", "content": user_text}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def build_rows(tasks: list[dict], wrappers: dict, tokenizer) -> list[dict]:
    rows = []
    for language in LANGUAGES:
        for task in tasks:
            for condition in CONDITIONS:
                user_text, wrapper_id = render_user_text(task, condition, language, wrappers)
                rendered = render_chat_prompt(tokenizer, user_text)
                n_tokens = len(tokenizer.encode(rendered, add_special_tokens=False))
                rows.append(
                    {
                        "pair_id": task["pair_id"],
                        "domain": task["domain"],
                        "language": language,
                        "condition": condition,
                        "wrapper_id": wrapper_id,
                        "user_text": user_text,
                        "rendered": rendered,
                        "n_tokens": n_tokens,
                        "gold_answer": task["gold_answer"],
                    }
                )
    return sort_rows(rows)


def sort_rows(rows: list[dict]) -> list[dict]:
    condition_rank = {c: i for i, c in enumerate(CONDITIONS)}
    return sorted(rows, key=lambda r: (r["language"], r["pair_id"], condition_rank[r["condition"]]))


def serialize_rows(rows: list[dict]) -> str:
    lines = [json.dumps(row, sort_keys=True, ensure_ascii=False) for row in rows]
    return "\n".join(lines) + "\n"


def main() -> None:
    tasks = load_tasks()
    wrappers = load_wrappers()

    print(f"Loading tokenizer {TOKENIZER_NAME} (tokenizer only, no model weights, no GPU)...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    rows = build_rows(tasks, wrappers, tokenizer)
    text = serialize_rows(rows)

    OUTPUT_PATH.write_text(text, encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
