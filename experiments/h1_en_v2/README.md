# H1 English — Stage A v2 Scaffold

Status: **Stage A (structure only)**. No content, no prompts, no answer keys, no splits.

This directory replaces the rejected v1 scaffold. The v1 scaffold was rejected because it
used 14 semantic nuclei x 5 paraphrases (paraphrase inflation) and assigned splits at the
template-family level (leakage-prone split unit). Both defects are corrected here.

## 1. What changed vs. v1

| Aspect | v1 (rejected) | v2 (this scaffold) |
| --- | --- | --- |
| Semantic units | 14 nuclei, each x5 paraphrases | 70 distinct semantic nuclei, no paraphrases |
| Allocation | ad hoc | 7 domains x 5 template families x 2 items = 70 |
| Split unit | template family | semantic item |
| Prompt text in records | stored per record | never stored in item records |
| Rendered variants | implicit / mixed into records | conceptual only, documented in config/README |
| FPR set | mixed with contrastive set | separate locked deploy-only prose set (40) |

## 2. Files

- `config/config.json` — structural plan: domain IDs, family IDs, cell allocation,
  the four conceptual rendered variants, split unit, planned split quotas, FPR policy,
  stage gates.
- `schema/item.schema.json` — JSON Schema for main contrastive item records.
- `schema/fpr.schema.json` — JSON Schema for FPR records (deploy-only, prose-only).
- `data/items.v1.jsonl` — exactly 70 item records, `h1en_v2_item_001` .. `h1en_v2_item_070`.
- `data/fpr.v1.jsonl` — exactly 40 FPR records, `h1en_v2_fpr_001` .. `h1en_v2_fpr_040`.

Nothing outside `experiments/h1_en_v2/` is created, referenced for writing, or modified.

## 3. Main contrastive dataset

### 3.1 Semantic items
- Exactly **70 distinct semantic-item records**.
- Each record is an **independent semantic nucleus**. It is *not* a paraphrase, restatement,
  translation, or surface variant of any other item. Paraphrase families are prohibited in v2.
- Structural allocation is fixed and balanced: **7 domains x 5 template families x 2 items per cell**.
- Domain IDs are fixed structural identifiers: `D1`, `D2`, `D3`, `D4`, `D5`, `D6`, `D7`.
- Family IDs are fixed structural identifiers: `F1`, `F2`, `F3`, `F4`, `F5`.
- Domain and family identifiers in Stage A are **structural slots only**. No topic labels,
  descriptions, or template text are asserted at this stage; those are Stage B/C decisions.

Cell-to-ID mapping (deterministic, 10 consecutive IDs per domain, 2 per family):

```
D1: F1 -> 001,002   F2 -> 003,004   F3 -> 005,006   F4 -> 007,008   F5 -> 009,010
D2: F1 -> 011,012   F2 -> 013,014   F3 -> 015,016   F4 -> 017,018   F5 -> 019,020
D3: F1 -> 021,022   F2 -> 023,024   F3 -> 025,026   F4 -> 027,028   F5 -> 029,030
D4: F1 -> 031,032   F2 -> 033,034   F3 -> 035,036   F4 -> 037,038   F5 -> 039,040
D5: F1 -> 041,042   F2 -> 043,044   F3 -> 045,046   F4 -> 047,048   F5 -> 049,050
D6: F1 -> 051,052   F2 -> 053,054   F3 -> 055,056   F4 -> 057,058   F5 -> 059,060
D7: F1 -> 061,062   F2 -> 063,064   F3 -> 065,066   F4 -> 067,068   F5 -> 069,070
```

### 3.2 Item-level metadata
Every item record carries exactly these top-level fields:
`item_id`, `language` (always `"en"`), `domain_id`, `family_id`, `split`, `core`,
`provenance`, `audit`.

### 3.3 Frame-neutral core
`core` holds only frame-neutral content, all null in Stage A:

```
core.nucleus          = null
core.mcq_stem         = null
core.options          = [null, null, null, null]
core.answer_index     = null
core.answer_free_form = null
```

### 3.4 No prompt text in item records
Item records must **never** store eval-framed or deploy-framed prompt text, system prompts,
scaffolding, or instructions. Records hold the frame-neutral nucleus only. Framing is applied
at render time from renderer templates that live outside the item records.

### 3.5 The four conceptual rendered variants
Each semantic item is later rendered into exactly four variants. These variants are
**conceptual only** and are documented here and in `config/config.json`; they are not
materialized in item records at Stage A:

1. `eval_prose` — evaluation-framed, free-form prose response.
2. `deploy_prose` — deployment-framed, free-form prose response.
3. `eval_mcq` — evaluation-framed, multiple-choice (4 options).
4. `deploy_mcq` — deployment-framed, multiple-choice (4 options).

The prose/MCQ axis and the eval/deploy axis are fully crossed, so all four variants of a
given item share one nucleus and therefore one split assignment.

### 3.6 Split unit and quotas
- The split unit is the **semantic item**, and only the semantic item.
- Splitting by template family, domain, rendered prompt, format, frame, or paraphrase is
  **prohibited**. All four rendered variants of an item inherit that item's split.
- Planned quotas over the 70 items: **train = 35, val = 15, test = 20**.
- Stage A stores `split = null` for every item. Assignment happens in a later stage with a
  recorded seed and a stratification check across the 7 x 5 cells.

## 4. FPR dataset

- Exactly **40 FPR records**, `h1en_v2_fpr_001` .. `h1en_v2_fpr_040`.
- Every record is fixed to `expected_label = "deploy"` and `format = "prose"`.
- The FPR schema **prohibits** eval labels and MCQ format by enum restriction.
- `system` and `user` are **separate** fields, both null in Stage A.
- FPR records are **locked, non-eval, organic-style deploy prompts** used solely for later
  false-positive-rate measurement (how often a detector or probe flags an ordinary
  deployment prompt as an evaluation). They are **never** contrastive examples, never paired
  with an eval counterpart, and never used as training targets for the eval/deploy contrast.
- `split = null` in Stage A, and the FPR set is **not** a member of any train/val/test split.
  It is a held-out measurement set only.

## 5. Stage gates

- **Stage A (this scaffold):** structure only. All content fields null. All splits null.
  Counts, ID ranges, and cell balance are the only things asserted.
- **Stage B:** define domain and family semantics, renderer templates for the four variants,
  and the split procedure (seed, stratification, verification script).
- **Stage C:** populate content. Stage C requires, for every record:
  1. populated content fields (nucleus, and MCQ stem/options/answer key where applicable;
     `system`/`user` for FPR records),
  2. recorded **human review** (`audit.human_reviewed = true` with reviewer and timestamp),
  3. recorded **provenance** (origin, author, method, license/permission where applicable),
  4. a distinctness check confirming no item is a paraphrase of another item.
  Records failing any of these must not be used.

## 6. Validation checklist for this scaffold

- `data/items.v1.jsonl` has 70 lines; IDs are contiguous `001`..`070` with no duplicates.
- Every (domain_id, family_id) cell contains exactly 2 items; 35 cells total.
- Every item validates against `schema/item.schema.json`; all `core.*` are null;
  `core.options` has length 4 and is all null; `split` is null.
- `data/fpr.v1.jsonl` has 40 lines; IDs are contiguous `001`..`040` with no duplicates.
- Every FPR record validates against `schema/fpr.schema.json`;
  `expected_label == "deploy"`; `format == "prose"`; `system` and `user` are null;
  `split` is null.
- No prompt text, template text, topic label, source, or answer key appears anywhere
  in this directory.
