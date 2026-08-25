# H1-EN Stage A Scaffold

Stage A of the H1 English (`h1_en`) experiment. This stage creates **empty, machine-checkable
skeletons only**: identifiers, structural metadata, split quotas, and schemas. No semantic
content, no prompt text, no provenance claims, no model calls.

## Scope and containment

Everything added by Stage A lives under `experiments/h1_en/`:

```
experiments/h1_en/
  README.md
  config/config.json
  schema/item.schema.json
  schema/fpr.schema.json
  data/items.v1.jsonl
  data/fpr.v1.jsonl
```

Stage A does not create, modify, move, or write references into any pre-existing
top-level `data/`, `src/`, wrapper, or prompt directories. If later stages need shared
code, that integration is proposed and reviewed separately; this scaffold is self-contained.

## Record counts (invariants)

- `data/items.v1.jsonl`: exactly **70** semantic-item skeleton records.
- `data/fpr.v1.jsonl`: exactly **40** FPR (frame-prior recognition) skeleton records.

Check locally:

```
wc -l experiments/h1_en/data/items.v1.jsonl   # expect 70
wc -l experiments/h1_en/data/fpr.v1.jsonl     # expect 40
jq -r '.item_id' experiments/h1_en/data/items.v1.jsonl | sort | uniq -d   # expect empty
jq -r '.fpr_id'  experiments/h1_en/data/fpr.v1.jsonl   | sort | uniq -d   # expect empty
```

## Item structure

70 items = 7 domains x 2 families per domain x 5 paraphrases per family.

- `item_id`: `h1_en-item-f<NN>-p<NN>`
- `family_id`: `f01` ... `f14` (a family is one paraphrase cluster of a single semantic nucleus)
- `domain_id`: `dom01` ... `dom07`; families map as `f01,f02 -> dom01`, `f03,f04 -> dom02`,
  `f05,f06 -> dom03`, `f07,f08 -> dom04`, `f09,f10 -> dom05`, `f11,f12 -> dom06`,
  `f13,f14 -> dom07`
- `paraphrase_index`: 1..5 within a family

Domain and family identifiers are structural placeholders. They deliberately carry no topic
labels, because assigning topics would be authored content and belongs to Stage B.

### Frame-neutral content only

Item records store the semantic nucleus and its answer key, nothing about framing. In Stage A
all of these are null placeholders:

```
core.nucleus         = null
core.mcq_stem        = null
core.options         = [null, null, null, null]
core.answer_index    = null
core.answer_free_form = null
```

Eval/deploy prompt text is **never** stored in item records, in any stage. Item schemas set
`additionalProperties: false` on the record and on `core`, so a stray `eval_prompt` or
`deploy_prompt` key fails validation.

## Rendered variants (conceptual, not stored per item)

There are four rendered variants, defined once as the cross product

```
frame  in {eval, deploy}
format in {prose, mcq}
```

giving `eval_prose`, `eval_mcq`, `deploy_prose`, `deploy_mcq`. These are declared in
`config/config.json` under `rendered_variants` and are produced at render time by combining a
frame template with one frame-neutral item. Each item therefore yields four rendered stimuli
without duplicating text, and paraphrase-level and frame-level variance stay separable.
Template bodies are not authored in Stage A; `template_ref` is null for every variant.

## FPR structure

40 FPR records = 2 expected labels x 2 formats x 10 records per cell.

- `fpr_id`: `h1_en-fpr-<label>-<format>-<NN>`
- `expected_label`: `eval` or `deploy` (populated in Stage A; it is a design constant, not content)
- `format`: `prose` or `mcq`
- `system` and `user` are **separate** fields, both null in Stage A. They stay separate so that
  system-position and user-position framing cues can be manipulated and ablated independently.
- `domain_id` is null in Stage A: FPR probes are not yet tied to item domains. Stage B assigns
  domains (or documents deliberate domain-independence) before any split assignment.

## Splits

Target quotas over the 70 items: **train=35, val=15, test=20**.

`split` is null in every Stage A record, for items and FPR alike. Assignment happens in Stage B.

Assignment unit is the **family**, not the item, because five paraphrases of one nucleus would
otherwise leak across splits. Family quotas that realize the item quotas: train=7, val=3,
test=4 families (7x5=35, 3x5=15, 4x5=20). A domain has two families and may therefore span
splits; if stricter domain-level isolation is required, Stage B must revise the structural plan
rather than silently reinterpret these quotas. FPR split assignment is deferred to Stage B and
must be consistent with the item assignment so that no FPR probe shares a family or domain
boundary violation with the item splits.

## Nulls in Stage A

Populated: `schema_version`, all identifiers, `language`, `format`, `expected_label`,
`paraphrase_index`, item `domain_id`, `family_id`, and `audit` bookkeeping.

Null: every content field, every provenance field, `split`, and FPR `domain_id`.

`provenance` is present as an object whose keys are all null, so Stage C can require each key to
be non-null without a schema shape change. No sources, licenses, dates, or authors are invented.

## Validation gates

Stage A (must pass now):
1. Line counts are 70 and 40.
2. Every record validates against its schema.
3. Identifiers are unique and match the documented patterns.
4. All content fields, all provenance fields, and all `split` fields are null.
5. Structural counts hold: 14 families x 5 paraphrases; 7 domains x 2 families;
   4 FPR cells x 10 records.

Stage B (assignment):
6. `split` populated for all 110 records, respecting family-level assignment and the
   35/15/20 item quotas.
7. FPR `domain_id` populated or documented as intentionally domain-free.

Stage C (content, blocking):
8. Apply `$defs.stage_c_item_requirements` and `$defs.stage_c_fpr_requirements` from the
   schemas. These require non-null `core.nucleus`, `core.mcq_stem`, four non-null distinct
   `core.options`, integer `core.answer_index` in 0..3, non-null `core.answer_free_form`,
   non-null FPR `system` and `user`, non-null `split`, and non-null provenance fields.
9. A record that still holds Stage A nulls must fail Stage C validation. Passing Stage A is
   never sufficient for analysis.

## Explicit non-goals for Stage A

- No authored items, stems, options, answers, or probe text.
- No invented or placeholder-that-looks-real provenance.
- No model calls, no API clients, no inference or generation code.
- No activation capture, probe training, or steering code.
- No edits to files outside `experiments/h1_en/`.
