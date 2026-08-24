# Research Proposal: Cross-Lingual Generalization of Eval-Awareness Probing and Steering

(Full proposal — glossary, title, overview, hypotheses H1-H3, model selection rationale, dataset table, methodology steps 1-5, execution timeline, rigor checklist, agent system prompt, submission doc structure, risk notes.)

See conversation history / attached proposal document for complete text. Key points:

## Hypotheses
- **H1 (Localization)**: Eval-awareness is linearly probe-able in Qwen2.5-7B-Instruct's mid-to-late residual stream (~layers 14-24 of 28) for English prompts.
- **H2 (Transfer degrades with resource level)**: English-trained probe's zero-shot accuracy degrades English -> Chinese/Arabic (high-resource) -> low-resource language, tracking data availability.
- **H3 (Steering misalignment)**: English-derived steering vector suppresses eval-aware posture less effectively and/or damages capability more in the low-resource language than in English.

## Model
Qwen/Qwen2.5-7B-Instruct only. No second model this round (see proposal for reasoning — depth over breadth).

## Datasets (verified sources)
| Language | Eval-context source | Deployment-context source |
|---|---|---|
| English | MMLU (wrapped in eval framing) | `allenai/WildChat-1M`, filter language==English |
| Chinese | `haonan-li/cmmlu` (HF) | WildChat, filter language==Chinese |
| Arabic | `MBZUAI/ArabicMMLU` (HF) | WildChat, filter language==Arabic (fallback: LLM-generate + hand-check) |
| Low-resource (Yoruba/Hausa/Swahili) | IrokoBench AfriMMLU (masakhane, HF) | Translated WildChat English subset, hand-verified |

Note: AfriMMLU test set is only ~500 examples across 17 languages (~83/language dev split) — budget for ~60-80 pairs per low-resource language, not more.

## Rigor checklist
1. Mandatory baselines (unsteered, random-vector, prompt-only) in every language.
2. Static prompt-level conditional triggering, not token-by-token.
3. Dataset purity check for FPR sets (zero MCQ/benchmark markers in non-eval prompts).
4. Manual translation quality control for the low-resource language.
5. Report transfer-gap honestly whichever direction it goes.
6. Confound check: verify baseline task fluency in the low-resource language with no steering, to rule out "model just doesn't understand this language" as a trivial explanation.
7. Sample-size honesty in the write-up given small per-language N.
