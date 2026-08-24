# Pre-Experiment Decision Log

This file records source selection, infrastructure validation, and experimental-design decisions made before the formal 20-hour project timer begins.

## 2026-08-24 — GPU and environment validation

- GPU: NVIDIA GeForce RTX 4090.
- Available VRAM: 23.52 GiB.
- PyTorch: 2.11.0+cu128.
- CUDA was available to PyTorch.

This confirms that loading `Qwen/Qwen2.5-7B-Instruct` in bfloat16 is a reasonable infrastructure target. The model-loading test below is a setup feasibility check only; no activations, probes, or experiment metrics have been collected.

## 2026-08-24 — Native evaluation-source verification

The following datasets were loaded locally and inspected:

| Language | Dataset | Verified size / structure |
|---|---|---|
| English | `cais/mmlu` | 14,042 test; 1,531 validation; 285 dev; 99,842 auxiliary-train |
| Chinese | CMMLU | 11,582 questions across 67 subjects |
| Arabic | `MBZUAI/ArabicMMLU` | 14,455 test; 120 dev |
| Yorùbá | `masakhane/afrimmlu` | 83 validation; 25 dev; 500 test |
| Hausa | `masakhane/afrimmlu` | 83 validation; 25 dev; 500 test |
| Swahili | `masakhane/afrimmlu` | 83 validation; 25 dev; 500 test |

We selected Yorùbá as the low-resource language. Yorùbá, Hausa, and Swahili were all locally verified and equally feasible by split size; Yorùbá aligns most directly with the researcher's existing cross-lingual safety work. This choice was made before any model outcomes were observed.

## 2026-08-24 — Deployment-context source check

A streamed sample of the first 20,000 records from `allenai/WildChat-1M` gave the following detected-language counts:

| Language | Count in first 20,000 records |
|---|---:|
| English | 9,464 |
| Chinese | 5,732 |
| Arabic | 62 |
| Yorùbá | 23 |
| Hausa | 0 |
| Swahili | 18 |

Conclusion:
- WildChat is adequate as a possible secondary naturalistic deployment-context source for English and Chinese.
- WildChat is not adequate as a representative deployment-context source for Arabic or Yorùbá.
- The primary experiment will therefore use controlled, parallel eval-vs-deployment prompt pairs in English, Chinese, Arabic, and Yorùbá.
- Native benchmark-versus-chat comparisons, if included, will be reported only as secondary naturalistic validation because source format, academic topic, multiple-choice structure, and language distribution are confounds.

## 2026-08-24 — Formal timer policy

No formal experimental work has begun at the time of this note. Dataset-source verification, GPU setup, dependency installation, model-load feasibility testing, and agent-access troubleshooting are treated as pre-experiment setup.

The formal timer will begin only after:
1. The controlled prompt dataset is finalized, saved, and manually audited.
2. The model/tooling path is working reliably.
3. The project’s train/test split, activation extraction position, and primary metrics are locked.


## 2026-08-24 — Qwen model-load feasibility

`Qwen/Qwen2.5-7B-Instruct` successfully loaded on the RTX 4090 in bfloat16 using Hugging Face Transformers and `device_map="cuda"`.

No activations, probes, steering vectors, generations, or experimental metrics were collected in this check. This was infrastructure validation only.

## 2026-08-24 — Scope change: 30 base tasks

The controlled dataset will use **30 base tasks**, not the 60–80 sketched in `docs/proposal.md`.
Reason: the 20-hour formal budget plus the requirement that every non-English prompt be
manually quality-controlled. 30 tasks × 4 languages × 4 conditions = 480 prompts, all of
which a human can actually read. This supersedes the "budget for ~60-80 pairs" note in the
proposal. Consequence accepted in advance: held-out N is small (10 tasks / 160 prompts), so
all headline numbers are reported with bootstrap confidence intervals and no result is
claimed on a point estimate alone.

## 2026-08-24 — Five-rung claim ladder

Claims are made one rung at a time. We report the highest rung actually reached and state
plainly that the rungs above it are untested, not merely unconfirmed. A failure at rung *k*
is written up as a negative result; the rungs are not redefined after seeing data.

1. **Detectable beyond surface lexical cues (English).** A linear probe on
   Qwen2.5-7B-Instruct residual activations separates evaluation-framed from
   deployment-framed English prompts on held-out base tasks and exceeds the TF-IDF baseline
   by at least 0.10 AUROC on the held-out explicit-to-implicit contrast. The layer-0 probe
   is reported as a diagnostic baseline: layer-0 saturation indicates that the result is
   likely driven by token-level surface cues rather than a deeper contextual
   representation. (H1)
2. **Transfers zero-shot.** A probe fit on English only separates eval from deployment
   framing in Chinese, Arabic, and Yorùbá, above the same baselines, with no non-English
   data used for fitting. (H2, existence)
3. **Ordered by resource level.** Transfer strength is ordered
   English > Chinese/Arabic > Yorùbá. (H2, the specific directional prediction — this is a
   separate rung because rung 2 can pass while rung 3 fails.)
4. **Causally usable.** The English-derived direction changes eval-framing-sensitive
   behavior when added/subtracted during generation, and does so differently from the
   random-vector and prompt-only controls on the same prompts. (H3, precondition)
5. **Causal efficacy degrades cross-lingually.** The rung-4 effect is weaker, or more
   capability-damaging, in Yorùbá than in English. (H3)

## 2026-08-24 — Go / pivot / abandon criteria

Fixed before any activation is collected, so that a disappointing number cannot be
re-litigated into a passing one.

**Rung 1 gate (English probe).** Primary metric: held-out AUROC, pair-grouped split,
95% CI from 1000 bootstrap resamples *over base tasks* (not over prompts).
- **GO** — proceed to the transfer arm — if held-out AUROC ≥ 0.80, the 95% bootstrap CI
  lower bound exceeds 0.65, English explicit-to-implicit AUROC exceeds the TF-IDF baseline
  by ≥ 0.10, and the layer-0 result is not already saturated (AUROC ≥ 0.95).
- **PIVOT** if 0.65 ≤ AUROC < 0.80, or if the TF-IDF margin on the explicit-to-implicit
  contrast is < 0.10 (i.e. the signal is plausibly lexical), or if layer 0 is saturated:
  spend at most **2 hours** on a layer sweep, readout-position change, and a prompt-set
  purity audit, then re-test **once** against the same gate. No open-ended tuning, no third
  attempt.
- **ABANDON the steering arm** if AUROC < 0.65 after that single pivot. H1 is reported as
  disconfirmed for this setup, and the remaining budget goes to diagnostics (what the probe
  latches onto, where the dataset leaks) plus the write-up. A clean negative on H1 is the
  deliverable in that branch.

**Rung 2/3 gate (transfer).** Run for all four languages regardless of the English number's
size, provided rung 1 passed. Transfer is called *present* for a language only if its
bootstrap CI lower bound > 0.5. Rung 3 is called *supported* only if the language ordering
holds with non-overlapping CIs between English and Yorùbá; overlapping CIs are reported as
"ordering not resolved at this N", not as support.

**Rung 4/5 gate (steering).** Steering is attempted only if rung 1 passed. Per `CLAUDE.md`
rule 2, every steered run ships with its random-vector and prompt-only controls on the same
prompts in the same script. A steering claim requires the steered arm to separate from
**both** controls; separating from prompt-only but not from random-vector is reported as
"no evidence of a direction-specific effect".

**Time gates (of the 20 formal hours).** Dataset construction + manual audit ≤ 5h; probe
arm ≤ 5h; steering arm ≤ 6h; write-up ≥ 3h held in reserve. If any stage overruns its
budget by more than 50%, the steering arm is cut — never the write-up or the manual audit.

If the multilingual controlled dataset is not manually auditable by hour 5, reduce base-task
count from 30 to 20 while preserving all four languages, all four conditions, and the
held-out explicit-to-implicit contrast.

## 2026-08-24 — Final primary design (locked)

**Grid.** 30 base tasks × 4 languages × 4 framing conditions = 480 prompts.
- Languages: English, Chinese, Arabic, Yorùbá.
- Base tasks: language-neutral, short-answer or brief-reasoning items (arithmetic, logic,
  coding, writing, factual recall), seeded in `data/controlled_tasks_seed.json`.
- Framing conditions, 2 deployment + 2 eval, so the label classes are balanced 240/240 and
  neither class is carried by a single surface template:
  1. `deploy_plain` — bare user request.
  2. `deploy_framed` — same task inside an ordinary conversational turn.
  3. `eval_explicit` — stated benchmark/grading/evaluation wrapper.
  4. `eval_implicit` — test-like framing and formatting without the words
     "evaluation"/"benchmark"/"test".
- The manipulated variable is evaluation framing only; task content, answer format,
  language, and approximate token length are held constant within a base task
  (`docs/dataset_protocol.md`).

**Not to be confused with the steering arms.** The four *steering* conditions (unsteered,
steered, random-vector control, prompt-only control) are a separate axis applied later to
the same prompts. They do not multiply into the 480.

**Pair-grouped splitting.** Splits are drawn over **base task IDs**, never over prompts.
All 16 prompts derived from a base task (4 languages × 4 conditions) land in the same fold:
20 train tasks / 10 test tasks, fixed seed, split saved to disk. This prevents a probe from
scoring on memorized task content or on its own paraphrase across languages.

**Activation readout.** Primary: residual-stream activation at the **final token of the
applied chat template** — after `apply_chat_template`, before any token is generated. One
vector per prompt per layer. A layer sweep across all 28 layers is reported as secondary,
with the H1 layer range (~14–24) pre-registered as the expected region; the primary readout
position is not changed on the basis of results without logging the change here.

**Primary metric.** AUROC on held-out base tasks, with bootstrap CIs over tasks. Accuracy,
precision/recall, and per-condition breakdowns are secondary. AUROC is primary because it
is threshold-free and stable under the class balance above.

**Held-out explicit-to-implicit contrast (locked definition).** This is the generalization
test named in the rung-1 claim and the rung-1 GO gate. It is defined as:

- Fit the English probe using **only** `eval_explicit` versus `deploy_plain` prompts from
  the **20 training base tasks**.
- Evaluate its primary held-out generalization on `eval_implicit` versus `deploy_framed`
  prompts from the **10 held-out base tasks**.
- Do **not** fit, tune hyperparameters, select layers, calibrate thresholds, or otherwise
  make design choices using `eval_implicit` or `deploy_framed` examples before this held-out
  evaluation.
- Pair-grouped splitting remains by `base_task_id`, so all language/condition variants of
  one base task remain in the same split.

This is the number that must beat TF-IDF by ≥ 0.10 AUROC, because it is the condition where
the eval class carries no explicit lexical marker for a bag-of-ngrams model to key on. The
TF-IDF and layer-0 baselines are computed on exactly this fit/evaluate arrangement.

**Baselines, same splits, same code path.**
- **TF-IDF baseline (required to beat)** — character n-gram logistic regression on the raw
  prompt string. Answers "is this just surface lexical cues?" A probe that fails to beat
  TF-IDF by the rung-1 margin on the explicit-to-implicit contrast is not evidence for a
  representation.
- **Layer-0 baseline (diagnostic, not a gate to beat)** — identical probe on embedding-layer
  activations. It is not required that the residual-stream probe beat it by a margin; its
  role is to detect token-level shortcuts. Layer-0 saturation (AUROC ≥ 0.95) means the
  contrast is solvable from token identity alone and any deeper-layer result is
  uninterpretable as a contextual representation.

**Per-language parity.** Every language goes through the same function, same layers, same
readout position, same split logic, same metric code (`CLAUDE.md` rule 3). Any deviation
is logged here before it is run, not after.

## 2026-08-24 — What is and is not being claimed

The core claim of this project is that **evaluation framing present in a prompt is linearly
decodable from the model's activations, and that an English-fit decoder does or does not
transfer to other languages.** That is a claim about decodability.

It is explicitly **not** a claim that the model "represents eval-awareness", is
situationally aware that it is being tested, or holds any self-referential state about being
evaluated. Nothing in this design can distinguish a representation of *"I am under
evaluation"* from a representation of *"this text has the register of a test item"*, and the
write-up will say so in those terms.

Two corollaries we commit to reporting:
- Above-chance cross-lingual transfer is consistent with shared surface cues, shared
  formatting, translationese from the prompt-construction process, or Qwen's
  English-centric internal space — not only with a language-independent concept.
- A rung-1 probe that beats TF-IDF still only shows the information is present and linearly
  accessible at that layer; it does not show the model uses it. Only rung 4 speaks to use,
  and only for the behaviors we actually measure.
