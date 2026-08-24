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
