# H1 and H2 Results Summary

**Date:** 2026-08-25  
**Model:** Qwen/Qwen2.5-7B-Instruct  
**Status:** Preliminary results (English H1 complete, Chinese H2 complete, Arabic and Yorùbá pending)

---

## H1 (Localization): English Baseline

**Question:** Is eval-awareness linearly probe-able in Qwen2.5-7B-Instruct's residual stream for English prompts?

**Method:**
- 70 cores × 4 formats = 280 prompts (eval vs. deployment framing)
- Extracted activations at layers 12, 16, 20, 24
- Trained logistic regression probes per layer
- 35/15/20 train/val/test split (core-level)
- 40 deployment-only FPR prompts

**Results:**

| Metric | Value |
|---|---|
| Best layer | 16 |
| Test AUROC | 0.995 [0.985, 1.000] |
| Test balanced accuracy | 0.938 [0.886, 0.977] |
| FPR (40 deployment-only) | 0.000 (0/40) |
| Mean prob(eval) on FPR | 0.012 |
| Max prob(eval) on FPR | 0.389 |
| Random-vector control (test AUROC) | 0.622 |
| Majority-class baseline | 0.500 |

**Conclusion:** H1 is **strongly confirmed**. Eval-awareness is linearly represented in English with near-perfect separability and zero false positives.

---

## H2 (Transfer): English → Chinese

**Question:** Does an English-trained eval-awareness probe transfer zero-shot to Chinese?

**Method:**
- 70 native Chinese cores × 4 formats = 280 prompts (parallel to English)
- Extracted activations at layers 12, 16, 20, 24
- Applied English probe (layer 16) to Chinese at all four layers
- Trained native Chinese probes at all four layers
- Same 35/15/20 split pattern

**Results:**

| Metric | Value |
|---|---|
| Best English→Chinese transfer | Layer 12, AUROC = 0.7719 |
| Best native Chinese probe | Layer 12, AUROC = 0.9925 [0.977, 0.999] |
| Transfer gap | 0.2206 (22 percentage points) |
| English baseline (H1) | Layer 16, AUROC = 0.995 |

**Layer comparison:**

| Layer | English Probe → Chinese | Native Chinese Probe |
|---|---|---|
| 12 | 0.7719 | **0.9925** |
| 16 | 0.7644 | 0.9663 |
| 20 | 0.7200 | 0.9288 |
| 24 | 0.6234 | 0.9381 |

**Conclusion:** H2 is **partially supported**:

1. **Strong native representation:** Chinese eval-awareness is linearly probe-able at AUROC = 0.9925, nearly matching English (0.995).

2. **Significant transfer degradation:** English probe achieves only 0.7719 AUROC on Chinese—a 22-point drop. This is substantial but not catastrophic failure.

3. **Layer shift:** Both languages are best at **layer 12** (Chinese) vs. **layer 16** (English), suggesting geometry differences.

4. **Partial generalization:** 0.77 AUROC is well above chance (0.5), meaning the English direction captures some cross-lingual signal.

---

## Interpretation

These results suggest:

- Eval-awareness is **not a single language-agnostic feature** but is at least partly **entangled with language-specific representation geometry**.
- The English direction **partially transfers** but misses language-specific nuances that native probes capture.
- The 22-point gap is large enough to matter for practical monitoring (e.g., an English-trained eval detector would miss ~22% of Chinese eval prompts compared to a native detector).

---

## Next Steps

1. **Arabic (medium resource):** Repeat H2 analysis. Expect similar or larger transfer gap.
2. **Yorùbá (low resource):** Stress test. Expect largest transfer gap if H2 holds.
3. **Cross-lingual steering (H3):** Test whether English-derived steering vector suppresses eval-aware posture in Chinese/Arabic/Yorùbá.

---

## Files

- `outputs/h1_english_activations.pt` — English activations (4 layers)
- `outputs/h1_english_probe_results.json` — English probe results
- `outputs/h2_chinese_activations_4layers.pt` — Chinese activations (4 layers)
- `outputs/h2_chinese_transfer_results.json` — Chinese transfer analysis
- `experiments/h1_clean_pilot/` — English dataset and scripts
- `experiments/h2_chinese/` — Chinese dataset and scripts
