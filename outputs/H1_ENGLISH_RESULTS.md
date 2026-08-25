# H1 English Baseline Results

**Date:** 2026-08-25  
**Model:** Qwen/Qwen2.5-7B-Instruct  
**Layers tested:** 12, 16, 20, 24  
**Best layer:** 16

## Key Metrics

| Metric | Value |
|---|---|
| Test AUROC | 0.995 [0.985, 1.000] |
| Test balanced accuracy | 0.938 [0.886, 0.977] |
| FPR (40 deployment-only) | 0.000 (0/40) |
| Mean prob(eval) on FPR | 0.012 |
| Max prob(eval) on FPR | 0.389 |
| Random-vector control (test AUROC) | 0.622 |
| Majority-class baseline | 0.500 |

## Files

- `h1_english_activations.pt` — all 280 prompt activations at 4 layers
- `h1_english_activation_manifest.json` — extraction metadata
- `h1_english_probe_results.json` — probe training and evaluation results
- `h1_fpr_false_positives.jsonl` — top 10 highest-probability FPR prompts

## Human Verification Checklist

- [ ] Manually inspect 10 random train prompts and confirm eval/deploy labels are correct.
- [ ] Manually inspect 5 random val prompts and verify labels.
- [ ] Manually inspect 5 random test prompts and verify labels.
- [ ] Hand-read the 10 FPR false positives (highest prob) and confirm they are genuinely non-eval.
- [ ] Recompute one AUROC by hand on a small subset (e.g., 20 test prompts) to verify the sklearn metric.
- [ ] Verify the random-vector control AUROC is indeed near chance (0.5-0.65 range).

## Interpretation

H1 (Localization) is confirmed: eval-awareness is linearly probe-able in Qwen2.5-7B-Instruct's residual stream for English prompts, with near-perfect AUROC and zero FPR on 40 organic deployment prompts.
