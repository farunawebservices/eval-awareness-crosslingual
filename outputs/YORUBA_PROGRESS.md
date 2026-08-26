# Yorùbá H2 Progress Summary

**Date:** 2026-08-26  
**Status:** Dataset complete, activation extraction in progress (100/700 = 14%)

## Completed

1. ✅ 70-core Yorùbá dataset generated (7 domains × 10 cores)
2. ✅ 5 wrapper families (F1-F5) × 4 formats = 20 prompts per core
3. ✅ 700 total prompts rendered
4. ✅ Train/val/test splits assigned (350/150/200)
5. 🔄 Activation extraction: 100/700 complete (14%)

## Remaining

1. ⏳ Complete activation extraction (~45 min at current rate, or ~10 min on A100)
2. ⏳ Run transfer analysis
3. ⏳ Write H2 Yorùbá results report
4. ⏳ H3 Steering transfer experiments (all 4 languages)

## Files Created

- `datasets/h2_yoruba_cores.py` — 70 Yorùbá cores
- `datasets/h2_yoruba_wrappers.py` — Yorùbá wrappers
- `datasets/h2_yoruba_prompts.jsonl` — 700 rendered prompts
- `datasets/h2_yoruba_splits.jsonl` — Split assignments
- `scripts/render_h2_dataset.py` — Rendering script
- `scripts/assign_splits.py` — Split assignment script
- `scripts/extract_activations_h2.py` — Activation extraction (slow, CPU offload)
- `scripts/extract_activations_h2_v2.py` — Memory-efficient version
- `scripts/analyze_transfer_h2.py` — Transfer analysis

## Next Steps

1. Complete activation extraction on larger GPU instance
2. Run transfer analysis
3. Write Yorùbá H2 report
4. Proceed to H3 steering experiments
