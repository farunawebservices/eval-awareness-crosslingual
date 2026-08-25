# H1 English Dataset Lock Record

**Lock date:** 2026-08-25  
**Locked by:** GAF  
**Lock type:** Full corpus freeze before activation extraction

## Locked components

| Component | File | Count | Status |
|---|---|---|---|
| Core allocation grid | `opus_h1_70_core_allocation.txt` | 70 task-type slots | Locked |
| Core texts (D1-D7) | `cores_by_domain/` | 70 cores | Locked |
| Wrapper families F1-F5 | Defined in render script | 5 families | Locked |
| Rendered prompts | `h1_70_cores_rendered.jsonl` | 280 prompts | Locked |
| FPR prompts | `h1_fpr_40_prompts.jsonl` | 40 prompts | Locked |
| Train/val/test splits | `h1_train_val_test_splits.json` | 35/15/20 cores | Locked |
| Annotated prompts | `h1_70_cores_with_splits.jsonl` | 280 prompts with split | Locked |

## Lock guarantees

- No core text may be modified after this lock without creating a new versioned dataset (e.g., h1_v2).
- No wrapper family text may be changed.
- No split assignment may be altered.
- Any changes require a new lock record with a new version tag.

## Rationale

This lock prevents post-hoc dataset modification after viewing model activations or probe results. All H1 conclusions must be based on this frozen corpus.

## Next steps after lock

1. Extract residual-stream activations at layers 12, 16, 20, 24 for all 280 prompts.
2. Train logistic-regression probes per layer on train split.
3. Select best layer by validation AUROC.
4. Evaluate on test split.
5. Run FPR analysis on 40 deployment-only prompts.
6. Document H1 result (AUROC, balanced accuracy, confidence intervals).
