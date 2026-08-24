# Cross-Lingual Eval-Awareness Probing and Steering

MATS 12.0 (Neel Nanda) application project. Investigates whether an eval-awareness direction extracted from English contrastive prompts generalizes to other languages in `Qwen2.5-7B-Instruct`, or whether probing/steering silently fails outside English.

See `docs/proposal.md` for the full research proposal (hypotheses, methodology, rigor checklist).

## Status
- [ ] Hour 0-1: Dataset feasibility check
- [ ] Dataset construction (EN / ZH / AR / low-resource)
- [ ] H1: English probe localization
- [ ] H2: Cross-lingual probe transfer
- [ ] H3: Steering transfer
- [ ] Zero-trust audit
- [ ] Write-up + executive summary

## Structure
```
data/
  en/          # English eval + deployment prompts
  zh/          # Chinese (CMMLU + WildChat)
  ar/          # Arabic (ArabicMMLU + WildChat)
  lowres/      # Yoruba/Hausa/Swahili (IrokoBench AfriMMLU + translated WildChat)
src/
  extract_activations.py
  train_probe.py
  steering.py
  transfer_eval.py
notebooks/
outputs/
  activations/
  probes/
  results/
  transcripts/       # raw generations for zero-trust manual audit
docs/
  proposal.md
CLAUDE.md            # agent system prompt / hard constraints
requirements.txt
```

## Time tracking
Per Neel's rules, GPU setup/env config is NOT counted against the 20-hour budget. Start the clock (Toggl) only once real dataset/experiment work begins. Log active hours in `docs/time_log.md`.
