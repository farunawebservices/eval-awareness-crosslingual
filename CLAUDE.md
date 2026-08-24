# Agent Constraints — Read Before Writing Any Code

You are a research coding assistant for a cross-lingual mechanistic interpretability project testing three hypotheses (H1, H2, H3 — see `docs/proposal.md`). Follow these constraints without exception:

1. **HYPOTHESIS DISCIPLINE**: Before writing any experiment code, restate which of H1, H2, or H3 this code is testing, and what result would confirm vs. disconfirm it. Do not write code for an experiment that doesn't map to one of these three hypotheses without flagging it first.

2. **MANDATORY CONTROLS**: Any activation steering experiment MUST automatically also run the random-vector control and the prompt-only control in the same script, on the same prompts, and report all three side by side. Never report a steered result without its control run alongside it.

3. **PER-LANGUAGE PARITY**: Any analysis run on the English dataset must be run identically (same code path, same metrics, same layer choices unless explicitly changed) on every other language. Flag immediately if a language's results are being computed differently than English's — this would invalidate comparison.

4. **ZERO-TRUST LOGGING**: Print tensor shapes, norms, and device placement before and after every hook. Save all raw generated text outputs to disk as JSONL (prompt, language, condition, raw output) in `outputs/transcripts/` — never summarize outputs in a variable without also persisting the raw string for manual reading.

5. **STATIC CONDITIONAL TRIGGERING**: When implementing conditional steering, compute the probe score ONCE on the prompt before generation begins, freeze it as a boolean `should_steer` flag, and do not re-evaluate the probe during token-by-token generation. If token-level re-evaluation seems needed, stop and ask first.

6. **NO SILENT FALLBACKS**: If a dataset, translation, or model call fails for one language, do not silently skip it or substitute placeholder/synthetic data — stop and report it.

7. **PERSISTENT STATE**: Load model weights ONCE into memory (persistent Jupyter kernel or tmux+ipython session). Never reload the model inside a loop. Save activations/probe weights/checkpoints to disk so a crashed kernel isn't a disaster.

8. **VERIFICATION IS NOT YOUR JOB ALONE**: The researcher will personally read transcripts and recompute at least one headline number by hand for each hypothesis. Structure outputs to make this easy (clear file names, small readable JSONL, not opaque pickles).
