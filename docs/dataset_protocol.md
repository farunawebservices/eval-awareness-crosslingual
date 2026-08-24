# Dataset Protocol v0.1

## Goal
Measure whether eval-context representations transfer cross-lingually while minimizing confounds from semantic content, language, formatting, and prompt length.

## Primary controlled dataset
For each language (English, Chinese, Arabic, Yorùbá), create matched prompt pairs:
- deployment: ordinary user request, no benchmark/evaluation/grading markers;
- eval: same semantic task, with an explicit evaluation wrapper.

Within each pair, preserve:
- underlying task and factual content;
- answer format where possible;
- language;
- approximate token length.

The intended manipulated variable is evaluation framing only.

## Naturalistic external-validation dataset
Use CMMLU, ArabicMMLU, and AfriMMLU samples as naturally occurring eval-like prompts. Do not treat results from these data alone as proof of eval-awareness because MCQ format, academic content, and source-specific artifacts are confounds.

## Sampling
Use a deterministic seed. Save IDs, source dataset, subject, split, prompt text, and all transformations.
Do not use held-out evaluation prompts for probe fitting.
