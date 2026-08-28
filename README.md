# The Probe That Thought It Was Awareness

## Red-Teaming Cross-Lingual Evaluation-Awareness Monitoring

This repository accompanies a MATS 12.0 application project investigating whether an English activation probe for controlled evaluation framing transfers across languages.

## Main result

On a core-disjoint English test split, a residual-stream logistic-regression probe reached AUROC 0.9950. A character TF-IDF baseline reached 0.9619, so nearly all benchmark discrimination was available from surface text.

A fixed probe trained only on 140 English layer-16 training activations achieved:

| Target | Test rows | English-to-target AUROC | Native target AUROC |
|---|---:|---:|---:|
| Chinese | 80 | 0.7719 | 0.9925 |
| Arabic | 80 | 0.8631 | 0.9962 |
| Yorùbá | 200 | 0.4527 | 1.0000 |

The main conclusion is not that the model has a universal evaluation-awareness direction. It is that eval-awareness monitors require lexical baselines, code-provenance audits, and cross-lingual stress tests before they should be trusted.

## Important provenance correction

A historical Yorùbá result artifact reported an AUROC of 1.0000 under the name `english_probe_transfer`. Code inspection showed that the script fit its logistic regression on the dataset supplied to it, including when that dataset was Yorùbá. The canonical reproduction script in this release fits only on English training data and produces the corrected English-to-Yorùbá AUROC of 0.4527.

## Contents

- `report/REPORT.md`: Full write-up.
- `figures/`: The three figures in the report.
- `results/`: CSV tables used to generate figures.
- `data/`: Prompt datasets and the Yorùbá activation manifest.
- `code/reproduce_verified_english_transfer.py`: Canonical source-probe transfer reproduction.
- `code/generate_verified_figures.py`: Figure-generation script.
- `audit/`: Machine-readable baseline, split, provenance, and verified transfer results.

## Reproduction note

This compact release intentionally excludes large activation tensors and model checkpoints. The included scripts document the exact analyses, but executing the transfer-reproduction script requires the source artifacts at these original paths:

```text
outputs/h1_english_activations.pt
outputs/h2_chinese_activations_4layers.pt
outputs/h2_arabic_activations_4layers.pt
outputs/h2_yoruba_activations_4layers.pt
```

The report should be read as the authoritative description of the verified analyses and limitations.
