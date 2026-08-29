#!/usr/bin/env python3
import json
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_RESULTS = Path("audit/h1_out_of_template_results.json")
FIG_DIR = Path("audit/figures_verified")
FIG_DIR.mkdir(parents=True, exist_ok=True)

new = json.loads(OUT_RESULTS.read_text(encoding="utf-8"))

rows = [
    {
        "method": "Activation probe",
        "original_template_auroc": 0.9950,
        "unseen_template_auroc": new["activation_probe"]["auroc"],
    },
    {
        "method": "Character TF-IDF",
        "original_template_auroc": 0.9619,
        "unseen_template_auroc": new["character_tfidf"]["auroc"],
    },
    {
        "method": "Prompt length",
        "original_template_auroc": 0.4869,
        "unseen_template_auroc": new["prompt_length"]["auroc"],
    },
]

df = pd.DataFrame(rows)
df["drop"] = df["original_template_auroc"] - df["unseen_template_auroc"]
df.to_csv(FIG_DIR / "figure4_out_of_template_data.csv", index=False)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
})

fig, ax = plt.subplots(figsize=(8.4, 5.4))

x = range(len(df))
width = 0.36

original_bars = ax.bar(
    [i - width / 2 for i in x],
    df["original_template_auroc"],
    width=width,
    color="#9e9e9e",
    edgecolor="black",
    linewidth=0.6,
    label="Original held-out templates",
)

new_bars = ax.bar(
    [i + width / 2 for i in x],
    df["unseen_template_auroc"],
    width=width,
    color=["#1f77b4", "#ff7f0e", "#7f7f7f"],
    edgecolor="black",
    linewidth=0.6,
    label="Unseen wrapper templates",
)

ax.set_title("Out-of-Template English Framing Generalization")
ax.set_ylabel("AUROC on 80 held-out English prompts")
ax.set_xticks(list(x))
ax.set_xticklabels(df["method"])
ax.set_ylim(0.0, 1.08)
ax.axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.7)
ax.legend(loc="lower left", frameon=True)

for bars in [original_bars, new_bars]:
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

ax.text(
    0.02,
    -0.28,
    "Both models were fit on original English training prompts. "
    "The unseen test replaces every original wrapper with one of five "
    "balanced new eval/deploy wrapper pairs.",
    transform=ax.transAxes,
    fontsize=9.5,
)

fig.tight_layout()
fig.savefig(
    FIG_DIR / "figure4_out_of_template_generalization.png",
    dpi=240,
    bbox_inches="tight",
)
plt.close(fig)

print("Saved:", FIG_DIR / "figure4_out_of_template_generalization.png")
print("Saved:", FIG_DIR / "figure4_out_of_template_data.csv")
print(df.to_string(index=False))
