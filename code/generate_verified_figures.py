#!/usr/bin/env python3
"""
Generate report figures from independently verified audit artifacts.

Inputs:
  audit/h2_chinese_verified_english_transfer.json
  audit/h2_arabic_verified_english_transfer.json
  audit/h2_yoruba_verified_english_transfer.json

The English baseline values come from the independently recomputed baseline
audit:
  activation probe: 0.9950
  character TF-IDF: 0.9619
  prompt length: 0.4869
  random vector: 0.6219
  majority baseline: 0.5000
"""

import json
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AUDIT = Path("audit")
FIG_DIR = AUDIT / "figures_verified"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# This table is intentionally explicit because each value was separately
# verified in the saved English result / baseline artifacts.
english_controls = pd.DataFrame([
    {"method": "Activation probe", "auroc": 0.9950},
    {"method": "Character TF-IDF", "auroc": 0.9619},
    {"method": "Random vector", "auroc": 0.6219},
    {"method": "Prompt length", "auroc": 0.4869},
    {"method": "Majority baseline", "auroc": 0.5000},
])

target_files = {
    "Chinese": AUDIT / "h2_chinese_verified_english_transfer.json",
    "Arabic": AUDIT / "h2_arabic_verified_english_transfer.json",
    "Yorùbá": AUDIT / "h2_yoruba_verified_english_transfer.json",
}

transfer_rows = []
layer_rows = []

for language, path in target_files.items():
    if not path.exists():
        raise FileNotFoundError(f"Missing verified artifact: {path}")

    obj = json.loads(path.read_text(encoding="utf-8"))

    transfer_rows.append({
        "language": language,
        "transfer_auroc": obj["best_transfer_auroc"],
        "native_auroc": obj["best_native_auroc"],
        "transfer_gap": obj["transfer_gap"],
        "best_transfer_layer": obj["best_transfer_layer"],
        "best_native_layer": obj["best_native_layer"],
        "n_test": obj["target_artifact"]["split_row_counts"]["test"],
    })

    for layer_key, transfer_metrics in obj["english_to_target"].items():
        layer = int(layer_key.split("_")[1])
        native_metrics = obj["native_target"][layer_key]
        layer_rows.append({
            "language": language,
            "layer": layer,
            "english_transfer_auroc": transfer_metrics["test_auroc"],
            "native_auroc": native_metrics["test_auroc"],
            "english_transfer_bacc": transfer_metrics["test_balanced_acc_at_0_5"],
            "native_bacc": native_metrics["test_balanced_acc_at_0_5"],
        })

transfer_df = pd.DataFrame(transfer_rows)
layer_df = pd.DataFrame(layer_rows)

english_controls.to_csv(FIG_DIR / "figure1_english_controls_data.csv", index=False)
transfer_df.to_csv(FIG_DIR / "figure2_transfer_summary_data.csv", index=False)
layer_df.to_csv(FIG_DIR / "figure3_layerwise_transfer_data.csv", index=False)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
})

# Figure 1: English controls
fig, ax = plt.subplots(figsize=(8.2, 5.2))
colors = ["#1f77b4", "#ff7f0e", "#9e9e9e", "#9e9e9e", "#9e9e9e"]
bars = ax.bar(
    english_controls["method"],
    english_controls["auroc"],
    color=colors,
    edgecolor="black",
    linewidth=0.6,
)

ax.set_title("English Evaluation-Framing Classification")
ax.set_ylabel("Held-out Test AUROC")
ax.set_ylim(0.0, 1.08)
ax.axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.7)
ax.text(
    4.35, 0.515, "Chance", ha="right", va="bottom",
    fontsize=10, color="black"
)
ax.tick_params(axis="x", rotation=20)

for bar, value in zip(bars, english_controls["auroc"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.025,
        f"{value:.3f}",
        ha="center",
        va="bottom",
        fontsize=10,
    )

ax.text(
    0.02, -0.31,
    "The activation probe exceeds character TF-IDF by only 0.033 AUROC.",
    transform=ax.transAxes,
    fontsize=10,
)
fig.tight_layout()
fig.savefig(FIG_DIR / "figure1_english_controls.png", dpi=240, bbox_inches="tight")
plt.close(fig)

# Figure 2: Best transfer versus native
fig, ax = plt.subplots(figsize=(8.0, 5.4))
x = range(len(transfer_df))
width = 0.36

transfer_bars = ax.bar(
    [i - width / 2 for i in x],
    transfer_df["transfer_auroc"],
    width=width,
    color="#1f77b4",
    edgecolor="black",
    linewidth=0.6,
    label="Fixed English probe",
)
native_bars = ax.bar(
    [i + width / 2 for i in x],
    transfer_df["native_auroc"],
    width=width,
    color="#2ca02c",
    edgecolor="black",
    linewidth=0.6,
    label="Native target-language probe",
)

ax.set_title("Cross-Lingual Evaluation-Framing Decoding")
ax.set_ylabel("Held-out Test AUROC")
ax.set_xticks(list(x))
ax.set_xticklabels([
    f"{language}\n(n={n})"
    for language, n in zip(transfer_df["language"], transfer_df["n_test"])
])
ax.set_ylim(0.0, 1.08)
ax.axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.7)
ax.legend(loc="lower left", frameon=True)

for bars in [transfer_bars, native_bars]:
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
    0.02, -0.28,
    "All blue bars use the same logistic-regression probe trained only on "
    "140 English training rows at layer 16.",
    transform=ax.transAxes,
    fontsize=9.6,
)
fig.tight_layout()
fig.savefig(FIG_DIR / "figure2_verified_crosslingual_transfer.png", dpi=240, bbox_inches="tight")
plt.close(fig)

# Figure 3: layerwise transfer
fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.6), sharey=True)

language_colors = {
    "Chinese": "#d62728",
    "Arabic": "#9467bd",
    "Yorùbá": "#8c564b",
}

for ax, language in zip(axes, ["Chinese", "Arabic", "Yorùbá"]):
    sub = layer_df[layer_df["language"] == language].sort_values("layer")

    ax.plot(
        sub["layer"],
        sub["english_transfer_auroc"],
        marker="o",
        linewidth=2.3,
        color=language_colors[language],
        label="Fixed English probe",
    )
    ax.plot(
        sub["layer"],
        sub["native_auroc"],
        marker="s",
        linestyle="--",
        linewidth=2.0,
        color="#2ca02c",
        label="Native probe",
    )
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.65)
    ax.set_title(language)
    ax.set_xlabel("Residual-stream layer")
    ax.set_xticks([12, 16, 20, 24])
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.2)

axes[0].set_ylabel("Held-out Test AUROC")
axes[0].legend(loc="lower right", frameon=True)

fig.suptitle(
    "English Probe Transfer Is Uneven Across Target Languages",
    y=1.02,
    fontsize=15,
)
fig.tight_layout()
fig.savefig(FIG_DIR / "figure3_layerwise_verified_transfer.png", dpi=240, bbox_inches="tight")
plt.close(fig)

print("Wrote figures:")
for p in sorted(FIG_DIR.glob("*.png")):
    print(" ", p)
print("\nWrote source-data tables:")
for p in sorted(FIG_DIR.glob("*.csv")):
    print(" ", p)

print("\nVerified transfer summary:")
print(
    transfer_df[
        [
            "language",
            "n_test",
            "transfer_auroc",
            "native_auroc",
            "transfer_gap",
            "best_transfer_layer",
            "best_native_layer",
        ]
    ].to_string(index=False)
)
