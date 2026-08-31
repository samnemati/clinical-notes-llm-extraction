"""Step 4b: figures comparing the two extractors, from outputs/metrics.json
and outputs/downstream.json.

Color convention across all figures (colorblind-safe, checked):
  LLM = blue, rule-based baseline = orange, gold truth = gray.
Every bar carries its value as a label, so nothing depends on color alone.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")  # no display window; just write files
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"llm": "#0072B2", "baseline": "#E69F00", "gold": "#767676"}


def tidy(ax):
    """Shared styling: no box around the plot, light grid behind the bars."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.grid(True, color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)


def grouped_hbars(ax, labels, series, xlim=(0, 1.12)):
    """Horizontal grouped bars. `series` is a dict name -> list of values;
    None values are drawn as a small 'n/a' marker instead of a bar."""
    y = np.arange(len(labels))
    height = 0.8 / len(series)
    for i, (name, values) in enumerate(series.items()):
        offset = (i - (len(series) - 1) / 2) * height
        for j, value in enumerate(values):
            if value is None:
                ax.text(0.01, y[j] + offset, "n/a", va="center", fontsize=8,
                        color="#999999")
                continue
            ax.barh(y[j] + offset, value, height=height * 0.9,
                    color=COLORS[name], label=name if j == 0 else None)
            ax.text(value + 0.015, y[j] + offset, f"{value:.2f}",
                    va="center", fontsize=8, color="#333333")
    ax.set_yticks(y, labels)
    ax.set_xlim(*xlim)
    ax.invert_yaxis()
    tidy(ax)


with open(ROOT / "outputs" / "metrics.json") as f:
    metrics = json.load(f)
with open(ROOT / "outputs" / "downstream.json") as f:
    downstream = json.load(f)

systems = [s for s in ["llm", "baseline"] if s in metrics]

# Figure 1: headline extraction quality --------------------------------------
rows = [
    ("conditions F1", lambda m: m["conditions"]["f1"]),
    ("condition status accuracy", lambda m: m["conditions"]["status_accuracy"]),
    ("medications F1", lambda m: m["medications"]["f1"]),
    ("medication status accuracy", lambda m: m["medications"]["status_accuracy"]),
    ("medication dose accuracy", lambda m: m["medications"]["dose_accuracy"]),
    ("smoking status accuracy", lambda m: m["smoking_accuracy"]),
    ("follow-up interval accuracy", lambda m: m["follow_up_accuracy"]),
    ("A1c accuracy", lambda m: m["a1c_accuracy"]),
]
fig, ax = plt.subplots(figsize=(7.5, 4.2))
grouped_hbars(
    ax,
    [label for label, _ in rows],
    {s: [getter(metrics[s]) for _, getter in rows] for s in systems},
)
ax.legend(frameon=False, loc="lower right")
ax.set_title("Extraction quality: LLM vs rule-based baseline", loc="left")
fig.tight_layout()
fig.savefig(ROOT / "outputs" / "f_extraction_quality.png", dpi=200)

# Figure 2: recall on the hard phenomena -------------------------------------
phen_labels = {
    "negation": "negated findings",
    "historical": "historical conditions",
    "hedged_dx": "hedged diagnoses",
    "discontinued_med": "discontinued medications",
    "misspelling": "entities in misspelled notes",
}
phens = [p for p in phen_labels if any(
    p in metrics[s]["recall_by_phenomenon"] for s in systems)]
fig, ax = plt.subplots(figsize=(7.5, 3.2))
grouped_hbars(
    ax,
    [phen_labels[p] for p in phens],
    {s: [metrics[s]["recall_by_phenomenon"].get(p, {}).get("recall") for p in phens]
     for s in systems},
)
ax.legend(frameon=False, loc="lower right")
ax.set_title("Recall on hard language, by phenomenon", loc="left")
ax.set_xlabel("recall (found with the correct status)")
fig.tight_layout()
fig.savefig(ROOT / "outputs" / "f_phenomena.png", dpi=200)

# Figure 3: downstream model AUC by feature source ---------------------------
model = downstream["readmission_model"]
order = [s for s in ["gold", "llm", "baseline"] if s in model]
fig, ax = plt.subplots(figsize=(6.0, 2.6))
y = np.arange(len(order))
for i, s in enumerate(order):
    ax.barh(i, model[s]["auc_mean"], height=0.6, color=COLORS[s],
            xerr=model[s]["auc_sd"], error_kw={"ecolor": "#333333", "capsize": 3})
    ax.text(model[s]["auc_mean"] + model[s]["auc_sd"] + 0.02, i,
            f"{model[s]['auc_mean']:.2f}", va="center", fontsize=9, color="#333333")
ax.set_yticks(y, [f"{s} features" for s in order])
ax.axvline(0.5, color="#bbbbbb", linewidth=1, linestyle="--")
ax.text(0.503, -0.42, "chance", fontsize=8, color="#999999")
ax.set_xlim(0.4, 1.0)
ax.invert_yaxis()
tidy(ax)
ax.set_title("Readmission model AUC by feature source", loc="left")
ax.set_xlabel("cross-validated AUC (error bar: sd across folds)")
fig.tight_layout()
fig.savefig(ROOT / "outputs" / "f_downstream_auc.png", dpi=200)

print("Wrote outputs/f_extraction_quality.png, f_phenomena.png, f_downstream_auc.png")
