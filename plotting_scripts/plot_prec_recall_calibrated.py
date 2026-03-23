"""
plot_prec_recall_calibrated.py
================================
Two-panel grouped bar figure showing the effect of post-zero calibration
on Scouter and VAE performance (full cross-cell-line transfer dataset).

Data sources
------------
Scouter (orig.):   Scouter/final_metric_cross.csv        (Full column)
Scouter (calib.):  Scouter/calibrated_metrics/all_calibrated_metrics.csv
VAE (orig.):       Manually input (cross-cell-line transfer experiment)
VAE (calib.):      Manually input
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ============================================================
# Load Scouter original (cross-cell-line, Full split)
# ============================================================

_scouter_orig = pd.read_csv("Scouter/final_metric_cross.csv", header=[0, 1], index_col=0)
_scouter_orig.columns = pd.MultiIndex.from_tuples(
    [(d.strip(), s.strip()) for d, s in _scouter_orig.columns]
)

def _sc_orig(ds, col):
    key = {"RPE1": "rpe1", "HepG2": "hepg2", "Jurkat": "jurkat"}[ds]
    if (key, "Full") in _scouter_orig.columns and col in _scouter_orig.index:
        return float(_scouter_orig.loc[col, (key, "Full")])
    return np.nan

# ============================================================
# Load Scouter calibrated
# ============================================================

_scouter_calib = pd.read_csv(
    "Scouter/calibrated_metrics/all_calibrated_metrics.csv", index_col="dataset"
)

def _sc_calib(ds, col):
    key = {"RPE1": "rpe1", "HepG2": "hepg2", "Jurkat": "jurkat"}[ds]
    if key in _scouter_calib.index and col in _scouter_calib.columns:
        return float(_scouter_calib.loc[key, col])
    return np.nan

# ============================================================
# VAE data (manually provided)
# ============================================================

_VAE_ORIG = {
    "precision_class_0": {"RPE1": 0.940623, "HepG2": 0.981701, "Jurkat": 0.963357},
    "precision_class_1": {"RPE1": 0.015931, "HepG2": 0.009025, "Jurkat": 0.004638},
    "precision_class_2": {"RPE1": 0.045390, "HepG2": 0.042269, "Jurkat": 0.053076},
    "recall_class_0":    {"RPE1": 0.288230, "HepG2": 0.391477, "Jurkat": 0.282419},
    "recall_class_1":    {"RPE1": 0.331596, "HepG2": 0.448176, "Jurkat": 0.320656},
    "recall_class_2":    {"RPE1": 0.467469, "HepG2": 0.637289, "Jurkat": 0.613702},
}

_VAE_CALIB = {
    "precision_class_0": {"RPE1": 0.972785, "HepG2": 0.991159, "Jurkat": 0.985936},
    "precision_class_1": {"RPE1": 0.050655, "HepG2": 0.024449, "Jurkat": 0.010532},
    "precision_class_2": {"RPE1": 0.059995, "HepG2": 0.058653, "Jurkat": 0.080064},
    "recall_class_0":    {"RPE1": 0.677807, "HepG2": 0.661812, "Jurkat": 0.595516},
    "recall_class_1":    {"RPE1": 0.202218, "HepG2": 0.362439, "Jurkat": 0.207942},
    "recall_class_2":    {"RPE1": 0.587657, "HepG2": 0.754951, "Jurkat": 0.730094},
}

# ============================================================
# Build data dicts
# ============================================================

DATASETS     = ["RPE1", "HepG2", "Jurkat"]
METHODS      = ["Scouter (orig.)", "Scouter (calib.)", "VAE (orig.)", "VAE (calib.)"]
CLASS_ORDER  = ["Non-DE", "Up", "Down"]
CLASS_LABELS = {
    "Non-DE": "Non-DE (class 0)",
    "Up":     "Up-regulated (class 1)",
    "Down":   "Down-regulated (class 2)",
}

def _build(metric_prefix, cls_idx):
    col = f"{metric_prefix}_class_{cls_idx}"
    return {
        "Scouter (orig.)":  [_sc_orig(ds, col)             for ds in DATASETS],
        "Scouter (calib.)": [_sc_calib(ds, col)            for ds in DATASETS],
        "VAE (orig.)":      [_VAE_ORIG.get(col, {}).get(ds, np.nan) for ds in DATASETS],
        "VAE (calib.)":     [_VAE_CALIB.get(col, {}).get(ds, np.nan) for ds in DATASETS],
    }

prec = {cls: _build("precision", i) for i, cls in enumerate(CLASS_ORDER)}
rec  = {cls: _build("recall",    i) for i, cls in enumerate(CLASS_ORDER)}

# ============================================================
# Visual design
# Same hue per model; lighter shade = original, darker = calibrated
# ============================================================

COLORS = {
    "Scouter (orig.)":  "#ECA9AB",   # light red
    "Scouter (calib.)": "#C44E52",   # dark red
    "VAE (orig.)":      "#C5B0D5",   # light purple
    "VAE (calib.)":     "#9467BD",   # dark purple
}
EDGE_COLORS = {
    "Scouter (orig.)":  "#C44E52",
    "Scouter (calib.)": "#8B1E1E",
    "VAE (orig.)":      "#9467BD",
    "VAE (calib.)":     "#6B4189",
}

BAR_WIDTH = 0.16

def draw_panel(axes, data, y_label, panel_label):
    n_methods  = len(METHODS)
    n_datasets = len(DATASETS)
    x          = np.arange(n_datasets)
    offsets    = (np.arange(n_methods) - (n_methods - 1) / 2) * BAR_WIDTH

    for ax, cls in zip(axes, CLASS_ORDER):
        for mi, method in enumerate(METHODS):
            vals = np.array(data[cls][method], dtype=float)
            for di in range(n_datasets):
                if np.isnan(vals[di]):
                    continue
                ax.bar(
                    x[di] + offsets[mi],
                    vals[di],
                    width=BAR_WIDTH,
                    color=COLORS[method],
                    edgecolor=EDGE_COLORS[method],
                    linewidth=0.8,
                    zorder=3,
                )

        ax.set_title(CLASS_LABELS[cls], fontweight="bold", fontsize=10, pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels(DATASETS, fontsize=9)
        ax.set_ylabel(y_label, fontsize=9)
        ax.set_ylim(0, 1.08)
        ax.yaxis.set_tick_params(labelsize=8)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

    axes[0].text(-0.18, 1.10, panel_label, transform=axes[0].transAxes,
                 fontsize=13, fontweight="bold", va="top")


fig, axes_all = plt.subplots(2, 3, figsize=(16, 9))
fig.subplots_adjust(hspace=0.55, wspace=0.30)
fig.suptitle("Effect of Post-Zero Calibration on Precision and Recall (Full Dataset)",
             fontsize=13, fontweight="bold", y=0.98)

draw_panel(axes_all[0], prec, "Precision", "(a)")
draw_panel(axes_all[1], rec,  "Recall",    "(b)")
def _p(m):
    return mpatches.Patch(facecolor=COLORS[m], edgecolor=EDGE_COLORS[m], linewidth=0.8)

_empty = mpatches.Patch(facecolor="none", edgecolor="none")

_handles = [
    _empty,            _empty,                           # col 0: group headers
    _p("Scouter (orig.)"), _p("VAE (orig.)"),            # col 1
    _p("Scouter (calib.)"), _p("VAE (calib.)"),          # col 2
    _empty, _empty,                                       # col 3: padding
]
_labels = [
    "Scouter", "VAE",
    "Scouter (orig.)", "VAE (orig.)",
    "Scouter (calib.)", "VAE (calib.)",
    "", "",
]

leg = fig.legend(
    _handles, _labels,
    loc="lower center",
    ncol=4,
    bbox_to_anchor=(0.5, -0.04),
    frameon=True,
    fontsize=9,
)
for text in leg.get_texts():
    if text.get_text() in ("Scouter", "VAE"):
        text.set_fontweight("bold")
        text.set_color("#333333")

out_base = "plotting_scripts/prec_recall_calibrated"
plt.savefig(f"{out_base}.png", dpi=150, bbox_inches="tight")
plt.savefig(f"{out_base}.pdf", bbox_inches="tight")
print(f"Saved: {out_base}.png / .pdf")
