"""
plot_dumbbell_calibration.py
=============================
Dumbbell plot showing the effect of post-zero calibration on Scouter and VAE.

Scouter (before):  Scouter/final_metric_cross.csv        (Full column)
Scouter (after):   Scouter/calibrated_metrics/all_calibrated_metrics.csv
VAE (before/after): Manually provided
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
})

# ============================================================
# Load Scouter data
# ============================================================

_scouter_orig_raw = pd.read_csv("Scouter/final_metric_cross.csv", header=[0, 1], index_col=0)
_scouter_orig_raw.columns = pd.MultiIndex.from_tuples(
    [(d.strip(), s.strip()) for d, s in _scouter_orig_raw.columns]
)

def _sc_orig(ds, col):
    key = {"RPE1": "rpe1", "HepG2": "hepg2", "Jurkat": "jurkat"}[ds]
    if (key, "Full") in _scouter_orig_raw.columns and col in _scouter_orig_raw.index:
        return float(_scouter_orig_raw.loc[col, (key, "Full")])
    return np.nan

_sc_calib_df = pd.read_csv(
    "Scouter/calibrated_metrics/all_calibrated_metrics.csv", index_col="dataset"
)

def _sc_calib(ds, col):
    key = {"RPE1": "rpe1", "HepG2": "hepg2", "Jurkat": "jurkat"}[ds]
    if key in _sc_calib_df.index and col in _sc_calib_df.columns:
        return float(_sc_calib_df.loc[key, col])
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
# Layout constants
# ============================================================

DATASETS = ["RPE1", "HepG2", "Jurkat"]

# y positions: Scouter above VAE within each dataset group; dataset groups separated by gap
# Top-to-bottom: RPE1, HepG2, Jurkat (reversed so RPE1 is at top of the plot)
Y_POS = {
    ("RPE1",   "Scouter"): 7,
    ("RPE1",   "VAE"):     6,
    # gap at 5
    ("HepG2",  "Scouter"): 4,
    ("HepG2",  "VAE"):     3,
    # gap at 2
    ("Jurkat", "Scouter"): 1,
    ("Jurkat", "VAE"):     0,
}
DS_LABEL_Y  = {"RPE1": 6.5, "HepG2": 3.5, "Jurkat": 0.5}
Y_SEPARATOR = [2.0, 5.0]   # horizontal dashed lines between dataset groups

COLORS = {
    "Scouter": "#C44E52",
    "VAE":     "#9467BD",
}
DOT_SIZE   = 90
LINE_ALPHA = 0.65
LINE_WIDTH = 2.0

CLASS_TITLES = {
    "Non-DE": "Non-DE (class 0)",
    "Up":     "Up-regulated (class 1)",
    "Down":   "Down-regulated (class 2)",
}
CLASS_IDX = {"Non-DE": 0, "Up": 1, "Down": 2}

# ============================================================
# Helper: get a single before/after value pair
# ============================================================

def _get_pair(metric_prefix, cls_idx, ds, model):
    col = f"{metric_prefix}_class_{cls_idx}"
    if model == "Scouter":
        before = _sc_orig(ds, col)
        after  = _sc_calib(ds, col)
    else:  # VAE
        before = _VAE_ORIG.get(col, {}).get(ds, np.nan)
        after  = _VAE_CALIB.get(col, {}).get(ds, np.nan)
    return before, after

# ============================================================
# Drawing function
# ============================================================

def draw_dumbbell(ax, metric_prefix, cls_key, x_label):
    cls_idx = CLASS_IDX[cls_key]

    for (ds, model), y in Y_POS.items():
        before, after = _get_pair(metric_prefix, cls_idx, ds, model)
        color = COLORS[model]

        # Connecting line
        if not (np.isnan(before) or np.isnan(after)):
            ax.plot([before, after], [y, y],
                    color=color, lw=LINE_WIDTH, alpha=LINE_ALPHA,
                    solid_capstyle='round', zorder=2)

        # Before: open circle
        if not np.isnan(before):
            ax.scatter(before, y, s=DOT_SIZE,
                       facecolor='white', edgecolor=color,
                       linewidth=1.8, zorder=4)

        # After: filled circle
        if not np.isnan(after):
            ax.scatter(after, y, s=DOT_SIZE,
                       facecolor=color, edgecolor=color,
                       linewidth=1.8, zorder=4)

    # Dataset group separators
    for yy in Y_SEPARATOR:
        ax.axhline(yy, color='#CCCCCC', lw=0.9, linestyle='--', zorder=1)

    # Dataset labels (drawn to the left of the plot using axis-fraction x)
    for ds, y_mid in DS_LABEL_Y.items():
        ax.text(-0.04, y_mid, ds,
                transform=ax.get_yaxis_transform(),
                ha='right', va='center',
                fontsize=8.5, fontweight='bold', color='#333333')

    # y-axis ticks: alternating Scouter / VAE labels, colored
    ytick_pos = [0, 1, 3, 4, 6, 7]
    ytick_labels = ["VAE", "Scouter", "VAE", "Scouter", "VAE", "Scouter"]
    ax.set_yticks(ytick_pos)
    ax.set_yticklabels(ytick_labels, fontsize=8)
    for tick, label in zip(ax.get_yticklabels(), ytick_labels):
        tick.set_color(COLORS[label])

    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.6, 7.6)
    ax.set_xlabel(x_label, fontsize=9)
    ax.set_title(CLASS_TITLES[cls_key], fontweight='bold', fontsize=10, pad=6)
    ax.axvline(0, color='black', lw=0.5)
    ax.grid(axis='x', linestyle='--', alpha=0.35, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)

# ============================================================
# Build figure
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
fig.subplots_adjust(hspace=0.55, wspace=0.38)
fig.suptitle("Effect of Post-Zero Calibration on Precision and Recall",
             fontsize=13, fontweight='bold', y=0.98)

# Panel (a): Precision
for ax, cls in zip(axes[0], ["Non-DE", "Up", "Down"]):
    draw_dumbbell(ax, "precision", cls, "Precision")
axes[0][0].text(-0.22, 1.10, "(a)", transform=axes[0][0].transAxes,
                fontsize=13, fontweight='bold', va='top')

# Panel (b): Recall
for ax, cls in zip(axes[1], ["Non-DE", "Up", "Down"]):
    draw_dumbbell(ax, "recall", cls, "Recall")
axes[1][0].text(-0.22, 1.10, "(b)", transform=axes[1][0].transAxes,
                fontsize=13, fontweight='bold', va='top')

# ============================================================
# Legend
# ============================================================

_open_dot  = mlines.Line2D([], [], marker='o', linestyle='None',
                            markerfacecolor='white', markeredgecolor='#666666',
                            markeredgewidth=1.8, markersize=8,
                            label='Before calibration')
_fill_dot  = mlines.Line2D([], [], marker='o', linestyle='None',
                            markerfacecolor='#666666', markeredgecolor='#666666',
                            markersize=8, label='After calibration')
_sc_line   = mlines.Line2D([], [], color=COLORS["Scouter"], lw=2.5,
                            label='Scouter')
_vae_line  = mlines.Line2D([], [], color=COLORS["VAE"],     lw=2.5,
                            label='VAE')

fig.legend(
    handles=[_sc_line, _vae_line, _open_dot, _fill_dot],
    loc='lower center',
    ncol=4,
    bbox_to_anchor=(0.5, -0.04),
    frameon=True,
    fontsize=9,
)

out_base = "plotting_scripts/dumbbell_calibration"
plt.savefig(f"{out_base}.png", dpi=150, bbox_inches="tight")
plt.savefig(f"{out_base}.pdf", bbox_inches="tight")
print(f"Saved: {out_base}.png / .pdf")
