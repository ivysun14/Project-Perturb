"""
plot_prec_recall_transfer_full.py
==================================
Two-panel grouped bar figure for cross-cell-line transfer (full dataset).
  Panel (a) top:    Per-class Precision
  Panel (b) bottom: Per-class Recall

Data sources
------------
MV:      baseline/majority_transfer_allGene_expr/results/{ds}_majority_transfer_metrics.csv
KNN:     KNN_vote/KNN_transfer_allGene_expr2/results/{ds}_{k}NN_knn_transfer_metrics.csv
STATE:   STATE/state_metrics.csv
Scouter: Scouter/final_metric_cross.csv  (Full column)
VAE:     Manually input
"""

import os
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
# Constants
# ============================================================

DATASETS = ["RPE1", "HepG2", "Jurkat"]
METHODS  = ["MV", "1-NN (GenePT)", "10-NN (GenePT)", "STATE", "Scouter", "VAE"]

_DS_FILE = {"RPE1": "rpe1", "HepG2": "hepg2", "Jurkat": "jurkat"}

# ============================================================
# Data loaders
# ============================================================

def _read_single_row(path):
    if not os.path.exists(path):
        return pd.Series(dtype=float)
    return pd.read_csv(path).iloc[0]

def _get(row, col):
    return float(row[col]) if col in row.index else np.nan

def _mv_row(ds):
    key = _DS_FILE[ds]
    return _read_single_row(
        f"baseline/majority_transfer_allGene_expr/results/"
        f"{key}_majority_transfer_metrics.csv"
    )

def _knn_row(ds, k):
    key = _DS_FILE[ds]
    return _read_single_row(
        f"KNN_vote/KNN_transfer_allGene_expr2/results/"
        f"{key}_{k}NN_knn_transfer_metrics.csv"
    )

_state_df = pd.read_csv("STATE/state_metrics.csv", index_col="dataset")

def _state_row(ds):
    key = _DS_FILE[ds]
    return _state_df.loc[key] if key in _state_df.index else pd.Series(dtype=float)

_scouter_raw = pd.read_csv("Scouter/final_metric_cross.csv", header=[0, 1], index_col=0)
_scouter_raw.columns = pd.MultiIndex.from_tuples(
    [(d.strip(), s.strip()) for d, s in _scouter_raw.columns]
)

def _scouter_val(ds, metric_col):
    key = _DS_FILE[ds]
    if (key, "Full") in _scouter_raw.columns and metric_col in _scouter_raw.index:
        return float(_scouter_raw.loc[metric_col, (key, "Full")])
    return np.nan

# VAE: manually provided per-class precision and recall
_VAE = {
    "precision_class_0": {"RPE1": 0.940623, "HepG2": 0.981701, "Jurkat": 0.963357},
    "precision_class_1": {"RPE1": 0.015931, "HepG2": 0.009025, "Jurkat": 0.004638},
    "precision_class_2": {"RPE1": 0.045390, "HepG2": 0.042269, "Jurkat": 0.053076},
    "recall_class_0":    {"RPE1": 0.288230, "HepG2": 0.391477, "Jurkat": 0.282419},
    "recall_class_1":    {"RPE1": 0.331596, "HepG2": 0.448176, "Jurkat": 0.320656},
    "recall_class_2":    {"RPE1": 0.467469, "HepG2": 0.637289, "Jurkat": 0.613702},
}

def _get_vals(metric_prefix, cls_idx):
    """Return {method: [RPE1, HepG2, Jurkat]} for given metric + class."""
    col = f"{metric_prefix}_class_{cls_idx}"
    result = {}
    for method in METHODS:
        vals = []
        for ds in DATASETS:
            if method == "MV":
                vals.append(_get(_mv_row(ds), col))
            elif method == "1-NN (GenePT)":
                vals.append(_get(_knn_row(ds, 1), col))
            elif method == "10-NN (GenePT)":
                vals.append(_get(_knn_row(ds, 10), col))
            elif method == "STATE":
                vals.append(_get(_state_row(ds), col))
            elif method == "Scouter":
                vals.append(_scouter_val(ds, col))
            elif method == "VAE":
                vals.append(_VAE.get(col, {}).get(ds, np.nan))
        result[method] = vals
    return result

# ============================================================
# Build data dicts
# ============================================================

CLASS_ORDER  = ["Non-DE", "Up", "Down"]
CLASS_LABELS = {
    "Non-DE": "Non-DE (class 0)",
    "Up":     "Up-regulated (class 1)",
    "Down":   "Down-regulated (class 2)",
}

recall = {
    "Non-DE": _get_vals("recall", 0),
    "Up":     _get_vals("recall", 1),
    "Down":   _get_vals("recall", 2),
}

precision = {
    "Non-DE": _get_vals("precision", 0),
    "Up":     _get_vals("precision", 1),
    "Down":   _get_vals("precision", 2),
}

# ============================================================
# Visual design  (no hatches; solid fill for all methods)
# ============================================================

COLORS = {
    "MV":             "#8C8C8C",
    "1-NN (GenePT)":  "#4C72B0",
    "10-NN (GenePT)": "#55A868",
    "STATE":          "#DD8452",
    "Scouter":        "#C44E52",
    "VAE":            "#9467BD",
}
EDGE_COLORS = {
    "MV":             "#6E6E6E",
    "1-NN (GenePT)":  "#3A5A8C",
    "10-NN (GenePT)": "#3D7A4E",
    "STATE":          "#B05A28",
    "Scouter":        "#943B3E",
    "VAE":            "#6B4189",
}

BAR_WIDTH = 0.12

# ============================================================
# Draw panels
# ============================================================

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


# ============================================================
# Build figure
# ============================================================

fig, axes_all = plt.subplots(2, 3, figsize=(16, 9))
fig.subplots_adjust(hspace=0.55, wspace=0.30)
fig.suptitle("Cross-cell-line Prediction Performance (Full Dataset)",
             fontsize=13, fontweight="bold", y=0.98)

draw_panel(axes_all[0], precision, "Precision", "(a)")
draw_panel(axes_all[1], recall,    "Recall",    "(b)")
_empty  = mpatches.Patch(facecolor="none", edgecolor="none")
_mv     = mpatches.Patch(facecolor=COLORS["MV"],             edgecolor=EDGE_COLORS["MV"],             linewidth=0.8)
_nn1    = mpatches.Patch(facecolor=COLORS["1-NN (GenePT)"],  edgecolor=EDGE_COLORS["1-NN (GenePT)"],  linewidth=0.8)
_nn10   = mpatches.Patch(facecolor=COLORS["10-NN (GenePT)"], edgecolor=EDGE_COLORS["10-NN (GenePT)"], linewidth=0.8)
_state  = mpatches.Patch(facecolor=COLORS["STATE"],          edgecolor=EDGE_COLORS["STATE"],          linewidth=0.8)
_scout  = mpatches.Patch(facecolor=COLORS["Scouter"],        edgecolor=EDGE_COLORS["Scouter"],        linewidth=0.8)
_vae    = mpatches.Patch(facecolor=COLORS["VAE"],            edgecolor=EDGE_COLORS["VAE"],            linewidth=0.8)

_handles = [_empty,            _empty,       _mv,   _state,       _nn1,              _scout,         _nn10,              _vae]
_labels  = ["Traditional ML", "DL / LLM", "MV", "STATE (FM)", "1-NN (GenePT)", "Scouter (DL)", "10-NN (GenePT)", "VAE (DL)"]

leg = fig.legend(
    _handles, _labels,
    loc="lower center",
    ncol=4,
    bbox_to_anchor=(0.5, -0.04),
    frameon=True,
    fontsize=9,
)
for text in leg.get_texts():
    if text.get_text() in ("Traditional ML", "DL / LLM"):
        text.set_fontweight("bold")
        text.set_color("#333333")

out_base = "plotting_scripts/prec_recall_transfer_full"
plt.savefig(f"{out_base}.png", dpi=150, bbox_inches="tight")
plt.savefig(f"{out_base}.pdf", bbox_inches="tight")
print(f"Saved: {out_base}.png / .pdf")
