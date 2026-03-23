"""
plot_prec_recall_perturbQA.py
=============================
Two-panel grouped bar figure for the PerturbQA downsampled datasets.
  Panel (a) top:    Per-class Precision
  Panel (b) bottom: Per-class Recall

Data sources
------------
MV:      hardcoded from SLURM logs (baseline/DE_PerturbQA_expr/)
KNN:     KNN_vote/KNN_perturbQA_expr/results/*genePT*knn_metrics.csv
SUMMER:  SUMMER/summer_metrics.csv (computed from SUMMER pkl files)
Scouter: Scouter/final_metrics_within.csv (Sample column)
VAE:     Manual input
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
# Loaders
# ============================================================

_KNN_PQA_PREFIX = {
    "K562": "perturbqa_k562",
    "RPE1": "perturbqa_rpe1",
    "HepG2": "perturbqa_hepg2",
    "Jurkat": "perturbqa_jurkat",
}
_KNN_RESULTS_DIR = "KNN_vote/KNN_perturbQA_expr/results"

DATASETS = ["K562", "RPE1", "HepG2", "Jurkat"]
_DATASET_KEYS = ["k562", "rpe1", "hepg2", "jurkat"]


def _read_knn(prefix, embedding, k, col):
    if prefix is None:
        return np.nan
    path = os.path.join(_KNN_RESULTS_DIR, f"{prefix}_{embedding}_{k}NN_knn_metrics.csv")
    if not os.path.exists(path):
        return np.nan
    return float(pd.read_csv(path).iloc[0][col])


def _knn(embedding, k, col):
    return [_read_knn(_KNN_PQA_PREFIX[ds], embedding, k, col) for ds in DATASETS]


_summer_csv = pd.read_csv("SUMMER/summer_metrics.csv", index_col="dataset")


def _summer(col):
    return [float(_summer_csv.loc[ds, col]) if ds in _summer_csv.index else np.nan
            for ds in _DATASET_KEYS]


_scouter_csv = pd.read_csv("Scouter/final_metrics_within.csv", header=[0, 1], index_col=0)
_scouter_csv.columns = pd.MultiIndex.from_tuples(
    [(d.strip(), s.strip()) for d, s in _scouter_csv.columns]
)


def _scouter(row, split):
    return [float(_scouter_csv.loc[row, (ds, split)])
            if (ds, split) in _scouter_csv.columns else np.nan
            for ds in _DATASET_KEYS]


# ============================================================
# Data
# ============================================================

METHODS = ["MV", "1-NN (GenePT)", "10-NN (GenePT)", "SUMMER", "Scouter", "VAE"]

prec = {
    "Non-DE": {
        "MV":             [0.947, 0.968, 0.964, 0.961],
        "1-NN (GenePT)":  _knn("genePT", 1,  "precision_class_0"),
        "10-NN (GenePT)": _knn("genePT", 10, "precision_class_0"),
        "SUMMER":         _summer("precision_class_0"),
        "Scouter":        _scouter("precision_class_0", "Sample"),
        "VAE":            [0.877, 0.851, 0.901, 0.846]
    },
    "Up": {
        "MV":             [0.755, 0.700, 0.658, 0.643],
        "1-NN (GenePT)":  _knn("genePT", 1,  "precision_class_1"),
        "10-NN (GenePT)": _knn("genePT", 10, "precision_class_1"),
        "SUMMER":         _summer("precision_class_1"),
        "Scouter":        _scouter("precision_class_1", "Sample"),
        "VAE":            [0.097, 0.078, 0.079, 0.029]
    },
    "Down": {
        "MV":             [0.394, 0.815, 0.708, 0.728],
        "1-NN (GenePT)":  _knn("genePT", 1,  "precision_class_2"),
        "10-NN (GenePT)": _knn("genePT", 10, "precision_class_2"),
        "SUMMER":         _summer("precision_class_2"),
        "Scouter":        _scouter("precision_class_2", "Sample"),
        "VAE":            [0.053, 0.121, 0.102, 0.153]
    },
}

rec = {
    "Non-DE": {
        "MV":             [0.979, 0.978, 0.969, 0.980],
        "1-NN (GenePT)":  _knn("genePT", 1,  "recall_class_0"),
        "10-NN (GenePT)": _knn("genePT", 10, "recall_class_0"),
        "SUMMER":         _summer("recall_class_0"),
        "Scouter":        _scouter("recall_class_0", "Sample"),
        "VAE":            [0.363, 0.308, 0.424, 0.321]
    },
    "Up": {
        "MV":             [0.614, 0.668, 0.634, 0.430],
        "1-NN (GenePT)":  _knn("genePT", 1,  "recall_class_1"),
        "10-NN (GenePT)": _knn("genePT", 10, "recall_class_1"),
        "SUMMER":         _summer("recall_class_1"),
        "Scouter":        _scouter("recall_class_1", "Sample"),
        "VAE":            [0.526, 0.474, 0.512, 0.362]
    },
    "Down": {
        "MV":             [0.260, 0.747, 0.685, 0.716],
        "1-NN (GenePT)":  _knn("genePT", 1,  "recall_class_2"),
        "10-NN (GenePT)": _knn("genePT", 10, "recall_class_2"),
        "SUMMER":         _summer("recall_class_2"),
        "Scouter":        _scouter("recall_class_2", "Sample"),
        "VAE":            [0.381, 0.628, 0.560, 0.570]
    },
}

# ============================================================
# Visual design
# ============================================================

COLORS = {
    "MV":             "#8C8C8C",
    "1-NN (GenePT)":  "#4C72B0",
    "10-NN (GenePT)": "#55A868",
    "SUMMER":         "#DD8452",
    "Scouter":        "#C44E52",
    "VAE":            "#9467BD",
}
EDGE_COLORS = {
    "MV":             "#6E6E6E",
    "1-NN (GenePT)":  "#3A5A8C",
    "10-NN (GenePT)": "#3D7A4E",
    "SUMMER":         "#B05A28",
    "Scouter":        "#943B3E",
    "VAE":            "#6B4189",
}

BAR_WIDTH   = 0.14
CLASS_ORDER = ["Non-DE", "Up", "Down"]
CLASS_LABELS = {
    "Non-DE": "Non-DE (class 0)",
    "Up":     "Up-regulated (class 1)",
    "Down":   "Down-regulated (class 2)",
}


def draw_panel(axes, data, y_label, panel_label):
    n_methods  = len(METHODS)
    n_datasets = len(DATASETS)
    x = np.arange(n_datasets)
    offsets = (np.arange(n_methods) - (n_methods - 1) / 2) * BAR_WIDTH

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
fig.suptitle("Precision and Recall by Differential Expression Class (PerturbQA Dataset)",
             fontsize=13, fontweight="bold", y=0.98)

draw_panel(axes_all[0], prec, "Precision", "(a)")
draw_panel(axes_all[1], rec,  "Recall",    "(b)")
_empty   = mpatches.Patch(facecolor="none", edgecolor="none")
_mv      = mpatches.Patch(facecolor=COLORS["MV"],             edgecolor=EDGE_COLORS["MV"],             linewidth=0.8)
_nn1     = mpatches.Patch(facecolor=COLORS["1-NN (GenePT)"],  edgecolor=EDGE_COLORS["1-NN (GenePT)"],  linewidth=0.8)
_nn10    = mpatches.Patch(facecolor=COLORS["10-NN (GenePT)"], edgecolor=EDGE_COLORS["10-NN (GenePT)"], linewidth=0.8)
_summer  = mpatches.Patch(facecolor=COLORS["SUMMER"],         edgecolor=EDGE_COLORS["SUMMER"],         linewidth=0.8)
_scouter = mpatches.Patch(facecolor=COLORS["Scouter"],        edgecolor=EDGE_COLORS["Scouter"],        linewidth=0.8)
_vae     = mpatches.Patch(facecolor=COLORS["VAE"],            edgecolor=EDGE_COLORS["VAE"],            linewidth=0.8)

_handles = [_empty,            _empty,      _mv,   _summer,       _nn1,              _scouter,      _nn10,              _vae      ]
_labels  = ["Traditional ML", "DL / LLM", "MV", "SUMMER (LLM)", "1-NN (GenePT)", "Scouter (DL)", "10-NN (GenePT)", "VAE (DL)"]

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

plt.savefig("plotting_scripts/prec_recall_perturbQA.png", dpi=150, bbox_inches="tight")
plt.savefig("plotting_scripts/prec_recall_perturbQA.pdf", bbox_inches="tight")
print("Saved: plotting_scripts/prec_recall_perturbQA.png / .pdf")
