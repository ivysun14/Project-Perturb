import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ===========
# INPUT DATA
# ===========

cell_lines_a = ["K562", "RPE1", "HEPG2", "JURKAT"]
cell_lines_b = ["K562", "RPE1", "HEPG2", "JURKAT"]

# Panel (a) – PerturbQA downsampled dataset
# TODO: Get accurate SUMMER statistics
data_a = {
    "Non-DE": {
        "MV":     [0.947, 0.968, 0.964, 0.961],
        "1-NN":   [0.887, 0.893, 0.889, 0.886],
        "10-NN":  [0.931, 0.941, 0.931, 0.934],
        "SUMMER": [0.89, 0.86, 0.90, 0.89],
    },
    "Up": {
        "MV":     [0.755, 0.700, 0.658, 0.643],
        "1-NN":   [0.948, 0.928, 0.907, 0.844],
        "10-NN":  [0.877, 0.829, 0.830, 0.768],
        "SUMMER": [0.81, 0.75, 0.51, 0.64],
    },
    "Down": {
        "MV":     [0.394, 0.815, 0.708, 0.728],
        "1-NN":   [0.638, 0.956, 0.948, 0.912],
        "10-NN":  [0.424, 0.885, 0.855, 0.845],
        "SUMMER": [0.46, 0.64, 0.85, 0.80],
    },
}

# Panel (b) – Full DE profile
data_b = {
    "Non-DE": {
        "MV":    [0.963, 0.962, 0.979, 0.984],
        "1-NN":  [0.973, 0.973, 0.985, 0.987],
        "10-NN": [0.968, 0.968, 0.982, 0.985],
    },
    "Up": {
        "MV":    [0.00, 0.467, 0.00, 0.00],
        "1-NN":  [0.295, 0.219, 0.237, 0.193],
        "10-NN": [0.525, 0.428, 0.456, 0.451],
    },
    "Down": {
        "MV":    [0.00, 0.504, 0.00, 0.00],
        "1-NN":  [0.267, 0.270, 0.234, 0.188],
        "10-NN": [0.546, 0.443, 0.427, 0.440],
    },
}


colors_4 = ["#4C72B0", "#55A868", "#DD8452", "#C4A0C0"]     # MV, 1-NN, 10-NN, SUMMER
colors_3 = ["#4C72B0", "#55A868", "#DD8452"]                # MV, 1-NN, 10-NN

BAR_WIDTH   = 0.20
SUBPLOT_GAP = 0.35
PANEL_GAP   = 0.45

def plot_panel(axes, data, cell_lines, method_names, colors, bar_width=BAR_WIDTH):
    """
    Draw three grouped-bar subplots (Non-DE, Up, Down) on the provided axes.
    """
    categories = ["Non-DE", "Up", "Down"]
    n_methods  = len(method_names)
    x          = np.arange(len(cell_lines))
    offsets    = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2, n_methods) * bar_width

    for ax, cat in zip(axes, categories):
        for i, (method, color) in enumerate(zip(method_names, colors)):
            ax.bar(x + offsets[i], data[cat][method], width=bar_width,
                   color=color, label=method)
        ax.set_title(cat, fontweight="bold", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(cell_lines, fontsize=9)
        ax.set_xlabel("Cell Line", fontsize=10)
        ax.set_ylabel("Precision", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_tick_params(labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    # Legend above the middle subplot
    handles = [mpatches.Patch(color=c, label=m) for m, c in zip(method_names, colors)]
    axes[1].legend(handles=handles, loc="upper center",
                   bbox_to_anchor=(0.5, 1.22), ncol=n_methods,
                   frameon=True, fontsize=9)


fig = plt.figure(figsize=(14, 9))
fig.subplots_adjust(hspace=PANEL_GAP)

# --- Panel (a) ---
axes_a = [fig.add_subplot(2, 3, i) for i in range(1, 4)]
plot_panel(axes_a, data_a, cell_lines_a,
           method_names=["MV", "1-NN", "10-NN", "SUMMER"],
           colors=colors_4)
fig.text(0.5, 0.97, "(a)", ha="center", va="top", fontsize=14, fontweight="bold")

# --- Panel (b) ---
axes_b = [fig.add_subplot(2, 3, i) for i in range(4, 7)]
plot_panel(axes_b, data_b, cell_lines_b,
           method_names=["MV", "1-NN", "10-NN"],
           colors=colors_3)
fig.text(0.5, 0.52, "(b)", ha="center", va="top", fontsize=14, fontweight="bold")

plt.savefig("per_class_precision_panels.png", dpi=150, bbox_inches="tight")