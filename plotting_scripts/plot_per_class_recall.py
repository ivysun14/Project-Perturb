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
        "MV":     [0.979, 0.978, 0.969, 0.980],
        "1-NN":   [0.999, 0.999, 0.999, 0.999],
        "10-NN":  [0.989, 0.991, 0.992, 0.992],
        "SUMMER": [0.995, 0.995, 0.995, 0.998],
    },
    "Up": {
        "MV":     [0.614, 0.668, 0.634, 0.430],
        "1-NN":   [0.200, 0.231, 0.225, 0.165],
        "10-NN":  [0.553, 0.525, 0.515, 0.476],
        "SUMMER": [0.27, 0.21, 0.17, 0.23],
    },
    "Down": {
        "MV":     [0.260, 0.747, 0.685, 0.716],
        "1-NN":   [0.051, 0.288, 0.218, 0.219],
        "10-NN":  [0.194, 0.623, 0.489, 0.530],
        "SUMMER": [0.09, 0.19, 0.21, 0.25],
    },
}

# Panel (b) – Full DE profile
data_b = {
    "Non-DE": {
        "MV":    [1.00, 0.999, 1.00, 1.00],
        "1-NN":  [0.972, 0.963, 0.977, 0.985],
        "10-NN": [0.995, 0.990, 0.995, 0.998],
    },
    "Up": {
        "MV":    [0.00, 0.004, 0.00, 0.00],
        "1-NN":  [0.290, 0.275, 0.326, 0.252],
        "10-NN": [0.165, 0.149, 0.186, 0.097],
    },
    "Down": {
        "MV":    [0.00, 0.021, 0.00, 0.00],
        "1-NN":  [0.289, 0.339, 0.329, 0.201],
        "10-NN": [0.142, 0.225, 0.182, 0.088],
    },
}


colors_4 = ["#4C72B0", "#55A868", "#DD8452", "#C4A0C0"]     # MV, 1-NN, 9-NN, SUMMER
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
        ax.set_ylabel("Recall", fontsize=10)
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

plt.savefig("per_class_recall_panels.png", dpi=150, bbox_inches="tight")