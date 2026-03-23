"""
check_coverage.py
=================
Check whether ESM2 embeddings and co-path neighbors cover all perturbations
in the K562, Jurkat, RPE1, and HEPG2 datasets.
"""

import json
import pickle
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

DATASETS = {
    "Jurkat":  "../data/DE_DOC_jurkat_splitted.csv",
    "K562":    "../data/DE_DOC_K562_Replogle_splitted.csv",
    "RPE1":    "../data/DE_DOC_rpe1_splitted.csv",
    "HEPG2":   "../data/DE_DOC_hepg2_splitted.csv",
}
ESM2_PKL    = "data/ESM2_protein_embedding.pkl"
COPATH_JSON = "data/gene_copath_neighbors.json"
PLOT_OUT    = Path("coverage_summary.png")


def load_esm2_genes(path: Path) -> set:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        return set(data.keys())
    raise ValueError(f"Unexpected ESM2 pkl format: {type(data)}")


def load_copath_genes(path: Path) -> set:
    with open(path, "r") as f:
        data = json.load(f)
    return set(data.keys())


def check_coverage(
    dataset_name: str,
    csv_path: Path,
    esm2_genes: set,
    copath_genes: set,
) -> dict:
    """Print coverage report and return counts for plotting."""
    df = pd.read_csv(csv_path)
    perts = set(df["pert"].unique())
    total = len(perts)
    print(f"\n{'='*55}")
    print(f"  {dataset_name}  ({total} unique perturbations)")
    print(f"{'='*55}")

    counts = {"dataset": dataset_name, "total": total}
    for label, ref_genes in [("ESM2 embedding", esm2_genes), ("Co-path neighbors", copath_genes)]:
        missing = perts - ref_genes
        covered = total - len(missing)
        pct = covered / total * 100
        status = "FULL" if not missing else f"PARTIAL ({pct:.1f}%)"
        print(f"  {label:<22} {status}")
        if missing:
            print(f"    Missing ({len(missing)}): {sorted(missing)}")
        key = "esm2" if "ESM2" in label else "copath"
        counts[f"{key}_covered"] = covered
        counts[f"{key}_missing"] = len(missing)

    # GenePT cover all perturbations in the dataset
    print(f"  {'GenePT embedding':<22} FULL (assumed)")
    counts["genept_covered"] = total
    counts["genept_missing"] = 0

    return counts


def plot_summary(rows: list[dict], out_path: Path) -> None:
    datasets  = [r["dataset"] for r in rows]
    n         = len(datasets)
    modes     = ["GenePT", "ESM2", "Co-path"]
    bar_w     = 0.27
    gap       = 0.03
    group_gap = 1.0
    x_centres = [i * group_gap for i in range(n)]
    offsets   = [-(bar_w + gap), 0, bar_w + gap]

    COLOR_COVERED  = "#4C9A8E"
    COLOR_MISSING  = "#E07B6A"
    COLOR_ANNOT    = "white"
    COLOR_PCT      = "#333333"

    max_total = max(r["total"] for r in rows)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    mode_key = {"GenePT": "genept", "ESM2": "esm2", "Co-path": "copath"}
    for mode_i, (mode, offset) in enumerate(zip(modes, offsets)):
        key = mode_key[mode]
        for di, (row, xc) in enumerate(zip(rows, x_centres)):
            x        = xc + offset
            covered  = row[f"{key}_covered"]
            missing  = row[f"{key}_missing"]
            total    = row["total"]

            # covered segment
            ax.bar(x, covered, width=bar_w, color=COLOR_COVERED,
                   zorder=2, linewidth=0)
            # missing segment stacked on top
            bar = ax.bar(x, missing, width=bar_w, bottom=covered,
                         color=COLOR_MISSING, zorder=2, linewidth=0)

            # count label inside missing segment (only if visible)
            if missing > 0:
                ax.text(x, covered + missing / 2, str(missing),
                        ha="center", va="center",
                        fontsize=8, fontweight="bold", color=COLOR_ANNOT,
                        zorder=3, clip_on=False)

            # percentage covered above bar
            pct = covered / total * 100
            ax.text(x, total + max_total * 0.02, f"{pct:.1f}%",
                    ha="center", va="bottom",
                    fontsize=7.5, color=COLOR_PCT, zorder=3, clip_on=False)

    # ── axes cosmetics ──────────────────────────────────────────────────────
    # Two-level x-axis labels:
    #   row 1 (just below bars): "ESM2" / "Co-path" at individual bar x positions
    #   row 2 (below row 1):     dataset name centred on the group
    from matplotlib.transforms import blended_transform_factory
    sub_tf = blended_transform_factory(ax.transData, ax.transAxes)

    ax.set_xticks([])                          # hide default ticks
    ax.tick_params(bottom=False)

    for mode, offset in zip(modes, offsets):
        for xc in x_centres:
            ax.text(xc + offset, -0.04, mode,
                    ha="center", va="top", fontsize=8.5, color="#555555",
                    transform=sub_tf, rotation=20, rotation_mode="anchor")

    for name, xc in zip(datasets, x_centres):
        ax.text(xc, -0.11, name,
                ha="center", va="top", fontsize=11, fontweight="bold",
                transform=sub_tf)

    ax.set_ylim(0, max_total * 1.15)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))
    ax.set_ylabel("Number of perturbations", fontsize=11)
    ax.set_title("Perturbation coverage by embedding type", fontsize=13,
                 fontweight="bold", pad=14)
    ax.set_xlim(x_centres[0] - 0.65, x_centres[-1] + 0.65)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    # ── legend — single box, outside axes to the right ──────────────────────
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_COVERED, label="Covered"),
        Patch(facecolor=COLOR_MISSING, label="Filtered (no representation)"),
    ]
    ax.legend(handles=legend_elements,
              bbox_to_anchor=(1.01, 1), loc="upper left",
              framealpha=0.9, fontsize=9, title="Coverage",
              borderaxespad=0)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved → {out_path}")


def main():
    print("Loading ESM2 embeddings ...")
    esm2_genes = load_esm2_genes(ESM2_PKL)
    print(f"  ESM2 genes: {len(esm2_genes)}")

    print("Loading co-path neighbors ...")
    copath_genes = load_copath_genes(COPATH_JSON)
    print(f"  Co-path genes: {len(copath_genes)}")

    rows = []
    for name, path in DATASETS.items():
        rows.append(check_coverage(name, path, esm2_genes, copath_genes))

    print()
    plot_summary(rows, PLOT_OUT)


if __name__ == "__main__":
    main()
