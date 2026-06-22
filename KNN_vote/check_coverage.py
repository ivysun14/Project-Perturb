"""
check_coverage.py
=================
Check whether ESM2, WikiCrow, and co-path neighbors cover all perturbations
in the K562, Jurkat, RPE1, and HepG2 datasets. GenePT is assumed to have full
coverage. Run from KNN_vote/.
"""

import json
import pickle
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

DATA_DIR    = Path("/home/jsun140/scratchyiqunchen/project_perturb_data/csv")
KNN_DIR     = Path("/home/jsun140/scratchyiqunchen/jsun140/Project-Perturb/KNN_vote")
WIKICROW_DIR = Path("/home/jsun140/scratchyiqunchen/jsun140/Project-Perturb/wikicrow_embeddings")

DATASETS = {
    "K562":   DATA_DIR / "DE_DOC_K562_splitted.csv",
    "RPE1":   DATA_DIR / "DE_DOC_RPE1_splitted.csv",
    "HepG2":  DATA_DIR / "DE_DOC_HepG2_splitted.csv",
    "Jurkat": DATA_DIR / "DE_DOC_Jurkat_splitted.csv",
}
ESM2_PKL     = KNN_DIR  / "ESM2_protein_embedding.pkl"
WIKICROW_PKL = WIKICROW_DIR / "wikicrow_embeddings.pkl"
COPATH_JSON  = KNN_DIR  / "gene_copath_neighbors.json"
PLOT_OUT     = Path("coverage_summary.pdf")


def load_pkl_genes(path: Path) -> set:
    """Load a {gene: embedding} pkl and return the set of gene names."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        return set(data.keys())
    raise ValueError(f"Unexpected pkl format at {path}: {type(data)}")


def load_copath_genes(path: Path) -> set:
    with open(path, "r") as f:
        data = json.load(f)
    return set(data.keys())


def check_coverage(
    dataset_name: str,
    csv_path: Path,
    esm2_genes: set,
    wikicrow_genes: set,
    copath_genes: set,
) -> dict:
    """Print per-dataset coverage report and return counts for plotting."""
    df = pd.read_csv(csv_path)
    perts = set(df["pert"].unique())
    total = len(perts)
    print(f"\n{'='*55}")
    print(f"  {dataset_name}  ({total} unique perturbations)")
    print(f"{'='*55}")

    counts = {"dataset": dataset_name, "total": total}
    for label, ref_genes, key in [
        ("ESM2 embedding",     esm2_genes,    "esm2"),
        ("WikiCrow embedding", wikicrow_genes, "wikicrow"),
        ("Co-path neighbors",  copath_genes,  "copath"),
    ]:
        missing = perts - ref_genes
        covered = total - len(missing)
        pct     = covered / total * 100
        status  = "FULL" if not missing else f"PARTIAL ({pct:.1f}%)"
        print(f"  {label:<24} {status}")
        if missing:
            print(f"    Missing ({len(missing)}): {sorted(missing)}")
        counts[f"{key}_covered"] = covered
        counts[f"{key}_missing"] = len(missing)

    # GenePT covers all perturbations
    print(f"  {'GenePT embedding':<24} FULL (assumed)")
    counts["genept_covered"] = total
    counts["genept_missing"] = 0

    return counts


def plot_summary(rows: list[dict], out_path: Path) -> None:
    datasets  = [r["dataset"] for r in rows]
    n         = len(datasets)
    modes     = ["GenePT", "ESM2", "WikiCrow", "Co-path"]
    mode_key  = {"GenePT": "genept", "ESM2": "esm2",
                 "WikiCrow": "wikicrow", "Co-path": "copath"}

    bar_w     = 0.20
    gap       = 0.03
    step      = bar_w + gap          # 0.23
    group_gap = 1.2
    x_centres = [i * group_gap for i in range(n)]
    # 4 bars centred on x_centre: positions at -1.5, -0.5, +0.5, +1.5 steps
    offsets = [(-1.5 + i) * step for i in range(4)]

    COLOR_COVERED = "#4C9A8E"
    COLOR_MISSING = "#E07B6A"
    COLOR_ANNOT   = "white"
    COLOR_PCT     = "#333333"

    max_total = max(r["total"] for r in rows)

    fig, ax = plt.subplots(figsize=(11, 8))

    for mode_i, (mode, offset) in enumerate(zip(modes, offsets)):
        key = mode_key[mode]
        for di, (row, xc) in enumerate(zip(rows, x_centres)):
            x       = xc + offset
            covered = row[f"{key}_covered"]
            missing = row[f"{key}_missing"]
            total   = row["total"]

            ax.bar(x, covered, width=bar_w, color=COLOR_COVERED,
                   zorder=2, linewidth=0)
            ax.bar(x, missing, width=bar_w, bottom=covered,
                   color=COLOR_MISSING, zorder=2, linewidth=0)

            if missing > 0:
                ax.text(x, covered + missing / 2, str(missing),
                        ha="center", va="center",
                        fontsize=10, fontweight="bold", color=COLOR_ANNOT,
                        zorder=3, clip_on=False)

            pct = covered / total * 100
            ax.text(x, total + max_total * 0.02, f"{pct:.1f}%",
                    ha="center", va="bottom",
                    fontsize=7.0, color=COLOR_PCT, zorder=3, clip_on=False)

    # ── two-level x-axis labels ──────────────────────────────────────────────
    from matplotlib.transforms import blended_transform_factory
    sub_tf = blended_transform_factory(ax.transData, ax.transAxes)

    ax.set_xticks([])
    ax.tick_params(bottom=False)

    for mode, offset in zip(modes, offsets):
        for xc in x_centres:
            ax.text(xc + offset, -0.04, mode,
                    ha="center", va="top", fontsize=8, fontweight="bold", color="#555555",
                    transform=sub_tf, rotation=25, rotation_mode="anchor")

    for name, xc in zip(datasets, x_centres):
        ax.text(xc, -0.13, name,
                ha="center", va="top", fontsize=11, fontweight="bold",
                transform=sub_tf)

    ax.set_ylim(0, max_total * 1.12)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))
    ax.set_ylabel("Number of perturbations", fontsize=11, fontweight="bold")
    ax.set_xlabel("Dataset", fontsize=11, fontweight="bold", labelpad=52)
    ax.set_title("Perturbation coverage by embedding type", fontsize=13,
                 fontweight="bold", pad=14)
    ax.set_xlim(x_centres[0] - 0.75, x_centres[-1] + 0.75)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
        label.set_fontsize(10)

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
    plt.subplots_adjust(bottom=0.22)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved → {out_path}")


def main():
    print("Loading ESM2 embeddings ...")
    esm2_genes = load_pkl_genes(ESM2_PKL)
    print(f"  ESM2 genes: {len(esm2_genes)}")

    print("Loading WikiCrow embeddings ...")
    wikicrow_genes = load_pkl_genes(WIKICROW_PKL)
    print(f"  WikiCrow genes: {len(wikicrow_genes)}")

    print("Loading co-path neighbors ...")
    copath_genes = load_copath_genes(COPATH_JSON)
    print(f"  Co-path genes: {len(copath_genes)}")

    rows = []
    for name, path in DATASETS.items():
        rows.append(check_coverage(name, path, esm2_genes, wikicrow_genes, copath_genes))

    print()
    plot_summary(rows, PLOT_OUT)


if __name__ == "__main__":
    main()
