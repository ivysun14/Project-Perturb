"""
plot_de_distributions.py
========================
Visualize per-gene and per-perturbation DE statistics produced by
compute_de_profile_stats.py.

Figures (saved to figures/):
  1. gene_logfc_distributions.pdf
       Left: distribution of per-gene mean logFC (across all perturbations).
       Right: distribution of per-gene mean |logFC|.
       Together reveal whether perturbation effects are symmetric and how large
       the typical effect on any gene is.

  2. gene_de_rate_distribution.pdf
       Distribution of per-gene DE rates (fraction of perturbations that render
       the gene DE). Most genes should have very low DE rate — the shape of the
       tail informs how many genes are broadly responsive.

  3. gene_up_down_asymmetry.pdf
       Scatter of n_up vs n_down per gene (separate panels per dataset).
       Asymmetry off the diagonal = systematic directional bias for that gene.

  4. pert_de_distributions.pdf
       Left: distribution of number of DE genes per perturbation.
       Right: distribution of per-perturbation DE rate.
       Reveals heterogeneity in perturbation strength.

Usage:
    python plot_de_distributions.py
    (Requires results/gene_de_stats_{dataset}.csv and pert_de_stats_{dataset}.csv.)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from config import DATASETS, RESULTS_DIR, FIGURES_DIR, DATASET_COLORS

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

os.makedirs(FIGURES_DIR, exist_ok=True)


def _load(prefix: str) -> dict:
    stats = {}
    for ds in DATASETS:
        path = os.path.join(RESULTS_DIR, f'{prefix}_{ds}.csv')
        if os.path.exists(path):
            stats[ds] = pd.read_csv(path)
            print(f"  Loaded {prefix} {ds}: {len(stats[ds]):,} rows")
        else:
            print(f"  Warning: {path} not found — skipping {ds}")
    return stats


def plot_gene_logfc_distributions(gene_stats: dict):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ds, df in gene_stats.items():
        kw = dict(bins=100, density=True, alpha=0.55, color=DATASET_COLORS[ds],
                  label=DATASETS[ds]['display_name'], histtype='stepfilled', linewidth=0.5)
        axes[0].hist(df['mean_logFC'],     **kw)
        axes[1].hist(df['mean_abs_logFC'], **kw)

    axes[0].axvline(0, color='k', lw=1, linestyle='--', alpha=0.6)
    axes[0].set_xlabel('Per-gene mean logFC (across all perturbations)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Density', fontsize=12, fontweight='bold')
    axes[0].set_title('Per-gene Mean Log Fold Change', fontsize=13, fontweight='bold')

    axes[1].set_xlabel('Per-gene mean |logFC|', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Density', fontsize=12, fontweight='bold')
    axes[1].set_title('Per-gene Mean Absolute Log Fold Change', fontsize=13, fontweight='bold')

    for ax in axes:
        ax.legend(fontsize=10, frameon=False)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'gene_logfc_distributions.pdf')
    plt.savefig(out, bbox_inches='tight', dpi=400)
    print(f"Saved: {out}")
    plt.close()


def plot_gene_de_rate(gene_stats: dict):
    """
    Heavy concentration near zero is expected — most genes are not DE for most perturbations.
    The right tail shows hub genes that are broadly responsive.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for ds, df in gene_stats.items():
        ax.hist(
            df['de_rate'], bins=60, density=True, alpha=0.55,
            color=DATASET_COLORS[ds], label=DATASETS[ds]['display_name'],
            histtype='stepfilled', linewidth=0.5,
        )

    ax.set_xlabel('Per-gene DE rate (fraction of perturbations)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Per-gene DE Rates', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, frameon=False)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'gene_de_rate_distribution.pdf')
    plt.savefig(out, bbox_inches='tight', dpi=400)
    print(f"Saved: {out}")
    plt.close()


def plot_gene_up_down_asymmetry(gene_stats: dict):
    """
    Points well above the diagonal: gene tends to be up-regulated.
    Points well below: gene tends to be down-regulated.
    Tight clustering on the diagonal: symmetric response.
    """
    n = len(gene_stats)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (ds, df) in zip(axes, gene_stats.items()):
        ax.scatter(df['n_up'], df['n_down'], s=3, alpha=0.25,
                   color=DATASET_COLORS[ds], rasterized=True)
        lim = max(df['n_up'].max(), df['n_down'].max()) * 1.02
        ax.plot([0, lim], [0, lim], 'k--', lw=1, alpha=0.6)
        ax.set_xlabel('# Perturbations where gene is up-regulated', fontsize=11, fontweight='bold')
        ax.set_ylabel('# Perturbations where gene is down-regulated', fontsize=11, fontweight='bold')
        ax.set_title(f'Up vs Down Regulation per Gene\n{DATASETS[ds]["display_name"]}', fontsize=13, fontweight='bold')

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'gene_up_down_asymmetry.pdf')
    plt.savefig(out, bbox_inches='tight', dpi=400)
    print(f"Saved: {out}")
    plt.close()


def plot_pert_de_distributions(pert_stats: dict):
    """
    Heterogeneity in n_de reveals that perturbations vary widely in strength —
    important context for calibration since a fixed threshold will over- or
    under-call DE for perturbations at the extremes.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ds, df in pert_stats.items():
        kw = dict(bins=60, density=True, alpha=0.55, color=DATASET_COLORS[ds],
                  label=DATASETS[ds]['display_name'], histtype='stepfilled', linewidth=0.5)
        axes[0].hist(df['n_de'],      **kw)
        axes[1].hist(df['de_rate'],   **kw)

    axes[0].set_xlabel('Number of DE genes per perturbation', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Density', fontsize=12, fontweight='bold')
    axes[0].set_title('Distribution of DE Gene Counts\nper Perturbation', fontsize=13, fontweight='bold')

    axes[1].set_xlabel('Per-perturbation DE rate', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Density', fontsize=12, fontweight='bold')
    axes[1].set_title('Distribution of DE Rates\nper Perturbation', fontsize=13, fontweight='bold')

    for ax in axes:
        ax.legend(fontsize=10, frameon=False)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'pert_de_distributions.pdf')
    plt.savefig(out, bbox_inches='tight', dpi=400)
    print(f"Saved: {out}")
    plt.close()


if __name__ == '__main__':
    print("Loading gene DE stats...")
    gene_stats = _load('gene_de_stats')
    print("Loading perturbation DE stats...")
    pert_stats = _load('pert_de_stats')

    if not gene_stats:
        print("No DE stats found. Run compute_de_profile_stats.py --dataset <name> first.")
    else:
        plot_gene_logfc_distributions(gene_stats)
        plot_gene_de_rate(gene_stats)
        plot_gene_up_down_asymmetry(gene_stats)

    if pert_stats:
        plot_pert_de_distributions(pert_stats)

    print("Done.")
