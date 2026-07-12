"""
plot_expression_distributions.py
=================================
Visualize per-gene expression statistics produced by compute_gene_stats.py.

Figures (saved to figures/):
  1. gene_marginal_distributions.pdf
       Distribution of per-gene mean, variance, and zero proportion in control
       cells — overlaid across all four datasets.

  2. gene_mean_variance.pdf
       Mean–variance scatter per dataset (one panel each).
       Reveals whether the dataset follows typical scRNA-seq overdispersion.

  3. gene_ctrl_vs_pert_mean.pdf
       Control vs pooled-perturbed per-gene mean — identity line included.
       Deviations indicate global expression shifts from perturbation.

  4. gene_ctrl_vs_pert_distributions.pdf
       Side-by-side control/perturbed histograms of mean, variance, and zero
       proportion per dataset. Reveals distributional shifts across all three
       statistics simultaneously.

Usage:
    python plot_expression_distributions.py
    (Requires results/gene_stats_{dataset}.csv for each dataset.)
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


def load_all_gene_stats() -> dict:
    stats = {}
    for ds in DATASETS:
        path = os.path.join(RESULTS_DIR, f'gene_stats_{ds}.csv')
        if os.path.exists(path):
            stats[ds] = pd.read_csv(path)
            print(f"  Loaded {ds}: {len(stats[ds]):,} genes")
        else:
            print(f"  Warning: {path} not found — skipping {ds}")
    return stats


def plot_marginal_distributions(stats: dict):
    """
    Per-gene mean, variance, and zero proportion — one figure per dataset.
    """
    for ds, df in stats.items():
        color = DATASET_COLORS[ds]
        name  = DATASETS[ds]['display_name']
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        kw = dict(density=True, alpha=0.7, color=color, histtype='stepfilled', linewidth=0.5)

        axes[0].hist(df['ctrl_mean'],      bins=100, **kw)
        axes[1].hist(df['ctrl_var'],       bins=100, **kw)
        axes[2].hist(df['ctrl_zero_prop'], bins=60,  **kw)

        axes[0].set_xlabel('Mean expression', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Density', fontsize=12, fontweight='bold')
        axes[0].set_title(f'Per-gene Mean Expression\n{name} — Control cells', fontsize=13, fontweight='bold')

        axes[1].set_xlabel('Variance', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Density', fontsize=12, fontweight='bold')
        axes[1].set_title(f'Per-gene Variance\n{name} — Control cells', fontsize=13, fontweight='bold')

        axes[2].set_xlabel('Zero proportion', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Density', fontsize=12, fontweight='bold')
        axes[2].set_title(f'Per-gene Zero Proportion\n{name} — Control cells', fontsize=13, fontweight='bold')

        for ax in axes:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        plt.tight_layout()
        out = os.path.join(FIGURES_DIR, f'gene_marginal_distributions_{ds}.pdf')
        plt.savefig(out, bbox_inches='tight', dpi=400)
        print(f"Saved: {out}")
        plt.close()


def plot_mean_variance(stats: dict):
    """
    Mean–variance scatter for each dataset (separate panels).
    scRNA-seq data typically shows a Poisson-like floor (var ≈ mean)
    and overdispersion for highly expressed genes.
    """
    n = len(stats)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (ds, df) in zip(axes, stats.items()):
        ax.scatter(
            df['ctrl_mean'], df['ctrl_var'],
            s=3, alpha=0.25, color=DATASET_COLORS[ds], rasterized=True,
        )
        x_ref = np.linspace(0, df['ctrl_mean'].max(), 200)
        ax.plot(x_ref, x_ref, 'k--', lw=1, alpha=0.6, label='Poisson (var = mean)')
        ax.set_xlabel('Mean expression', fontsize=12, fontweight='bold')
        ax.set_ylabel('Variance', fontsize=12, fontweight='bold')
        ax.set_title(f'Mean–Variance Relationship\n{DATASETS[ds]["display_name"]} — Control cells', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, frameon=False)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'gene_mean_variance.pdf')
    plt.savefig(out, bbox_inches='tight', dpi=400)
    print(f"Saved: {out}")
    plt.close()


def plot_ctrl_vs_pert_mean(stats: dict):
    """
    Control mean vs pooled-perturbed mean per gene (separate panels).
    Points above the identity line are genes that are globally up-regulated
    in perturbed cells; below indicates global down-regulation.
    Note: this pools ALL perturbations, so only consistent global shifts will appear.
    """
    n = len(stats)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (ds, df) in zip(axes, stats.items()):
        ax.scatter(
            df['ctrl_mean'], df['pert_mean'],
            s=3, alpha=0.2, color=DATASET_COLORS[ds], rasterized=True,
        )
        lim = max(df['ctrl_mean'].max(), df['pert_mean'].max()) * 1.02
        ax.plot([0, lim], [0, lim], 'k--', lw=1, alpha=0.6, label='y = x')
        ax.set_xlabel('Per-gene mean in control cells', fontsize=12, fontweight='bold')
        ax.set_ylabel('Per-gene mean in perturbed cells (pooled)', fontsize=12, fontweight='bold')
        ax.set_title(f'Control vs Perturbed Mean Expression\n{DATASETS[ds]["display_name"]}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, frameon=False)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'gene_ctrl_vs_pert_mean.pdf')
    plt.savefig(out, bbox_inches='tight', dpi=400)
    print(f"Saved: {out}")
    plt.close()


def plot_ctrl_vs_pert_distributions(stats: dict):
    """
    For each dataset: overlaid control vs perturbed histograms of mean, variance,
    and zero proportion. Distributional shift between the two conditions is more
    informative than the scalar mean-shift alone.
    """
    n = len(stats)
    fig, axes = plt.subplots(n, 3, figsize=(15, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, (ds, df) in enumerate(stats.items()):
        color = DATASET_COLORS[ds]
        name  = DATASETS[ds]['display_name']
        kw_ctrl = dict(bins=100, density=True, alpha=0.55, histtype='stepfilled',
                       linewidth=0.5, label='Control')
        kw_pert = dict(bins=100, density=True, alpha=0.55, histtype='stepfilled',
                       linewidth=0.5, label='Perturbed (pooled)')

        axes[row, 0].hist(df['ctrl_mean'],      color=color,  **kw_ctrl)
        axes[row, 0].hist(df['pert_mean'],      color='grey', **kw_pert)
        axes[row, 0].set_xlabel('Mean expression (log TP10k)', fontsize=11, fontweight='bold')
        axes[row, 0].set_ylabel('Density', fontsize=11, fontweight='bold')
        axes[row, 0].set_title(f'{name} — Mean Expression', fontsize=12, fontweight='bold')
        axes[row, 0].legend(fontsize=9, frameon=False)

        axes[row, 1].hist(df['ctrl_var'],       color=color,  **kw_ctrl)
        axes[row, 1].hist(df['pert_var'],       color='grey', **kw_pert)
        axes[row, 1].set_xlabel('Variance', fontsize=11, fontweight='bold')
        axes[row, 1].set_ylabel('Density', fontsize=11, fontweight='bold')
        axes[row, 1].set_title(f'{name} — Variance', fontsize=12, fontweight='bold')
        axes[row, 1].legend(fontsize=9, frameon=False)

        axes[row, 2].hist(df['ctrl_zero_prop'], color=color,  **{**kw_ctrl, 'bins': 60})
        axes[row, 2].hist(df['pert_zero_prop'], color='grey', **{**kw_pert, 'bins': 60})
        axes[row, 2].set_xlabel('Zero proportion', fontsize=11, fontweight='bold')
        axes[row, 2].set_ylabel('Density', fontsize=11, fontweight='bold')
        axes[row, 2].set_title(f'{name} — Zero Proportion', fontsize=12, fontweight='bold')
        axes[row, 2].legend(fontsize=9, frameon=False)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'gene_ctrl_vs_pert_distributions.pdf')
    plt.savefig(out, bbox_inches='tight', dpi=400)
    print(f"Saved: {out}")
    plt.close()


def plot_ctrl_vs_pert_distributions_violin(stats: dict):
    """
    Same comparison as plot_ctrl_vs_pert_distributions but as split violins.
    Each panel shows one statistic; each dataset gets its own row.
    Violins are split (control left, perturbed right) to directly compare shape.
    """
    metrics = [
        ('ctrl_mean',      'pert_mean',      'Mean expression (log TP10k)'),
        ('ctrl_var',       'pert_var',        'Variance'),
        ('ctrl_zero_prop', 'pert_zero_prop',  'Zero proportion'),
    ]

    n = len(stats)
    fig, axes = plt.subplots(n, 3, figsize=(15, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, (ds, df) in enumerate(stats.items()):
        color = DATASET_COLORS[ds]
        name  = DATASETS[ds]['display_name']

        for col, (ctrl_col, pert_col, xlabel) in enumerate(metrics):
            ax = axes[row, col]

            ctrl_vals = df[ctrl_col].dropna().values
            pert_vals = df[pert_col].dropna().values

            parts = ax.violinplot(
                [ctrl_vals, pert_vals],
                positions=[0, 1],
                showmedians=True,
                showextrema=False,
            )
            # Color control with dataset color, perturbed grey
            parts['bodies'][0].set_facecolor(color)
            parts['bodies'][0].set_alpha(0.7)
            parts['bodies'][1].set_facecolor('grey')
            parts['bodies'][1].set_alpha(0.55)
            parts['cmedians'].set_color('black')
            parts['cmedians'].set_linewidth(1.5)

            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Control', 'Perturbed\n(pooled)'], fontsize=10)
            ax.set_ylabel(xlabel, fontsize=11, fontweight='bold')
            ax.set_title(f'{name} — {xlabel}', fontsize=12, fontweight='bold')

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'gene_ctrl_vs_pert_distributions_violin.pdf')
    plt.savefig(out, bbox_inches='tight', dpi=400)
    print(f"Saved: {out}")
    plt.close()


if __name__ == '__main__':
    print("Loading gene stats...")
    stats = load_all_gene_stats()
    if not stats:
        print("No stats files found. Run compute_gene_stats.py --dataset <name> first.")
    else:
        plot_marginal_distributions(stats)
        plot_mean_variance(stats)
        plot_ctrl_vs_pert_mean(stats)
        plot_ctrl_vs_pert_distributions(stats)
        plot_ctrl_vs_pert_distributions_violin(stats)
        print("Done.")
