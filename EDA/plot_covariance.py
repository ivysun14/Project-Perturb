"""
plot_covariance.py
==================
Visualize gene–gene and perturbation–perturbation covariance matrices produced
by compute_covariance.py. Covariance is converted to Pearson correlation for
display so the color scale is interpretable across datasets.

Figures (saved to figures/, vector PDF with raster elements at 400 dpi):
  gene_covariance_{dataset}.pdf
      Correlation heatmap of the top-N most variable genes in control cells,
      with rows/columns ordered by hierarchical clustering (average linkage).

  pert_covariance_{dataset}.pdf
      Correlation heatmap of all perturbations based on their logFC profiles,
      with rows/columns ordered by hierarchical clustering.
      A block structure indicates groups of perturbations with similar effects.

Usage:
    python plot_covariance.py --dataset k562
    python plot_covariance.py --dataset hepg2
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

sys.path.insert(0, os.path.dirname(__file__))
from config import DATASETS, RESULTS_DIR, FIGURES_DIR

plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 11})
os.makedirs(FIGURES_DIR, exist_ok=True)


def cov_to_corr(cov: np.ndarray) -> np.ndarray:
    """Convert covariance matrix to Pearson correlation, handling zero-variance entries."""
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Pearson correlation: corr[i,j] = cov[i,j] / (std[i] * std[j]).
    # Clipping std to 1e-12 prevents division-by-zero for zero-variance genes.
    std = np.sqrt(np.diag(cov).clip(min=1e-12))
    
    # np.outer(std, std) builds the (N×N) denominator matrix of all std[i]*std[j] pairs.
    corr = cov / np.outer(std, std)
    
    # Clip to [-1, 1] to correct any floating-point values that drift slightly outside.
    return corr.clip(-1.0, 1.0)


def hierarchical_order(corr: np.ndarray) -> np.ndarray:
    """Return leaf ordering from average-linkage hierarchical clustering of a correlation matrix."""
    # Convert correlation to distance: highly correlated pairs (corr≈1) get distance≈0.
    dist = 1.0 - corr
    
    # Symmetrize and zero the diagonal to fix minor floating-point asymmetries before
    # passing to scipy, which requires a valid symmetric distance matrix.
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    
    # squareform converts the (N×N) distance matrix to the condensed upper-triangle
    # vector that scipy's linkage expects.
    condensed = squareform(dist, checks=False)
    
    # Average-linkage merges clusters based on the mean distance between all pairs
    # across the two clusters — a good balance between single/complete linkage.
    Z = linkage(condensed, method='average')
    
    # dendrogram extracts the leaf order that places similar entries adjacent;
    # no_plot=True suppresses rendering since we only need the index array.
    return np.array(dendrogram(Z, no_plot=True)['leaves'])


def plot_heatmap(corr: np.ndarray, labels: np.ndarray, title: str, out_path: str):
    """
    Draw a correlation heatmap with hierarchically clustered rows/columns.
    Labels are only printed if there are ≤ 60 entries — otherwise tick marks are hidden.
    """
    order = hierarchical_order(corr)
    corr_ord = corr[np.ix_(order, order)]
    labels_ord = labels[order]
    n = len(labels_ord)

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr_ord, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Pearson correlation', fontsize=11)

    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)

    if n <= 60:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels_ord, rotation=90, fontsize=6)
        ax.set_yticklabels(labels_ord, fontsize=6)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f'{n} entries (hierarchically clustered)', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'{n} entries (hierarchically clustered)', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=400)
    print(f"  Saved: {out_path}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot gene-gene and pert-pert covariance heatmaps')
    parser.add_argument('--dataset', required=True, choices=list(DATASETS.keys()))
    args = parser.parse_args()

    ds   = args.dataset
    name = DATASETS[ds]['display_name']

    # --- Gene–gene ---
    gene_path = os.path.join(RESULTS_DIR, f'gene_covariance_{ds}.npz')
    if os.path.exists(gene_path):
        d = np.load(gene_path, allow_pickle=True)
        corr  = cov_to_corr(d['cov'])
        genes = d['genes']
        n     = len(genes)
        plot_heatmap(
            corr, genes,
            title=f'Gene–Gene Correlation (top {n} variable genes)\n{name} — Control cells',
            out_path=os.path.join(FIGURES_DIR, f'gene_covariance_{ds}.pdf'),
        )
    else:
        print(f"Not found: {gene_path} — skipping gene covariance plot")

    # --- Perturbation–perturbation ---
    pert_path = os.path.join(RESULTS_DIR, f'pert_covariance_{ds}.npz')
    if os.path.exists(pert_path):
        d = np.load(pert_path, allow_pickle=True)
        corr  = cov_to_corr(d['cov'])
        perts = d['perts']
        P     = len(perts)
        plot_heatmap(
            corr, perts,
            title=f'Perturbation–Perturbation Correlation (logFC profiles)\n{name} — {P} perturbations',
            out_path=os.path.join(FIGURES_DIR, f'pert_covariance_{ds}.pdf'),
        )
    else:
        print(f"Not found: {pert_path} — skipping perturbation covariance plot")
