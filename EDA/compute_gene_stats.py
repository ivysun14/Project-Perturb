"""
compute_gene_stats.py
=====================
Compute per-gene expression statistics from h5ad objects after log TP10k normalization.
Statistics are computed separately for control cells and all perturbed cells (pooled, i.e. marginalizing out perturbations).

Usage:
    python compute_gene_stats.py --dataset k562
    python compute_gene_stats.py --dataset hepg2

Output → results/gene_stats_{dataset}.csv with columns:
    gene, ctrl_mean, ctrl_var, ctrl_zero_prop,
    ctrl_q25, ctrl_q50, ctrl_q75, ctrl_q90, ctrl_q99,
    pert_mean, pert_var, pert_zero_prop,
    pert_q25, pert_q50, pert_q75, pert_q90, pert_q99
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy import sparse
sys.path.insert(0, os.path.dirname(__file__))
from config import DATASETS, RESULTS_DIR


def load_adata(cfg: dict) -> ad.AnnData:
    adata = ad.read_h5ad(cfg['h5ad'])
    sc.pp.normalize_total(adata, target_sum=1e4, key_added='size_factors')
    sc.pp.log1p(adata)
    return adata


def gene_summary_stats(X: np.ndarray) -> dict:
    """
    Compute per-gene summary statistics over a (cells × genes) dense matrix.
    Returns a dict of 1-D arrays, each of length n_genes.
    """
    return {
        'mean':      X.mean(axis=0),
        'var':       X.var(axis=0),
        'zero_prop': (X == 0).mean(axis=0),
        'q25':       np.percentile(X, 25, axis=0),
        'q50':       np.percentile(X, 50, axis=0),
        'q75':       np.percentile(X, 75, axis=0),
        'q90':       np.percentile(X, 90, axis=0),
        'q99':       np.percentile(X, 99, axis=0),
    }


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Compute per-gene expression statistics from h5ad')
    parser.add_argument('--dataset', required=True, choices=list(DATASETS.keys()))
    args = parser.parse_args()

    cfg = DATASETS[args.dataset]
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Loading {cfg['display_name']} h5ad...")
    adata = load_adata(cfg)
    print(f"  {adata.n_obs} cells × {adata.n_vars} genes")

    gene_names = adata.var[cfg['gene_name_col']].tolist()
    ctrl_mask = (adata.obs[cfg['target_col']] == cfg['control_key']).values

    # Densify once — needed for percentile computation
    X = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
    X_ctrl = X[ctrl_mask]
    X_pert = X[~ctrl_mask]
    print(f"  Control cells: {X_ctrl.shape[0]:,}  |  Perturbed cells: {X_pert.shape[0]:,}")
    del adata  # free memory before heavy computation

    print("Computing per-gene statistics in control cells...")
    ctrl = gene_summary_stats(X_ctrl)

    # Pooled across all perturbations — mirrors control stats for direct comparison
    print("Computing per-gene statistics in perturbed cells (pooled)...")
    pert = gene_summary_stats(X_pert)

    stats = pd.DataFrame({
        'gene':            gene_names,
        'ctrl_mean':       ctrl['mean'],
        'ctrl_var':        ctrl['var'],
        'ctrl_zero_prop':  ctrl['zero_prop'],
        'ctrl_q25':        ctrl['q25'],
        'ctrl_q50':        ctrl['q50'],
        'ctrl_q75':        ctrl['q75'],
        'ctrl_q90':        ctrl['q90'],
        'ctrl_q99':        ctrl['q99'],
        'pert_mean':       pert['mean'],
        'pert_var':        pert['var'],
        'pert_zero_prop':  pert['zero_prop'],
        'pert_q25':        pert['q25'],
        'pert_q50':        pert['q50'],
        'pert_q75':        pert['q75'],
        'pert_q90':        pert['q90'],
        'pert_q99':        pert['q99'],
    })

    out_path = os.path.join(RESULTS_DIR, f'gene_stats_{args.dataset}.csv')
    stats.to_csv(out_path, index=False)
    print(f"Saved {len(stats):,} gene stats → {out_path}")
