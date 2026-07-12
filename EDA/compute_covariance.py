"""
compute_covariance.py
=====================
Compute two complementary covariance structures on the GROUND TRUTH H5AD OBJECTS:

  1. Gene–gene covariance (from h5ad control cells, log TP10k normalized).
     Restricted to the top-N most variable genes to keep the matrix tractable.
     Captures baseline co-expression before any perturbation.

  2. Perturbation–perturbation covariance (from DE_DOC CSV logFC profiles).
     Each perturbation is represented as a vector of logFC values across all genes.
     Captures how similar perturbations are in their transcriptome-wide effects.

Usage:
    python compute_covariance.py --dataset k562
    python compute_covariance.py --dataset hepg2 --n_top_genes 300

Outputs → results/:
    gene_covariance_{dataset}.npz   — keys: 'cov' (N×N), 'genes' (N,)
    pert_covariance_{dataset}.npz   — keys: 'cov' (P×P), 'perts' (P,)
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Compute gene-gene and pert-pert covariance matrices')
    parser.add_argument('--dataset', required=True, choices=list(DATASETS.keys()))
    parser.add_argument('--n_top_genes', type=int, default=500, help='Top-N most variable genes for gene-gene covariance (default: 500)')
    args = parser.parse_args()

    cfg = DATASETS[args.dataset]
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Gene–gene covariance in control cells
    # ------------------------------------------------------------------
    print(f"Loading {cfg['display_name']} h5ad for gene–gene covariance...")
    adata = load_adata(cfg)
    # Use only non-targeting (control) cells to capture baseline co-expression
    # uncontaminated by any perturbation effect.
    ctrl_mask  = (adata.obs[cfg['target_col']] == cfg['control_key']).values
    gene_names = np.array(adata.var[cfg['gene_name_col']].tolist())

    X_ctrl = adata.X[ctrl_mask]
    # AnnData stores expression as a sparse matrix; np.cov requires a dense array.
    if sparse.issparse(X_ctrl):
        X_ctrl = X_ctrl.toarray()
    print(f"  Control cells: {X_ctrl.shape[0]:,}  |  Total genes: {X_ctrl.shape[1]:,}")

    # Select top-N by variance so the (N×N) matrix stays tractable.
    gene_vars = X_ctrl.var(axis=0)
    # argsort gives ascending order; take the last n_top_genes indices (highest
    # variance) then reverse so index 0 = most variable gene.
    top_idx   = np.argsort(gene_vars)[-args.n_top_genes:][::-1]
    top_genes = gene_names[top_idx]
    X_top     = X_ctrl[:, top_idx]
    print(f"  Computing {args.n_top_genes}×{args.n_top_genes} gene–gene covariance...")
    
    # np.cov expects shape (variables, observations), so we transpose from
    # (cells × genes) → (genes × cells) to get an (N × N) gene covariance matrix.
    gene_cov = np.cov(X_top.T)
    out_gene = os.path.join(RESULTS_DIR, f'gene_covariance_{args.dataset}.npz')
    
    # Saves a .npz archive with two arrays: 'cov' (N×N float64 covariance matrix)
    # and 'genes' (N,) string array of gene names in the same order as matrix rows/cols.
    # Load later with: data = np.load(path); cov = data['cov']; genes = data['genes']
    np.savez(out_gene, cov=gene_cov, genes=top_genes)
    print(f"  Saved → {out_gene}")
    del adata, X_ctrl, X_top  # free memory before loading the DE CSV

    # ------------------------------------------------------------------
    # 2. Perturbation–perturbation covariance via logFC profiles
    # ------------------------------------------------------------------
    print(f"\nLoading DE CSV for {cfg['display_name']} perturbation–perturbation covariance...")
    df = pd.read_csv(cfg['de_csv'])

    # Pivot to (perturbations × genes) logFC matrix.
    # Missing pairs (if any) are filled with 0 (non-DE / no shift).
    logfc_mat  = df.pivot_table(index='pert', columns='gene', values='logFC', fill_value=0.0)
    perts      = np.array(logfc_mat.index.tolist())
    P, G       = logfc_mat.shape
    print(f"  logFC matrix: {P} perturbations × {G} genes")

    print(f"  Computing {P}×{P} perturbation–perturbation covariance...")
    # np.cov with shape (P, G) treats each perturbation (row) as a variable
    # and each gene (column) as an observation → output is (P, P)
    pert_cov = np.cov(logfc_mat.values)

    out_pert = os.path.join(RESULTS_DIR, f'pert_covariance_{args.dataset}.npz')
    np.savez(out_pert, cov=pert_cov, perts=perts)
    print(f"  Saved → {out_pert}")
