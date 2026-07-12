"""
compute_de_profile_stats.py
===========================
Compute per-gene and per-perturbation DE statistics from DE_DOC CSV files.
Works directly on the precomputed logFC values — no h5ad loading required.

Usage:
    python compute_de_profile_stats.py --dataset k562

Outputs → results/:
    gene_de_stats_{dataset}.csv   — one row per gene
    pert_de_stats_{dataset}.csv   — one row per perturbation

Per-gene columns:
    gene, n_perts, n_de, n_up, n_down, de_rate,
    mean_logFC, var_logFC, mean_abs_logFC, q25_abs_logFC, q50_abs_logFC, q75_abs_logFC, q90_abs_logFC

Per-perturbation columns:
    pert, n_genes, n_de, n_up, n_down, de_rate,
    mean_logFC, var_logFC, mean_abs_logFC, q25_abs_logFC, q50_abs_logFC, q75_abs_logFC, q90_abs_logFC
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from config import DATASETS, RESULTS_DIR


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Compute per-gene and per-pert DE statistics from DE_DOC CSV')
    parser.add_argument('--dataset', required=True, choices=list(DATASETS.keys()))
    args = parser.parse_args()

    cfg = DATASETS[args.dataset]
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Loading DE CSV for {cfg['display_name']}...")
    df = pd.read_csv(cfg['de_csv'])
    print(f"  {len(df):,} pairs  |  {df['pert'].nunique():,} perturbations  |  {df['gene'].nunique():,} genes")

    # --- Per-gene statistics ---
    # Aggregated across all perturbations that tested each gene.
    # mean_logFC near zero means the gene is consistently non-responsive;
    # high var_logFC means its response is perturbation-specific.
    print("Computing per-gene DE statistics...")
    gene_stats = (
        df.groupby('gene')
        .agg(
            n_perts        = ('pert',   'count'),
            n_de           = ('label',  lambda x: (x != 0).sum()),
            n_up           = ('label',  lambda x: (x == 1).sum()),
            n_down         = ('label',  lambda x: (x == 2).sum()),
            mean_logFC     = ('logFC',  'mean'),
            var_logFC      = ('logFC',  'var'),
            mean_abs_logFC = ('logFC',  lambda x: x.abs().mean()),
            q25_abs_logFC  = ('logFC',  lambda x: x.abs().quantile(0.25)),
            q50_abs_logFC  = ('logFC',  lambda x: x.abs().quantile(0.50)),
            q75_abs_logFC  = ('logFC',  lambda x: x.abs().quantile(0.75)),
            q90_abs_logFC  = ('logFC',  lambda x: x.abs().quantile(0.90)),
        )
        .reset_index()
    )
    gene_stats['de_rate'] = gene_stats['n_de'] / gene_stats['n_perts']
    gene_stats = gene_stats[['gene', 'n_perts', 'n_de', 'n_up', 'n_down', 'de_rate',
                              'mean_logFC', 'var_logFC', 'mean_abs_logFC',
                              'q25_abs_logFC', 'q50_abs_logFC', 'q75_abs_logFC', 'q90_abs_logFC']]
    out_gene = os.path.join(RESULTS_DIR, f'gene_de_stats_{args.dataset}.csv')
    gene_stats.to_csv(out_gene, index=False)
    print(f"  Saved {len(gene_stats):,} gene rows → {out_gene}")

    # --- Per-perturbation statistics ---
    # n_de and de_rate reveal perturbation strength (e.g. essential genes vs. weak perturbations).
    # var_logFC captures how broadly the perturbation reshapes the transcriptome.
    print("Computing per-perturbation DE statistics...")
    pert_stats = (
        df.groupby('pert')
        .agg(
            n_genes        = ('gene',   'count'),
            n_de           = ('label',  lambda x: (x != 0).sum()),
            n_up           = ('label',  lambda x: (x == 1).sum()),
            n_down         = ('label',  lambda x: (x == 2).sum()),
            mean_logFC     = ('logFC',  'mean'),
            var_logFC      = ('logFC',  'var'),
            mean_abs_logFC = ('logFC',  lambda x: x.abs().mean()),
            q25_abs_logFC  = ('logFC',  lambda x: x.abs().quantile(0.25)),
            q50_abs_logFC  = ('logFC',  lambda x: x.abs().quantile(0.50)),
            q75_abs_logFC  = ('logFC',  lambda x: x.abs().quantile(0.75)),
            q90_abs_logFC  = ('logFC',  lambda x: x.abs().quantile(0.90)),
        )
        .reset_index()
    )
    pert_stats['de_rate'] = pert_stats['n_de'] / pert_stats['n_genes']
    pert_stats = pert_stats[['pert', 'n_genes', 'n_de', 'n_up', 'n_down', 'de_rate',
                              'mean_logFC', 'var_logFC', 'mean_abs_logFC',
                              'q25_abs_logFC', 'q50_abs_logFC', 'q75_abs_logFC', 'q90_abs_logFC']]
    out_pert = os.path.join(RESULTS_DIR, f'pert_de_stats_{args.dataset}.csv')
    pert_stats.to_csv(out_pert, index=False)
    print(f"  Saved {len(pert_stats):,} perturbation rows → {out_pert}")

    # Quick summary
    print(f"\nSummary for {cfg['display_name']}:")
    print(f"  Global DE rate:            {gene_stats['de_rate'].mean():.3f}")
    print(f"  Median per-gene DE rate:   {gene_stats['de_rate'].median():.3f}")
    print(f"  Median per-pert n_de:      {pert_stats['n_de'].median():.0f}")
    print(f"  Up / Down ratio:           {df['label'].eq(1).sum()} / {df['label'].eq(2).sum()}")
