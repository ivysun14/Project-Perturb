"""
DE_DOC_stats.py
===============

Given an AnnData object on a gene perturbation experiment, compute the differential 
expression statistics and direction of change for each perturbation-gene pair.

This script calculates:
1. Differential expression using Wilcoxon rank-sum test with BH correction
2. Log fold change (log2FC) between perturbed and control cells
3. Direction of change (DOC) for DE genes: up-regulation (1) or down-regulation (2)

Output CSV format:
- pert: perturbation name
- gene: gene name
- label: 0 (non-DE), 1 (up-regulation), 2 (down-regulation)
- p_value: BH-corrected p-value
- logFC: log fold change (perturbed mean - control mean), base e
- log2FC: log based 2 fold change for better interpretability
"""

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy.stats import ranksums
from statsmodels.stats.multitest import multipletests
from typing import Optional
import argparse

def resolve_repeats(adata, gene_row="gene_name"):
    """
    Take column-wise averages for repeatedly measured genes.

    Args:
        adata: an anndata of cell by gene matrix
        gene_row: adata.var row name representing genes measured
    
    Returns: 
        a new AnnData object with uniquely measured gene columns
    """
    
    genes = adata.var[gene_row]
    groups = {}
    for i, g in enumerate(genes):
        groups.setdefault(g, []).append(i)
    
    # take average of columns with repeated gene name
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    X_avg = np.stack([X[:, idxs].mean(axis=1) for idxs in groups.values()], axis=1)
    
    # build new var and new anndata
    var_new = adata.var.iloc[[idxs[0] for idxs in groups.values()]].copy()
    adata = ad.AnnData(
        X=X_avg,
        obs=adata.obs.copy(),
        var=var_new
    )

    return adata

def compute_de_statistics(
    adata: ad.AnnData,
    control_key: str = 'non-targeting',
    target_col: str = 'target_gene',
    gene_row: str = 'gene_name',
    alpha: float = 0.05,
    log2fc_threshold: float = 0.25,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Compute differential expression statistics for each (perturbation, gene) pair.

    A gene is considered DE if:
    1. BH-corrected p-value < alpha (significant)
    2. |log2FC| >= log2fc_threshold (biologically meaningful change)

    For DE genes, direction is determined by sign of log2FC:
    - label = 1: up-regulated (log2FC > 0)
    - label = 2: down-regulated (log2FC < 0)
    - label = 0: non-DE (fails significance or log2FC threshold)

    Args:
        adata: AnnData object with cells as rows, genes as columns
        control_key: Value indicating control cells in obs[target_col]
        target_col: Column in obs containing perturbation labels
        gene_row: Column in var containing gene names
        alpha: Significance threshold for BH-corrected p-value (default: 0.05)
        log2fc_threshold: Minimum absolute log fold change threshold (default: 0.25)
        save_path: Path to save CSV results (optional)

    Returns:
        DataFrame with columns: pert, gene, label, p_value, logFC, log2FC
    """

    results = []

    # Get control cells expression
    control_mask = (adata.obs[target_col] == control_key).values
    control_expr = adata.X[control_mask]

    # Get unique perturbations
    perturbations = [p for p in adata.obs[target_col].unique() if p != control_key]

    print(f"\nComputing DE statistics for {len(perturbations)} perturbations across {adata.n_vars} genes...")
    print(f"Parameters: alpha={alpha}, log2FC_threshold={log2fc_threshold}")

    # Process each perturbation
    for pert_idx, pert in enumerate(perturbations):

        if (pert_idx + 1) % 10 == 0:
            print(f"  Processing perturbation {pert_idx + 1}/{len(perturbations)}")

        pert_mask = (adata.obs[target_col] == pert).values
        pert_expr = adata.X[pert_mask]

        # Collect statistics for all genes
        pvalues = []
        logfcs = []
        log2fcs = []
        gene_names = []

        for gene_idx, gene_name in enumerate(adata.var[gene_row]):
            control_vals = control_expr[:, gene_idx]
            pert_vals = pert_expr[:, gene_idx]

            # Calculate logFC (mean difference)
            # NOTE: Assume data is natural log-transformed, so this is log fold change
            control_mean = np.mean(control_vals)
            pert_mean = np.mean(pert_vals)
            logfc = pert_mean - control_mean
            log2_fc = logfc / np.log(2)             # Convert for reporting base-2 fold change

            # Perform Wilcoxon rank-sum test
            try:
                statistic, pval = ranksums(pert_vals, control_vals)
                pvalues.append(pval)
                logfcs.append(logfc)
                log2fcs.append(log2_fc)
                gene_names.append(gene_name)
            except:
                # If test fails (e.g., all values identical), assign p=1
                pvalues.append(1.0)
                logfcs.append(logfc)
                log2fcs.append(log2_fc)
                gene_names.append(gene_name)

        # Apply BH correction for this perturbation
        if len(pvalues) > 0:
            reject, pvals_corrected, _, _ = multipletests(pvalues, alpha=alpha, method='fdr_bh')

            # Determine DE labels with both p-value and log2FC thresholds
            for gene_name, is_significant, pval_corrected, logfc, log2_fc in zip(
                gene_names, reject, pvals_corrected, logfcs, log2fcs
            ):
                passes_pval = is_significant  # p_value < alpha after BH correction
                passes_log2fc = abs(log2_fc) >= log2fc_threshold

                if passes_pval and passes_log2fc:
                    # DE gene: determine direction
                    if log2_fc > 0:
                        label = 1  # Up-regulated
                    else:
                        label = 2  # Down-regulated
                else:
                    # Non-DE gene
                    label = 0

                results.append({
                    'pert': pert,
                    'gene': gene_name,
                    'label': label,
                    'p_value': pval_corrected,
                    'logFC': logfc,
                    'log2FC': log2_fc
                })

    # Create DataFrame
    df = pd.DataFrame(results)

    # Summary statistics
    print(f"\nDE computation complete!")
    print(f"Total perturbation-gene pairs: {len(df)}")
    print(f"Non-DE (label=0): {(df['label'] == 0).sum()}")
    print(f"Up-regulated (label=1): {(df['label'] == 1).sum()}")
    print(f"Down-regulated (label=2): {(df['label'] == 2).sum()}")

    # Save to CSV if path provided
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"\nDE results saved to {save_path}")

    return df


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Compute differential expression statistics for perturbation data')
    parser.add_argument('--adata', type=str, required=True, help='Path to AnnData h5ad file')
    parser.add_argument('--output', type=str, required=True, help='Path to save output CSV file')
    parser.add_argument('--control', type=str, default='non-targeting', help='Control perturbation label (default: non-targeting)')
    parser.add_argument('--target_col', type=str, default='target_gene', help='Column with perturbation labels (default: target_gene)')
    parser.add_argument('--gene_row', type=str, default='gene_name', help='Column with gene names (default: gene_name)')
    parser.add_argument('--alpha', type=float, default=0.05, help='Significance threshold for BH-corrected p-value (default: 0.05)')
    parser.add_argument('--log2fc_threshold', type=float, default=0.25, help='Minimum absolute log2 fold change threshold (default: 0.25)')

    args = parser.parse_args()

    # Load AnnData and conduct pre-processing
    print(f"Loading AnnData from {args.adata}...")
    adata = ad.read_h5ad(args.adata)
    print(f"Loaded: {adata.shape[0]} cells x {adata.shape[1]} genes")
    adata.obs.rename(columns={"gene": "target_gene"}, inplace=True)
    adata = resolve_repeats(adata, gene_row="gene_name")
    
    # Use this only when processing hESC dataset
    #if args.gene_row not in adata.var.columns:
    #    # Create gene_name column in var if it doesn't exist
    #    print(f"Creating '{args.gene_row}' column in adata.var from index...")
    #    adata.var[args.gene_row] = adata.var.index

    # sc.pp.normalize_total(adata, target_sum=1e4, key_added="size_factors")
    # sc.pp.log1p(adata)
    # print("Log TP10k normalization completed (matching PerturbQA).")

    # Compute DE statistics together with DOC
    df = compute_de_statistics(
        adata=adata,
        control_key=args.control,
        target_col=args.target_col,
        gene_row=args.gene_row,
        alpha=args.alpha,
        log2fc_threshold=args.log2fc_threshold,
        save_path=args.output
    )

    print("\nDone!")
