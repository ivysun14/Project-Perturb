"""
process_CD4_TCell.py
====================

Process the GWCD4i.DE_stats.h5ad file to create CSV files with differential
expression statistics for each culture condition (Rest, Stim8hr, Stim48hr).

This script reads pre-computed DESeq2 results and outputs CSV files in the
same format as DE_DOC_stats.py:
- pert: perturbation name (target gene)
- gene: gene name
- label: 0 (non-DE), 1 (up-regulation), 2 (down-regulation)
- p_value: FDR-adjusted p-value
- logFC: log2 fold change from DESeq2
- log2FC: same as logFC (DESeq2 uses log2)
"""

import numpy as np
import pandas as pd
import anndata as ad
from typing import Optional
import os


def process_cd4_tcell_de_stats(
    adata: ad.AnnData,
    alpha: float = 0.10,
    log2fc_threshold: float = 0.25,
    output_dir: Optional[str] = None,
    output_prefix: str = "CD4_TCell"
) -> dict:
    """
    Process CD4+ T cell perturb-seq DE statistics and create CSV files for each condition.

    Args:
        adata: AnnData object with pre-computed DE statistics
        alpha: Significance threshold for adjusted p-value (default: 0.10, matches 10% FDR)
        log2fc_threshold: Minimum absolute log2 fold change threshold (default: 0.25)
        output_dir: Directory to save output CSV files (optional)
        output_prefix: Prefix for output file names (default: "CD4_TCell")

    Returns:
        Dictionary mapping condition names to DataFrames
    """

    # Get unique culture conditions
    conditions = adata.obs['culture_condition'].unique()
    print(f"Found {len(conditions)} culture conditions: {list(conditions)}")

    results = {}

    for condition in conditions:
        print(f"\nProcessing condition: {condition}")

        # Filter to the condition
        condition_mask = adata.obs['culture_condition'] == condition
        condition_adata = adata[condition_mask]

        print(f"  Unique Perturbations: {condition_adata.obs['target_contrast_gene_name'].nunique()}")
        print(f"  Unique Genes: {condition_adata.n_vars}")

        # Extract data
        log_fc = condition_adata.layers['log_fc']  # DESeq2 log2FC
        adj_pval = condition_adata.layers['adj_p_value']

        # Get perturbation and gene names
        pert_names = condition_adata.obs['target_contrast_gene_name'].values
        gene_names = condition_adata.var['gene_name'].values

        # Build results list
        condition_results = []

        for i, pert in enumerate(pert_names):
            for j, gene in enumerate(gene_names):
                pval = adj_pval[i, j]
                lfc = log_fc[i, j]

                # Handle NaN values
                if np.isnan(pval):
                    pval = 1.0
                if np.isnan(lfc):
                    lfc = 0.0

                # Determine label based on significance and fold change
                passes_pval = pval < alpha
                passes_log2fc = abs(lfc) >= log2fc_threshold

                if passes_pval and passes_log2fc:
                    if lfc > 0:
                        label = 1  # Up-regulated
                    else:
                        label = 2  # Down-regulated
                else:
                    label = 0  # Non-DE

                condition_results.append({
                    'pert': pert,
                    'gene': gene,
                    'label': label,
                    'p_value': pval,
                    'logFC': lfc,  # DESeq2 already uses log2
                    'log2FC': lfc
                })

        # Create DataFrame
        df = pd.DataFrame(condition_results)
        results[condition] = df

        # Summary statistics
        print(f"  Total perturbation-gene pairs: {len(df)}")
        print(f"  Non-DE (label=0): {(df['label'] == 0).sum()}")
        print(f"  Up-regulated (label=1): {(df['label'] == 1).sum()}")
        print(f"  Down-regulated (label=2): {(df['label'] == 2).sum()}")

        # Save to CSV if output directory provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{output_prefix}_{condition}.csv")
            df.to_csv(output_path, index=False)
            print(f"  Saved to: {output_path}")

    return results


if __name__ == "__main__":

    # Load AnnData
    print(f"Loading AnnData...")
    adata = ad.read_h5ad("datasets/GWCD4i.DE_stats.h5ad")
    print(f"Loaded: {adata.n_obs} cells x {adata.n_vars} genes")

    # Process and save
    results = process_cd4_tcell_de_stats(
        adata=adata,
        output_dir="files"
    )

    print("\nDone!")
