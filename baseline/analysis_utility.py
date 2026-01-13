"""
analysis_utility.py
===================

Utility functions used in analysis workflow and model training.
"""

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc


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

def update_DE_DOC_labels(
    df: pd.DataFrame,
    alpha: float = 0.05,
    log2fc_threshold: float = 0.25,
    p_value_col: str = 'p_value',
    log2fc_col: str = 'log2FC',
    label_col: str = 'label'
) -> pd.DataFrame:
    """
    Update DE and DOC labels in an existing DataFrame based on new thresholds.

    A gene is considered DE if:
    1. BH-corrected p-value < alpha (significant)
    2. |log2FC| >= log2fc_threshold (biologically meaningful change)

    For DE genes, direction is determined by sign of log2FC:
    - label = 1: up-regulated (log2FC > 0)
    - label = 2: down-regulated (log2FC < 0)
    - label = 0: non-DE (fails significance or log2FC threshold)

    Args:
        df: DataFrame with DE statistics (must contain p_value and log2FC columns)
        alpha: New significance threshold for BH-corrected p-value (default: 0.05)
        log2fc_threshold: New minimum absolute log2 fold change threshold (default: 0.25)
        p_value_col: Name of column containing BH-corrected p-values (default: 'p_value')
        log2fc_col: Name of column containing log2 fold changes (default: 'log2FC')
        label_col: Name of column to update with new labels (default: 'label')

    Returns:
        DataFrame with updated labels (modifies a copy, original df unchanged)

    Examples:
        >>> # Load existing DE results
        >>> df = pd.read_csv('de_results.csv')
        >>>
        >>> # Apply stricter thresholds
        >>> df_strict = update_DE_DOC_labels(df, alpha=0.01, log2fc_threshold=0.5)
        >>>
        >>> # Apply more lenient thresholds
        >>> df_lenient = update_DE_DOC_labels(df, alpha=0.1, log2fc_threshold=0.1)
    """

    # Create a copy to avoid modifying original
    df_updated = df.copy()

    # Validate required columns
    required_cols = [p_value_col, log2fc_col]
    missing_cols = [col for col in required_cols if col not in df_updated.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Update labels based on new thresholds
    def classify_label(row):
        passes_pval = row[p_value_col] < alpha
        passes_log2fc = abs(row[log2fc_col]) >= log2fc_threshold

        if passes_pval and passes_log2fc:
            # DE gene: determine direction
            if row[log2fc_col] > 0:
                return 1  # Up-regulated
            else:
                return 2  # Down-regulated
        else:
            # Non-DE gene
            return 0

    df_updated[label_col] = df_updated.apply(classify_label, axis=1)

    # Print summary statistics
    total = len(df_updated)
    non_de = (df_updated[label_col] == 0).sum()
    up_reg = (df_updated[label_col] == 1).sum()
    down_reg = (df_updated[label_col] == 2).sum()

    print(f"\nLabels updated with new thresholds:")
    print(f"  alpha = {alpha}")
    print(f"  log2FC_threshold = {log2fc_threshold}")
    print(f"\nResults:")
    print(f"  Total pairs: {total}")
    print(f"  Non-DE (label=0): {non_de} ({100*non_de/total:.1f}%)")
    print(f"  Up-regulated (label=1): {up_reg} ({100*up_reg/total:.1f}%)")
    print(f"  Down-regulated (label=2): {down_reg} ({100*down_reg/total:.1f}%)")

    return df_updated