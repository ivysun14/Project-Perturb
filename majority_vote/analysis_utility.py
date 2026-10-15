"""
analysis_utility.py
===================

Utility functions for model training workflow.
"""

import os
import warnings
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List

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

    # skip all matrix work if no duplicates exist
    if all(len(v) == 1 for v in groups.values()):
        return adata

    # average duplicate columns; operate sparse column-wise to avoid full densification
    import scipy.sparse as sp
    X = adata.X
    is_sparse = sp.issparse(X)
    cols = []
    for idxs in groups.values():
        if is_sparse:
            col = X[:, idxs[0]]
            for idx in idxs[1:]:
                col = col + X[:, idx]
            cols.append(col / len(idxs))
        else:
            cols.append(X[:, idxs].mean(axis=1, keepdims=True))
    X_new = sp.hstack(cols, format='csr') if is_sparse else np.hstack(cols)
    var_new = adata.var.iloc[[idxs[0] for idxs in groups.values()]].copy()
    return ad.AnnData(X=X_new, obs=adata.obs.copy(), var=var_new)

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
    """
    df_updated = df.copy()
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


def train_gene_majority_vote(train_data: pd.DataFrame, random_seed: int = None) -> tuple:
    """
    Train gene-level majority vote predictor for 3-class problem.

    Args:
        train_data: DataFrame with columns [pert, gene, label]
        random_seed: Optional random seed for reproducible tie-breaking

    Returns:
        gene_majority: Dict mapping gene -> majority class (0, 1, or 2)
        gene_proba: Dict mapping gene -> np.ndarray of shape (3,) with class proportions
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    gene_stats = defaultdict(lambda: {0: 0, 1: 0, 2: 0})
    for _, row in train_data.iterrows():
        gene = row['gene']
        label = row['label']
        gene_stats[gene][label] += 1

    gene_majority = {}
    gene_proba = {}
    tied_genes = []
    for gene, counts in gene_stats.items():
        total = sum(counts.values())
        gene_proba[gene] = np.array([counts[c] / total for c in range(3)])

        max_count = max(counts.values())
        tied_classes = [cls for cls, cnt in counts.items() if cnt == max_count]
        if len(tied_classes) > 1:
            tied_genes.append((gene, tied_classes, counts))
            majority_class = np.random.choice(tied_classes)
        else:
            majority_class = tied_classes[0]
        gene_majority[gene] = majority_class

    if tied_genes:
        print(f"\nNumber of genes with tied majority: {len(tied_genes)}")
        print(f"Tied genes (randomly resolved):")
        for gene, tied_classes, counts in tied_genes:
            class_names = {0: 'Non-DE', 1: 'Up', 2: 'Down'}
            tied_str = ', '.join([f"{class_names.get(c, c)}({counts[c]})" for c in tied_classes])
            assigned = gene_majority[gene]
            print(f"  {gene}: tied between [{tied_str}] -> assigned {class_names.get(assigned, assigned)}")
    else:
        print(f"\nNo genes with tied majority classes.")

    return gene_majority, gene_proba


def predict_from_gene_majority(test_data: pd.DataFrame,
                                gene_majority: Dict[str, int],
                                gene_proba: Dict[str, np.ndarray],
                                default_pred: int = 0) -> tuple:
    """
    Generate predictions and per-class probabilities based on gene-level majority vote.

    Args:
        test_data: DataFrame with column [gene]
        gene_majority: Dict mapping gene -> majority class
        gene_proba: Dict mapping gene -> np.ndarray of shape (3,) with class proportions
        default_pred: Default prediction for unseen genes (default: 0)

    Returns:
        predictions: Array of predicted labels
        unseen_mask: Boolean array, True where gene was unseen in training
        probabilities: Array of shape (n_samples, 3); unseen genes get [1/3, 1/3, 1/3]
    """
    predictions = []
    probabilities = []
    n_unseen = 0
    unseen_mask = []
    for _, row in test_data.iterrows():
        gene = row['gene']
        if gene in gene_majority:
            predictions.append(gene_majority[gene])
            probabilities.append(gene_proba[gene])
            unseen_mask.append(False)
        else:
            n_unseen += 1
            unseen_mask.append(True)
            predictions.append(default_pred)
            probabilities.append(np.array([1/3, 1/3, 1/3]))
    print(f"Observed {n_unseen} unseen genes from the training phase in the test set.")
    return np.array(predictions), np.array(unseen_mask), np.array(probabilities)


def plot_data_summary(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset_name: str,
    save_path: str = None,
    class_names: List[str] = None,
    output_dir: str = 'error_analysis'
):
    """
    Plot label distribution and perturbation counts for train and test sets.

    Args:
        train_df: Training DataFrame with columns [pert, gene, label]
        test_df: Test DataFrame with columns [pert, gene, label]
        dataset_name: Name of the dataset (for plot title)
        save_path: Path to save the figure (optional)
        class_names: List of class names for display (default: ['Non-DE (0)', 'Up-reg (1)', 'Down-reg (2)'])
        output_dir: Directory to save the figure if save_path not specified (default: 'error_analysis')
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if class_names is None:
        class_names = ['Non-DE (0)', 'Up-reg (1)', 'Down-reg (2)']
    n_classes = max(train_df['label'].max(), test_df['label'].max()) + 1

    # 1. Label distribution (bar chart)
    train_counts = [(train_df['label'] == i).sum() for i in range(n_classes)]
    test_counts = [(test_df['label'] == i).sum() for i in range(n_classes)]

    x = np.arange(len(class_names[:n_classes]))
    width = 0.35

    bars1 = axes[0].bar(x - width/2, train_counts, width, label='Train', color='steelblue')
    bars2 = axes[0].bar(x + width/2, test_counts, width, label='Test', color='darkorange')

    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'{dataset_name}: Label Distribution')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(class_names[:n_classes])
    axes[0].legend()

    # Add count labels on bars
    for bar in bars1:
        height = bar.get_height()
        axes[0].annotate(f'{int(height)}',
                         xy=(bar.get_x() + bar.get_width()/2, height),
                         xytext=(0, 3), textcoords="offset points",
                         ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        axes[0].annotate(f'{int(height)}',
                         xy=(bar.get_x() + bar.get_width()/2, height),
                         xytext=(0, 3), textcoords="offset points",
                         ha='center', va='bottom', fontsize=8)

    # 2. Perturbation counts
    train_perts = train_df['pert'].nunique()
    test_perts = test_df['pert'].nunique()
    train_genes = train_df['gene'].nunique()
    test_genes = test_df['gene'].nunique()

    categories = ['Perturbations', 'Genes', 'Total Pairs']
    train_values = [train_perts, train_genes, len(train_df)]
    test_values = [test_perts, test_genes, len(test_df)]

    x = np.arange(len(categories))

    bars1 = axes[1].bar(x - width/2, train_values, width, label='Train', color='steelblue')
    bars2 = axes[1].bar(x + width/2, test_values, width, label='Test', color='darkorange')

    axes[1].set_xlabel('Category')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'{dataset_name}: Train/Test Split Summary')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(categories)
    axes[1].legend()

    # Add count labels on bars
    for bar in bars1:
        height = bar.get_height()
        axes[1].annotate(f'{int(height)}',
                         xy=(bar.get_x() + bar.get_width()/2, height),
                         xytext=(0, 3), textcoords="offset points",
                         ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        axes[1].annotate(f'{int(height)}',
                         xy=(bar.get_x() + bar.get_width()/2, height),
                         xytext=(0, 3), textcoords="offset points",
                         ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nData summary plot saved to {save_path}")
    else:
        output_path = f'{output_dir}/{dataset_name}_data_summary.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nData summary plot saved to {output_path}")

    plt.close()

    print(f"\n{'='*50}")
    print("Data Summary")
    print(f"{'='*50}")
    print(f"\nTrain set:")
    print(f"  Perturbations: {train_perts}")
    print(f"  Genes: {train_genes}")
    print(f"  Total pairs: {len(train_df)}")
    label_dist = ', '.join([f"{class_names[i].split('(')[0].strip()}={train_counts[i]}" for i in range(n_classes)])
    print(f"  Label distribution: {label_dist}")

    print(f"\nTest set:")
    print(f"  Perturbations: {test_perts}")
    print(f"  Genes: {test_genes}")
    print(f"  Total pairs: {len(test_df)}")
    label_dist = ', '.join([f"{class_names[i].split('(')[0].strip()}={test_counts[i]}" for i in range(n_classes)])
    print(f"  Label distribution: {label_dist}")