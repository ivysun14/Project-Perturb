"""
analysis_utility.py
===================

Utility functions used in analysis workflow and model training.
"""

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from collections import defaultdict
from typing import Dict
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


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

def filter_perturbqa_style(
    df: pd.DataFrame,
    de_pvalue_threshold1: float = 0.05,
    de_pvalue_threshold2: float = 0.01,
    non_de_pvalue_threshold: float = 0.1,
    min_degs_per_pert: int = 5,
    top_degs_per_pert: int = 20,
    non_degs_per_pert: int = 100,
    n_negative_controls: int = 100,
    p_value_col: str = 'p_value',
    pert_col: str = 'pert',
    gene_col: str = 'gene',
    label_col: str = 'label',
    random_seed: int = 42,
    save_path: str = None
) -> pd.DataFrame:
    """
    Filter DE_DOC results following PerturbQA selection criteria.

    Selection rules:
    1. Exclude perturbations with ≤ min_degs_per_pert DEGs (p < de_pvalue_threshold1)
       Exception: Keep n_negative_controls perturbations with 0 DEGs as negative controls
    2. For remaining perturbations:
       - DE pairs: p < de_pvalue_threshold2
       - Non-DE pairs: p > non_de_pvalue_threshold2
    3. Per perturbation, select:
       - Top top_degs_per_pert DEGs ranked by p-value as positives
       - Sample non_degs_per_pert non-DEGs as negatives

    Args:
        df: DataFrame with DE_DOC statistics (output from DE_DOC_stats.py)
        de_pvalue_threshold1: P-value threshold for counting DEGs per perturbation (default: 0.05)
        de_pvalue_threshold2: P-value threshold for selecting DE pairs (default: 0.01)
        non_de_pvalue_threshold: P-value threshold for non-DE (default: 0.1)
        min_degs_per_pert: Minimum DEGs per perturbation to keep (default: 5)
        top_degs_per_pert: Number of top DEGs to select per perturbation (default: 20)
        non_degs_per_pert: Number of non-DEGs to sample per perturbation (default: 100)
        n_negative_controls: Number of 0-DEG perturbations to keep as controls (default: 100)
        p_value_col: Column name for p-values (default: 'p_value')
        pert_col: Column name for perturbation (default: 'pert')
        gene_col: Column name for gene (default: 'gene')
        label_col: Column name for DE/DOC label (default: 'label')
        random_seed: Random seed for sampling (default: 42)
        save_path: Optional path to save filtered CSV (default: None)

    Returns:
        Filtered DataFrame with selected perturbation-gene pairs

    Examples:
        >>> df = pd.read_csv('DE_DOC_rpe1.csv')
        >>> filtered_df = filter_perturbqa_style(df, save_path='DE_DOC_rpe1_filtered.csv')
    """
    np.random.seed(random_seed)

    print(f"\n{'='*70}")
    print("PerturbQA-Style Filtering")
    print(f"{'='*70}")
    print(f"Input: {len(df)} perturbation-gene pairs")
    print(f"\nParameters:")
    print(f"  DE p-value threshold 1 (for counting DEGs): {de_pvalue_threshold1}")
    print(f"  DE p-value threshold 2 (for selecting DE pairs): {de_pvalue_threshold2}")
    print(f"  Non-DE p-value threshold: {non_de_pvalue_threshold}")
    print(f"  Min DEGs per perturbation: {min_degs_per_pert}")
    print(f"  Top DEGs per perturbation: {top_degs_per_pert}")
    print(f"  Non-DEGs per perturbation: {non_degs_per_pert}")
    print(f"  Negative control perturbations: {n_negative_controls}")

    # Count DEGs per perturbation using threshold1
    deg_counts = {}
    for pert in df[pert_col].unique():
        pert_df = df[df[pert_col] == pert]
        n_degs = (pert_df[p_value_col] < de_pvalue_threshold1).sum()
        deg_counts[pert] = n_degs

    # Separate perturbations by DEG count
    zero_deg_perts = [p for p, c in deg_counts.items() if c == 0]
    low_deg_perts = [p for p, c in deg_counts.items() if 0 < c <= min_degs_per_pert]
    high_deg_perts = [p for p, c in deg_counts.items() if c > min_degs_per_pert]

    print(f"\nPerturbation statistics:")
    print(f"  Perturbations with 0 DEGs: {len(zero_deg_perts)}")
    print(f"  Perturbations with 1-{min_degs_per_pert} DEGs: {len(low_deg_perts)}")
    print(f"  Perturbations with >{min_degs_per_pert} DEGs: {len(high_deg_perts)}")

    # Sample negative control perturbations (0 DEGs)
    if len(zero_deg_perts) > n_negative_controls:
        sampled_neg_controls = np.random.choice(
            zero_deg_perts,
            n_negative_controls,
            replace=False
        ).tolist()
    else:
        sampled_neg_controls = zero_deg_perts
        print(f"  Warning: Only {len(zero_deg_perts)} zero-DEG perturbations available, using all")

    # Perturbations to keep
    kept_perturbations = set(high_deg_perts + sampled_neg_controls)

    print(f"\nKept perturbations: {len(kept_perturbations)}")
    print(f"  High-DEG perturbations: {len(high_deg_perts)}")
    print(f"  Negative control perturbations: {len(sampled_neg_controls)}")
    print(f"\nExcluded perturbations: {len(low_deg_perts) + len(zero_deg_perts) - sampled_neg_controls}")

    # Filter and select pairs per perturbation
    selected_pairs = []

    for pert in kept_perturbations:
        pert_df = df[df[pert_col] == pert].copy()

        # Classify as DE or non-DE based on p-value thresholds (using threshold2 for selection)
        de_pairs = pert_df[pert_df[p_value_col] < de_pvalue_threshold2]
        non_de_pairs = pert_df[pert_df[p_value_col] > non_de_pvalue_threshold]

        # Select top DEGs ranked by p-value
        if len(de_pairs) > 0:
            de_pairs_sorted = de_pairs.sort_values(p_value_col)
            selected_de = de_pairs_sorted.head(top_degs_per_pert)
            selected_pairs.append(selected_de)

        # Sample non-DEGs
        if len(non_de_pairs) > 0:
            n_sample = min(non_degs_per_pert, len(non_de_pairs))
            sampled_non_de = non_de_pairs.sample(n=n_sample, random_state=random_seed)
            selected_pairs.append(sampled_non_de)

    # Combine all selected pairs
    if selected_pairs:
        filtered_df = pd.concat(selected_pairs, ignore_index=True)
    else:
        filtered_df = pd.DataFrame(columns=df.columns)

    # Update labels based on new thresholds
    # DE: p < de_pvalue_threshold2, keep original direction labels (1=up, 2=down)
    def update_label(row):
        if row[p_value_col] < de_pvalue_threshold2:
            # Keep original direction label (1 or 2)
            return row[label_col]
        else:
            # Non-DE
            return 0

    filtered_df[label_col] = filtered_df.apply(update_label, axis=1)

    # Print summary statistics
    print(f"\n{'='*70}")
    print("Filtering Results")
    print(f"{'='*70}")
    print(f"Total filtered pairs: {len(filtered_df)}")
    print(f"\nLabel distribution:")
    print(f"  Non-DE (0): {(filtered_df[label_col] == 0).sum()}")
    print(f"  Up-regulated (1): {(filtered_df[label_col] == 1).sum()}")
    print(f"  Down-regulated (2): {(filtered_df[label_col] == 2).sum()}")

    print(f"\nPairs per perturbation:")
    pairs_per_pert = filtered_df.groupby(pert_col).size()
    print(f"  Mean: {pairs_per_pert.mean():.1f}")
    print(f"  Median: {pairs_per_pert.median():.1f}")
    print(f"  Min: {pairs_per_pert.min()}")
    print(f"  Max: {pairs_per_pert.max()}")

    # Save if path provided
    if save_path:
        filtered_df.to_csv(save_path, index=False)
        print(f"\nFiltered results saved to {save_path}")

    return filtered_df


def train_gene_majority_vote(train_data: pd.DataFrame) -> Dict[str, int]:
    """
    Train gene-level majority vote predictor for 3-class problem.

    Args:
        train_data: DataFrame with columns [gene, label]
                   where label is 0 (non-DE), 1 (up-regulated), or 2 (down-regulated)

    Returns:
        Dict mapping gene -> majority class (0, 1, or 2)

    Examples:
        >>> train_df = pd.DataFrame({
        ...     'gene': ['GENE1', 'GENE1', 'GENE2', 'GENE2'],
        ...     'label': [1, 1, 0, 2]
        ... })
        >>> gene_majority = train_gene_majority_vote(train_df)
        >>> gene_majority['GENE1']  # Should be 1 (up-regulated)
        1
    """
    gene_stats = defaultdict(lambda: {0: 0, 1: 0, 2: 0})

    for _, row in train_data.iterrows():
        gene = row['gene']
        label = row['label']
        gene_stats[gene][label] += 1

    # Determine majority class per gene
    gene_majority = {}
    for gene, counts in gene_stats.items():
        # Find class with maximum count
        majority_class = max(counts.items(), key=lambda x: x[1])[0]
        gene_majority[gene] = majority_class

    return gene_majority


def predict_from_gene_majority(test_data: pd.DataFrame,
                                gene_majority: Dict[str, int],
                                default_pred: int = 0) -> np.ndarray:
    """
    Generate predictions based on gene-level majority vote.

    Args:
        test_data: DataFrame with column [gene]
        gene_majority: Dict mapping gene -> majority class
        default_pred: Default prediction for unseen genes (default: 0)

    Returns:
        Array of predictions

    Examples:
        >>> test_df = pd.DataFrame({'gene': ['GENE1', 'GENE2', 'GENE3']})
        >>> gene_majority = {'GENE1': 1, 'GENE2': 2}
        >>> predictions = predict_from_gene_majority(test_df, gene_majority, default_pred=0)
        >>> # GENE1 -> 1, GENE2 -> 2, GENE3 -> 0 (default)
    """
    predictions = []
    for _, row in test_data.iterrows():
        gene = row['gene']
        if gene in gene_majority:
            predictions.append(gene_majority[gene])
        else:
            predictions.append(default_pred)

    return np.array(predictions)


def compute_3class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute classification metrics for 3-class problem.

    Computes overall and per-class metrics for:
    - Class 0: Non-DE
    - Class 1: Up-regulated
    - Class 2: Down-regulated

    Args:
        y_true: True labels (0, 1, or 2)
        y_pred: Predicted labels (0, 1, or 2)

    Returns:
        Dict with keys:
            - accuracy: Overall accuracy
            - balanced_accuracy: Balanced accuracy (accounts for class imbalance)
            - precision_macro: Macro-averaged precision
            - recall_macro: Macro-averaged recall
            - f1_macro: Macro-averaged F1 score
            - accuracy_class_{i}: Accuracy for class i (i=0,1,2)
            - precision_class_{i}: Precision for class i (i=0,1,2)
            - recall_class_{i}: Recall for class i (i=0,1,2)
            - f1_class_{i}: F1 score for class i (i=0,1,2)

    Examples:
        >>> y_true = np.array([0, 1, 2, 0, 1])
        >>> y_pred = np.array([0, 1, 2, 1, 1])
        >>> metrics = compute_3class_metrics(y_true, y_pred)
        >>> print(metrics['accuracy'])  # Overall accuracy
        0.8
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0)
    }

    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    for class_idx in range(3):
        # Per-class accuracy: (true positives for this class) / (total samples of this class)
        class_mask = (y_true == class_idx)
        if class_mask.sum() > 0:
            accuracy_per_class = (y_pred[class_mask] == class_idx).sum() / class_mask.sum()
        else:
            accuracy_per_class = 0.0

        metrics[f'accuracy_class_{class_idx}'] = accuracy_per_class
        metrics[f'precision_class_{class_idx}'] = precision_per_class[class_idx]
        metrics[f'recall_class_{class_idx}'] = recall_per_class[class_idx]
        metrics[f'f1_class_{class_idx}'] = f1_per_class[class_idx]

    return metrics