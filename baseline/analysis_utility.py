"""
analysis_utility.py
===================

Utility functions for model training workflow.
"""

import os
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
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


def train_gene_majority_vote(train_data: pd.DataFrame, random_seed: int = None) -> Dict[str, int]:
    """
    Train gene-level majority vote predictor for 3-class problem.

    Args:
        train_data: DataFrame with columns [pert, gene, label]
        random_seed: Optional random seed for reproducible tie-breaking

    Returns:
        Dict mapping gene -> majority class (0, 1, or 2)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    gene_stats = defaultdict(lambda: {0: 0, 1: 0, 2: 0})

    for _, row in train_data.iterrows():
        gene = row['gene']
        label = row['label']
        gene_stats[gene][label] += 1

    # Determine majority class per gene
    gene_majority = {}
    tied_genes = []

    for gene, counts in gene_stats.items():
        # Find maximum count
        max_count = max(counts.values())
        # Find all classes with the maximum count
        tied_classes = [cls for cls, cnt in counts.items() if cnt == max_count]

        if len(tied_classes) > 1:
            # Randomly select one class with equal probability
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
        Array of boolean mask: predictions[unseen_mask] isolate out only seen gene in test
    """
    predictions = []
    n_unseen = 0
    unseen_mask = []
    
    for _, row in test_data.iterrows():
        gene = row['gene']
        if gene in gene_majority:
            predictions.append(gene_majority[gene])
            unseen_mask.append(False)
        else:
            n_unseen += 1
            unseen_mask.append(True)
            predictions.append(default_pred)

    print(f"Observed {n_unseen} unseen genes from the training phase in the test set.")
    return np.array(predictions), np.array(unseen_mask)


def compute_3class_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> Dict[str, float]:
    """
    Compute classification metrics for 3-class problem.

    Computes overall and per-class metrics for:
    - Class 0: Non-DE
    - Class 1: Up-regulated
    - Class 2: Down-regulated

    Args:
        y_true: True labels (0, 1, or 2)
        y_pred: Predicted labels (0, 1, or 2)
        y_proba: Optional predicted probabilities of shape (n_samples, 3).
                 If provided, One-vs-Rest AUROC metrics will be computed.

    Returns:
        Dict with keys:
            - accuracy: Overall accuracy
            - balanced_accuracy: Balanced accuracy (accounts for class imbalance)
            - precision_macro: Macro-averaged precision
            - recall_macro: Macro-averaged recall
            - f1_macro: Macro-averaged F1 score
            - precision_class_{i}: Precision for class i (i=0,1,2)
            - recall_class_{i}: Recall for class i (i=0,1,2) (aka. accuracy)
            - f1_class_{i}: F1 score for class i (i=0,1,2)
            - auroc_ovr_class_{i}: One-vs-Rest AUROC for class i (if y_proba provided)
            - auroc_ovr_macro: Macro-average One-vs-Rest AUROC (if y_proba provided)
    """

    # Overall metrics
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
        metrics[f'precision_class_{class_idx}'] = precision_per_class[class_idx]
        metrics[f'recall_class_{class_idx}'] = recall_per_class[class_idx]
        metrics[f'f1_class_{class_idx}'] = f1_per_class[class_idx]

    # Compute One-vs-Rest AUROC if probabilities are provided
    if y_proba is not None:
        auroc_per_class = []
        for class_idx in range(3):
            # Create binary labels: 1 for current class, 0 for all others
            y_true_binary = (y_true == class_idx).astype(int)
            
            # Get probabilities for current class
            y_score = y_proba[:, class_idx]

            # Only compute AUROC if both classes are present
            if len(np.unique(y_true_binary)) > 1:
                auroc = roc_auc_score(y_true_binary, y_score)
                metrics[f'auroc_ovr_class_{class_idx}'] = auroc
                auroc_per_class.append(auroc)
            else:
                # If only one class present, AUROC is undefined
                metrics[f'auroc_ovr_class_{class_idx}'] = np.nan

        # Compute macro-average AUROC (unweighted mean across classes)
        if auroc_per_class:
            metrics['auroc_ovr_macro'] = np.mean(auroc_per_class)
        else:
            metrics['auroc_ovr_macro'] = np.nan

    return metrics


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
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if class_names is None:
        class_names = ['Non-DE (0)', 'Up-reg (1)', 'Down-reg (2)']

    # Determine number of classes from data
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

    # Print summary statistics
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