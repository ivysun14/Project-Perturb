"""
analysis_utility.py
===================

Utility functions for model training workflow.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List

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