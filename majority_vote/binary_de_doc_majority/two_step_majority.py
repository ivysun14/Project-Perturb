"""
two_step_baseline.py
====================

Two-step gene-level majority vote baseline for 3-class DE prediction.

This script implements a two-step classification approach:
1. Step 1: Determine if a gene is DE vs non-DE by majority vote
2. Step 2: For genes classified as DE, determine up-regulation (1) vs down-regulation (2)
           by majority among DE instances

Input is precomputed DE/DOC statistics in the output format of DE_DOC_stats.py.

Error Analysis: Performed separately for each binary step:
- Step 1: DE (1) vs non-DE (0)
- Step 2: Up (1) vs Down (0) for DE genes only
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
# Add parent directory for function loading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from error_utility import run_error_analysis_from_results
from analysis_utility import plot_data_summary, compute_3class_metrics


def load_de_doc_file(file_path: str, test_size: float = 0.25, random_state: int = 305) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load DE+DOC CSV file and split into train/test.

    Args:
        file_path: Path to DE_DOC CSV file
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_df, test_df)
    """
    df = pd.read_csv(file_path)

    print(f"Loaded DE+DOC file: {len(df)} perturbation-gene pairs")
    print(f"\n3-Class label distribution:")
    print(f"  Non-DE (0): {(df['label'] == 0).sum()}")
    print(f"  Up-regulated (1): {(df['label'] == 1).sum()}")
    print(f"  Down-regulated (2): {(df['label'] == 2).sum()}")

    # Split into train/test if not already exist in the csv
    if 'split' in df.columns:
        train_df = df[df['split'] == 'train']
        test_df = df[df['split'] == 'test']
    else:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df['label']  # Stratified split to maintain class distribution
        )

    return train_df, test_df


def train_two_step_majority_vote(train_data: pd.DataFrame, random_seed: int = None) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Train two-step gene-level majority vote predictor.

    Step 1: For each gene, determine if it is more often DE (label 1 or 2) or non-DE (label 0)
    Step 2: For each gene, among its DE instances (label 1 or 2), determine the majority direction

    Args:
        train_data: DataFrame with columns [pert, gene, label]
        random_seed: Optional random seed for reproducible tie-breaking

    Returns:
        Tuple of:
            - gene_de_majority: Dict mapping gene -> 0 (non-DE majority) or 1 (DE majority)
            - gene_direction_majority: Dict mapping gene -> 1 (up) or 2 (down) for direction
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Step 1: Count DE vs non-DE for each gene
    gene_de_stats = defaultdict(lambda: {'de': 0, 'non_de': 0})

    # Step 2: Count up vs down for each gene (only when DE)
    gene_direction_stats = defaultdict(lambda: {'up': 0, 'down': 0})

    for _, row in train_data.iterrows():
        gene = row['gene']
        label = row['label']

        # Step 1 stats: DE (1 or 2) vs non-DE (0)
        if label == 0:
            gene_de_stats[gene]['non_de'] += 1
        else:
            gene_de_stats[gene]['de'] += 1
            # Step 2 stats: only count direction when DE
            if label == 1:
                gene_direction_stats[gene]['up'] += 1
            elif label == 2:
                gene_direction_stats[gene]['down'] += 1

    # Determine Step 1 majority: DE vs non-DE
    gene_de_majority = {}
    de_tied_genes = []

    for gene, stats in gene_de_stats.items():
        de_count = stats['de']
        non_de_count = stats['non_de']

        if de_count > non_de_count:
            gene_de_majority[gene] = 1  # DE majority
        elif non_de_count > de_count:
            gene_de_majority[gene] = 0  # non-DE majority
        else:
            # Tie: randomly select
            de_tied_genes.append((gene, de_count, non_de_count))
            gene_de_majority[gene] = np.random.choice([0, 1])

    # Determine Step 2 majority: up vs down (for genes with any DE instances)
    gene_direction_majority = {}
    dir_tied_genes = []

    for gene, stats in gene_direction_stats.items():
        up_count = stats['up']
        down_count = stats['down']

        if up_count > down_count:
            gene_direction_majority[gene] = 1  # Up majority
        elif down_count > up_count:
            gene_direction_majority[gene] = 2  # Down majority
        else:
            # Tie: randomly select between up (1) and down (2)
            dir_tied_genes.append((gene, up_count, down_count))
            gene_direction_majority[gene] = np.random.choice([1, 2])

    # Print summary
    print(f"\nStep 1 - DE vs non-DE majority:")
    n_de_majority = sum(1 for v in gene_de_majority.values() if v == 1)
    n_non_de_majority = sum(1 for v in gene_de_majority.values() if v == 0)
    print(f"  Genes with DE majority: {n_de_majority}")
    print(f"  Genes with non-DE majority: {n_non_de_majority}")
    if de_tied_genes:
        print(f"  Genes with tied DE/non-DE (randomly resolved): {len(de_tied_genes)}")

    print(f"\nStep 2 - Direction majority (among DE instances):")
    n_up_majority = sum(1 for v in gene_direction_majority.values() if v == 1)
    n_down_majority = sum(1 for v in gene_direction_majority.values() if v == 2)
    print(f"  Genes with up-regulation majority: {n_up_majority}")
    print(f"  Genes with down-regulation majority: {n_down_majority}")
    if dir_tied_genes:
        print(f"  Genes with tied direction (randomly resolved): {len(dir_tied_genes)}")

    return gene_de_majority, gene_direction_majority


def predict_two_step(test_data: pd.DataFrame,
                     gene_de_majority: Dict[str, int],
                     gene_direction_majority: Dict[str, int],
                     default_de_pred: int = 0,
                     default_dir_pred: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate predictions using two-step majority vote.

    Step 1: Check if gene's majority is DE or non-DE
    Step 2: If DE, use direction majority; if non-DE, predict 0

    Args:
        test_data: DataFrame with column [gene]
        gene_de_majority: Dict mapping gene -> 0 (non-DE) or 1 (DE)
        gene_direction_majority: Dict mapping gene -> 1 (up) or 2 (down)
        default_de_pred: Default DE/non-DE prediction for unseen genes (default: 0 = non-DE)
        default_dir_pred: Default direction for DE genes without direction info (default: 1 = up)

    Returns:
        Tuple of:
            - final_predictions: Array of 3-class predictions (0=non-DE, 1=up, 2=down)
            - step1_predictions: Array of binary DE predictions (0=non-DE, 1=DE)
            - step2_predictions: Array of direction predictions for DE samples (1=up, 2=down), -1 for non-DE
    """
    final_predictions = []
    step1_predictions = []
    step2_predictions = []
    n_unseen = 0
    n_de_no_direction = 0

    for _, row in test_data.iterrows():
        gene = row['gene']

        # Step 1: Check DE majority
        if gene in gene_de_majority:
            is_de = gene_de_majority[gene]
        else:
            is_de = default_de_pred
            n_unseen += 1

        step1_predictions.append(is_de)

        # Step 2: If DE, determine direction; if non-DE, predict 0
        if is_de == 0:
            final_predictions.append(0)  # Non-DE
            step2_predictions.append(-1)  # No direction prediction for non-DE
        else:
            # Gene is DE, determine direction
            if gene in gene_direction_majority:
                direction = gene_direction_majority[gene]
            else:
                # Edge case: gene marked as DE but no direction info
                direction = default_dir_pred
                n_de_no_direction += 1
            final_predictions.append(direction)
            step2_predictions.append(direction)

    if n_unseen > 0:
        print(f"  {n_unseen} unseen genes in test set (defaulted to non-DE)")
    if n_de_no_direction > 0:
        print(f"  {n_de_no_direction} DE genes without direction info (defaulted to up)")

    return np.array(final_predictions), np.array(step1_predictions), np.array(step2_predictions)


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1) -> Dict[str, float]:
    """
    Compute classification metrics for binary problem.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        pos_label: The positive class label

    Returns:
        Dict with binary metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
        'recall': recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
        'f1': f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    }
    return metrics

def format_mean_se(values):
    """Calculate mean and standard error."""
    mean = np.mean(values)
    # Standard error = std / sqrt(n)
    se = np.std(values, ddof=1) / np.sqrt(len(values))
    return mean, se


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Two-step gene-based majority vote baseline for 3-class DE prediction')
    parser.add_argument('--de_doc_file', type=str, required=True, help='Path to DE_DOC CSV file from DE_DOC_stats.py')
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name (for output files)')
    parser.add_argument('--test_size', type=float, default=0.25, help='Fraction of data for testing (default: 0.25)')
    parser.add_argument('--n_runs', type=int, default=3, help='Number of runs with different train subsampling (default: 3)')
    parser.add_argument('--train_fraction', type=float, default=0.8, help='Fraction of training data to use in each run (default: 0.8)')
    parser.add_argument('--random_seed', type=int, default=305, help='Random seed for train/test split (default: 305)')

    args = parser.parse_args()

    # Create error_analysis directory if it doesn't exist
    os.makedirs('error_analysis', exist_ok=True)

    print("="*70)
    print("Two-Step Gene-Based Majority Vote Baseline")
    print("="*70)
    print("\nApproach:")
    print("  Step 1: Classify gene as DE vs non-DE by majority vote")
    print("  Step 2: For DE genes, classify as up (1) vs down (2) by majority")
    print("="*70)

    # Load and split data
    print("\nLoading data...")
    train_df, test_df = load_de_doc_file(args.de_doc_file, test_size=args.test_size, random_state=args.random_seed)

    print(f"\nTrain set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")

    # Plot data summary
    plot_data_summary(train_df, test_df, args.dataset_name + "_two_step")

    # Store results from multiple runs
    all_metrics_3class = defaultdict(list)
    all_metrics_step1 = defaultdict(list)
    all_metrics_step2 = defaultdict(list)

    # Run multiple times with different train subsampling
    for run in range(args.n_runs):
        print(f"\n{'='*70}")
        print(f"Run {run + 1}/{args.n_runs}")
        print(f"{'='*70}")

        # Subsample training data
        np.random.seed(run)
        n_train = int(len(train_df) * args.train_fraction)
        train_indices = np.random.choice(len(train_df), n_train, replace=False)
        train_subsample = train_df.iloc[train_indices]

        print(f"Training on {len(train_subsample)} samples ({args.train_fraction*100:.0f}% of train set)")

        # Train two-step majority vote
        print("\nTraining two-step majority vote...")
        gene_de_majority, gene_direction_majority = train_two_step_majority_vote(train_subsample, random_seed=run)
        print(f"Learned DE majority for {len(gene_de_majority)} genes")
        print(f"Learned direction majority for {len(gene_direction_majority)} genes")

        # Predict on test set
        print("\nPredicting on test set...")
        y_pred_3class, y_pred_step1, y_pred_step2 = predict_two_step(
            test_df, gene_de_majority, gene_direction_majority
        )
        y_true_3class = test_df['label'].values

        # Create binary ground truth for Step 1: DE (1) vs non-DE (0)
        y_true_step1 = (y_true_3class > 0).astype(int)

        # Create binary ground truth and predictions for Step 2 (DE samples only)
        # For Step 2, we evaluate only on samples that are truly DE (label 1 or 2)
        de_mask = y_true_3class > 0
        y_true_step2_de_only = y_true_3class[de_mask]  # Will be 1 or 2
        y_pred_step2_de_only = y_pred_step2[de_mask]   # Will be 1 or 2 (or -1 if predicted non-DE)

        # Convert to binary: 1=up (class 1), 0=down (class 2)
        y_true_step2_binary = (y_true_step2_de_only == 1).astype(int)  # 1 for up, 0 for down
        y_pred_step2_binary = (y_pred_step2_de_only == 1).astype(int)  # 1 for up, 0 for down

        # Compute 3-class metrics
        metrics_3class = compute_3class_metrics(y_true_3class, y_pred_3class)
        for key, value in metrics_3class.items():
            all_metrics_3class[key].append(value)

        # Compute Step 1 metrics (DE vs non-DE)
        metrics_step1 = compute_binary_metrics(y_true_step1, y_pred_step1, pos_label=1)
        for key, value in metrics_step1.items():
            all_metrics_step1[key].append(value)

        # Compute Step 2 metrics (Up vs Down, DE samples only)
        if len(y_true_step2_binary) > 0:
            metrics_step2 = compute_binary_metrics(y_true_step2_binary, y_pred_step2_binary, pos_label=1)
        else:
            print("WARNING: No DE genes in this dataset!")
            metrics_step2 = {k: 0.0 for k in ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1']}
        for key, value in metrics_step2.items():
            all_metrics_step2[key].append(value)

        # Print results for this run
        print(f"\n--- Results (Run {run + 1}) ---")

        print(f"\n[3-Class Overall]")
        print(f"  Accuracy:          {metrics_3class['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {metrics_3class['balanced_accuracy']:.4f}")
        print(f"  F1 (macro):        {metrics_3class['f1_macro']:.4f}")

        print(f"\n[Step 1: DE vs non-DE] (n={len(y_true_step1)})")
        print(f"  Accuracy:          {metrics_step1['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {metrics_step1['balanced_accuracy']:.4f}")
        print(f"  Precision (DE):    {metrics_step1['precision']:.4f}")
        print(f"  Recall (DE):       {metrics_step1['recall']:.4f}")
        print(f"  F1 (DE):           {metrics_step1['f1']:.4f}")

        print(f"\n[Step 2: Up vs Down] (n={len(y_true_step2_binary)}, DE samples only)")
        print(f"  Accuracy:          {metrics_step2['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {metrics_step2['balanced_accuracy']:.4f}")
        print(f"  Precision (Up):    {metrics_step2['precision']:.4f}")
        print(f"  Recall (Up):       {metrics_step2['recall']:.4f}")
        print(f"  F1 (Up):           {metrics_step2['f1']:.4f}")

    # Calculate mean and standard error across runs
    print(f"\n{'='*70}")
    print("FINAL RESULTS (Mean ± SE across runs)")
    print(f"{'='*70}")

    print(f"\n[3-Class Overall Metrics]")
    for metric_name in ['accuracy', 'balanced_accuracy', 'precision_macro', 'recall_macro', 'f1_macro']:
        mean, se = format_mean_se(all_metrics_3class[metric_name])
        display_name = metric_name.replace('_', ' ').title()
        print(f"  {display_name:20s}: {mean:.4f} ± {se:.4f}")

    print(f"\n[Step 1: DE vs non-DE]")
    for metric_name in ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1']:
        mean, se = format_mean_se(all_metrics_step1[metric_name])
        display_name = metric_name.capitalize()
        print(f"  {display_name:20s}: {mean:.4f} ± {se:.4f}")

    print(f"\n[Step 2: Up vs Down (DE samples only)]")
    for metric_name in ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1']:
        mean, se = format_mean_se(all_metrics_step2[metric_name])
        display_name = metric_name.capitalize()
        print(f"  {display_name:20s}: {mean:.4f} ± {se:.4f}")

    # Print detailed classification reports for the last run
    print(f"\n{'='*70}")
    print("Detailed Classification Reports (Last Run)")
    print(f"{'='*70}")

    print("\n[Step 1: DE vs non-DE]")
    print(classification_report(
        y_true_step1,
        y_pred_step1,
        target_names=['non-DE', 'DE'],
        zero_division=0
    ))

    print("\n[Step 2: Up vs Down (DE samples only)]")
    if len(y_true_step2_binary) > 0:
        print(classification_report(
            y_true_step2_binary,
            y_pred_step2_binary,
            target_names=['Down', 'Up'],
            zero_division=0
        ))
    else:
        print("  No DE samples to evaluate.")

    # Run error analysis on the last run - SEPARATELY for each step
    print(f"\n{'='*70}")
    print("ERROR ANALYSIS (Last Run)")
    print(f"{'='*70}")

    # Step 1 Error Analysis: DE vs non-DE (binary)
    print(f"\n{'='*60}")
    print("Step 1 Error Analysis: DE vs non-DE")
    print(f"{'='*60}")

    test_keys_step1 = list(zip(test_df['pert'], test_df['gene']))

    analyzer_step1, error_results_step1 = run_error_analysis_from_results(
        dataset=args.dataset_name,
        test_keys=test_keys_step1,
        predictions=y_pred_step1,
        true_labels=y_true_step1,
        adata=None,
        task_name="step1_DE_vs_nonDE",
        class_names=['non-DE', 'DE']
    )

    # Step 2 Error Analysis: Up vs Down (binary, DE samples only)
    print(f"\n{'='*60}")
    print("Step 2 Error Analysis: Up vs Down (DE samples only)")
    print(f"{'='*60}")

    if len(y_true_step2_binary) > 0:
        # Get test keys for DE samples only
        test_keys_step2 = [test_keys_step1[i] for i in range(len(test_keys_step1)) if de_mask[i]]

        analyzer_step2, error_results_step2 = run_error_analysis_from_results(
            dataset=args.dataset_name,
            test_keys=test_keys_step2,
            predictions=y_pred_step2_binary,
            true_labels=y_true_step2_binary,
            adata=None,
            task_name="step2_Up_vs_Down",
            class_names=['Down', 'Up']
        )
    else:
        print("  No DE samples to analyze.")

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
