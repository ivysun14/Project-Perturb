"""
majority_DE_DOC_file.py
=======================

Majority-vote baseline (per gene) for perturbation–gene pairs, from DE_DOC_stats.py output.

This script builds a 3-class classifier using the combined DE+DOC CSV file:
- Class 0: Non-DE
- Class 1: Up-regulated
- Class 2: Down-regulated

Input CSV format (from DE_DOC_stats.py):
- pert: perturbation name
- gene: gene name
- label: 0 (non-DE), 1 (up-regulation), 2 (down-regulation)
- p_value: BH-corrected p-value
- logFC: natural log fold change
- log2FC: log2 fold change
"""

import os
import numpy as np
import pandas as pd
import argparse
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from typing import Dict, Tuple
from analysis_utility import (
    train_gene_majority_vote,
    predict_from_gene_majority,
    compute_3class_metrics,
    plot_data_summary
)
from error_utility import run_error_analysis_from_results


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

    if 'split' in df.columns:
        train_df = df[df['split'] == 'train']
        test_df = df[df['split'] == 'test']
    else:
        # NOTE: This part is split by samples, not by perturbations!
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df['label']  # Stratified split to maintain class distribution
        )

    return train_df, test_df


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Gene-based majority vote baseline for 3-class DE prediction from DE_DOC file')
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
    print("3-Class Gene-Based Majority Vote Baseline (DE_DOC File)")
    print("="*70)

    # Load and split data
    print("\nLoading data...")
    train_df, test_df = load_de_doc_file(args.de_doc_file, test_size=args.test_size, random_state=args.random_seed)

    print(f"\nTrain set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")

    # Plot data summary
    plot_data_summary(train_df, test_df, args.dataset_name)

    # Store results from multiple runs
    all_metrics = defaultdict(list)

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

        # Train gene-level majority vote
        print("Training gene-level majority vote...")
        gene_majority = train_gene_majority_vote(train_subsample, random_seed=run)
        print(f"Learned majority class for {len(gene_majority)} genes")

        # Predict on test set
        print("Predicting on test set...")
        y_pred, _ = predict_from_gene_majority(test_df, gene_majority, default_pred=0)
        y_true = test_df['label'].values

        # Compute metrics
        metrics = compute_3class_metrics(y_true, y_pred)

        # Store metrics
        for key, value in metrics.items():
            all_metrics[key].append(value)

        # Print results for this run
        print(f"\nResults (Run {run + 1}):")
        print(f"  Accuracy:          {metrics['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"  Precision (macro): {metrics['precision_macro']:.4f}")
        print(f"  Recall (macro):    {metrics['recall_macro']:.4f}")
        print(f"  F1 (macro):        {metrics['f1_macro']:.4f}")
        print(f"\n  Per-class metrics:")
        for class_idx in range(3):
            class_name = ['Non-DE', 'Up-regulated', 'Down-regulated'][class_idx]
            print(f"    {class_name} (class {class_idx}):")
            print(f"      Precision: {metrics[f'precision_class_{class_idx}']:.4f}")
            print(f"      Recall:    {metrics[f'recall_class_{class_idx}']:.4f}")
            print(f"      F1:        {metrics[f'f1_class_{class_idx}']:.4f}")

    # Calculate mean and standard error across runs
    print(f"\n{'='*70}")
    print("FINAL RESULTS (Mean ± SE across runs)")
    print(f"{'='*70}")

    def format_mean_se(values):
        mean = np.mean(values)
        # Standard error = std / sqrt(n)
        se = np.std(values, ddof=1) / np.sqrt(len(values))
        return mean, se

    print(f"\nOverall metrics:")
    for metric_name in ['accuracy', 'balanced_accuracy', 'precision_macro', 'recall_macro', 'f1_macro']:
        mean, se = format_mean_se(all_metrics[metric_name])
        display_name = metric_name.replace('_', ' ').title()
        print(f"  {display_name:20s}: {mean:.4f} ± {se:.4f}")

    print(f"\nPer-class metrics:")
    for class_idx in range(3):
        class_name = ['Non-DE', 'Up-regulated', 'Down-regulated'][class_idx]
        print(f"  {class_name} (class {class_idx}):")
        for metric_type in ['precision', 'recall', 'f1']:
            metric_key = f'{metric_type}_class_{class_idx}'
            mean, se = format_mean_se(all_metrics[metric_key])
            print(f"    {metric_type.capitalize():10s}: {mean:.4f} ± {se:.4f}")

    # Print detailed classification report for the last run
    print(f"\n{'='*70}")
    print("Detailed Classification Report (Last Run)")
    print(f"{'='*70}")
    print(classification_report(
        y_true,
        y_pred,
        target_names=['Non-DE', 'Up-regulated', 'Down-regulated'],
        zero_division=0
    ))

    # Run error analysis on the last run
    print(f"\n{'='*70}")
    print("ERROR ANALYSIS (Last Run)")
    print(f"{'='*70}")

    # Create test_keys as list of (pert, gene) tuples
    test_keys = list(zip(test_df['pert'], test_df['gene']))

    analyzer, error_results = run_error_analysis_from_results(
        dataset=args.dataset_name,
        test_keys=test_keys,
        predictions=y_pred,
        true_labels=y_true,
        adata=None,  # No AnnData for expression analysis
        task_name="3class_DE",
        class_names=['Non-DE', 'Up-regulated', 'Down-regulated']
    )

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
