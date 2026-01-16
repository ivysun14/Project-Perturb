"""
majority_transfer.py
=====================
"""

import os
import numpy as np
import pandas as pd
import argparse
from collections import defaultdict
from sklearn.metrics import classification_report
from typing import Dict, List, Tuple
from analysis_utility import (
    train_gene_majority_vote,
    predict_from_gene_majority,
    compute_3class_metrics,
    plot_data_summary
)
from error_utility import run_error_analysis_from_results


def load_and_merge_de_doc(de_path: str, doc_path: str) -> pd.DataFrame:
    """
    Load DE and DOC files and merge into 3-class labels.

    Args:
        de_path: Path to DE CSV file
        doc_path: Path to DOC CSV file

    Returns:
        DataFrame with columns: pert, gene, label (0/1/2), split where label is:
        - 0: Non-DE
        - 1: Up-regulated (DE with positive change)
        - 2: Down-regulated (DE with negative change)
    """

    de_df = pd.read_csv(de_path)
    doc_df = pd.read_csv(doc_path)

    print(f"Loaded DE file: {len(de_df)} pairs")
    print(f"Loaded DOC file: {len(doc_df)} pairs")

    # Create a mapping from (pert, gene) to DOC label
    doc_dict = {}
    for _, row in doc_df.iterrows():
        key = (row['pert'], row['gene'])
        # DOC label: 1 = up, 0 = down
        # Convert to our scheme: 1 = up, 2 = down
        doc_dict[key] = 1 if row['label'] == 1 else 2

    # Process DE file and assign 3-class labels
    merged_data = []
    for _, row in de_df.iterrows():
        key = (row['pert'], row['gene'])

        if row['label'] == 0:
            # Non-DE
            final_label = 0
        else:
            # DE gene - check DOC file for direction
            if key in doc_dict:
                final_label = doc_dict[key]
            else:
                # NOTE: This shouldn't happen based on assumption but currently default to non-DE
                print(f"Warning: DE gene {key} not found in DOC file")
                final_label = 0

        merged_data.append({
            'pert': row['pert'],
            'gene': row['gene'],
            'label': final_label,
            'split': row['split']
        })

    merged_df = pd.DataFrame(merged_data)

    print(f"\n3-Class label distribution:")
    print(f"  Non-DE (0): {(merged_df['label'] == 0).sum()}")
    print(f"  Up-regulated (1): {(merged_df['label'] == 1).sum()}")
    print(f"  Down-regulated (2): {(merged_df['label'] == 2).sum()}")

    return merged_df


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Gene-based majority vote baseline for 3-class DE prediction')
    parser.add_argument('--de_file_train', type=str, required=True, help='Path to DE CSV file (training)')
    parser.add_argument('--doc_file_train', type=str, required=True, help='Path to DOC CSV file (training)')
    parser.add_argument('--de_file_test', type=str, required=True, help='Path to DE CSV file (testing)')
    parser.add_argument('--doc_file_test', type=str, required=True, help='Path to DOC CSV file (testing)')
    parser.add_argument('--dataset_name_train', type=str, required=True, help='Train dataset name (for output files)')
    parser.add_argument('--dataset_name_test', type=str, required=True, help='Test dataset name (for output files)')
    parser.add_argument('--n_runs', type=int, default=3, help='Number of runs with different train subsampling (default: 3)')
    parser.add_argument('--train_fraction', type=float, default=0.8, help='Fraction of training data to use in each run (default: 0.8)')

    args = parser.parse_args()

    # Create error_analysis directory if it doesn't exist
    os.makedirs('error_analysis', exist_ok=True)

    print("="*70)
    print("3-Class Gene-Based Majority Vote Baseline - Transferability Experiment")
    print("="*70)

    # Load and merge data
    print("\nLoading data...")
    merged_df_train = load_and_merge_de_doc(args.de_file_train, args.doc_file_train)
    merged_df_test = load_and_merge_de_doc(args.de_file_test, args.doc_file_test)

    # Split into train and test
    train_df = merged_df_train[merged_df_train['split'] == 'train'].reset_index(drop=True)
    test_df = merged_df_test[merged_df_test['split'] == 'test'].reset_index(drop=True)

    print(f"\nTrain set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")

    # Plot data summary
    plot_data_summary(train_df, test_df, args.dataset_name_train + "_to_" + args.dataset_name_test)

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
        y_pred = predict_from_gene_majority(test_df, gene_majority, default_pred=0)
        y_true = test_df['label'].values

        # Compute metrics
        metrics = compute_3class_metrics(y_true, y_pred)

        # Store metrics
        for key, value in metrics.items():
            all_metrics[key].append(value)

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
        dataset=args.dataset_name_train + "_to_" + args.dataset_name_test,
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
