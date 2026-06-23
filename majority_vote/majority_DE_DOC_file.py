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
from sklearn.metrics import classification_report
from typing import Dict, Tuple
from analysis_utility import (
    train_gene_majority_vote,
    predict_from_gene_majority,
    compute_3class_metrics,
    plot_data_summary
)
from error_utility import run_error_analysis_from_results


def load_de_doc_file(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load DE+DOC CSV file and return the 60/20/20 train/calib/test splits
    using the 'split_calib' column.

    Args:
        file_path: Path to DE_DOC CSV file

    Returns:
        Tuple of (train_df, calib_df, test_df)
    """
    df = pd.read_csv(file_path)

    print(f"Loaded DE+DOC file: {len(df)} perturbation-gene pairs")
    print(f"\n3-Class label distribution:")
    print(f"  Non-DE (0): {(df['label'] == 0).sum()}")
    print(f"  Up-regulated (1): {(df['label'] == 1).sum()}")
    print(f"  Down-regulated (2): {(df['label'] == 2).sum()}")

    train_df = df[df['split_calib'] == 'train']
    calib_df = df[df['split_calib'] == 'calibration']
    test_df  = df[df['split_calib'] == 'test']

    return train_df, calib_df, test_df


def print_metrics(metrics: dict, header: str):
    print(f"\n{'='*70}")
    print(header)
    print(f"{'='*70}")
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Gene-based majority vote baseline for 3-class DE prediction from DE_DOC file')
    parser.add_argument('--de_doc_file',  type=str, required=True,
                        help='Path to DE_DOC CSV file from DE_DOC_stats.py')
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='Dataset name (for output files)')
    parser.add_argument('--output_dir',   type=str,
                        default='DE_full_within_expr/results',
                        help='Output directory for results')

    args = parser.parse_args()
    error_dir = os.path.join(os.path.dirname(args.output_dir), 'error_analysis')
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(error_dir, exist_ok=True)

    print("="*70)
    print("3-Class Gene-Based Majority Vote Baseline (DE_DOC File)")
    print("="*70)

    print("\nLoading data...")
    train_df, calib_df, test_df = load_de_doc_file(args.de_doc_file)
    combined_df = pd.concat([calib_df, test_df], ignore_index=True)
    print(f"\nTrain set size:    {len(train_df)}")
    print(f"Calib set size:    {len(calib_df)}")
    print(f"Test set size:     {len(test_df)}")
    print(f"Combined set size: {len(combined_df)}")

    plot_data_summary(train_df, combined_df, args.dataset_name, output_dir=error_dir)

    print("\nTraining gene-level majority vote on train set...")
    gene_majority, gene_proba = train_gene_majority_vote(train_df, random_seed=305)
    print(f"Learned majority class for {len(gene_majority)} genes")

    print("\nPredicting on calibration set...")
    y_pred_calib, _, y_proba_calib = predict_from_gene_majority(calib_df, gene_majority, gene_proba, default_pred=0)
    y_true_calib = calib_df['label'].values

    print("\nPredicting on test set...")
    y_pred_test, _, y_proba_test = predict_from_gene_majority(test_df, gene_majority, gene_proba, default_pred=0)
    y_true_test = test_df['label'].values

    y_pred_combined  = np.concatenate([y_pred_calib, y_pred_test])
    y_proba_combined = np.concatenate([y_proba_calib, y_proba_test], axis=0)
    y_true_combined  = np.concatenate([y_true_calib, y_true_test])

    metrics_calib    = compute_3class_metrics(y_true_calib,    y_pred_calib)
    metrics_test     = compute_3class_metrics(y_true_test,     y_pred_test)
    metrics_combined = compute_3class_metrics(y_true_combined, y_pred_combined)

    print_metrics(metrics_calib,    "RESULTS — Calibration Set (20%)")
    print_metrics(metrics_test,     "RESULTS — Test Set (20%)")
    print_metrics(metrics_combined, "RESULTS — Combined Calib + Test Set (40%)")

    for split_label, df_split, metrics, y_pred, y_true, y_proba in [
        ("calib",    calib_df,    metrics_calib,    y_pred_calib,    y_true_calib,    y_proba_calib),
        ("test",     test_df,     metrics_test,     y_pred_test,     y_true_test,     y_proba_test),
        ("combined", combined_df, metrics_combined, y_pred_combined, y_true_combined, y_proba_combined),
    ]:
        pd.DataFrame([metrics]).to_csv(os.path.join(args.output_dir, f'{args.dataset_name}_metrics_{split_label}.csv'), index=False)

        pred_df = df_split.copy()
        pred_df['prediction']   = y_pred
        pred_df['correct']      = (y_pred == y_true)
        pred_df['prob_class_0'] = y_proba[:, 0]
        pred_df['prob_class_1'] = y_proba[:, 1]
        pred_df['prob_class_2'] = y_proba[:, 2]
        pred_df.to_csv(os.path.join(args.output_dir, f'{args.dataset_name}_predictions_{split_label}.csv'), index=False)

    for label, y_true, y_pred in [
        ("Calibration", y_true_calib,    y_pred_calib),
        ("Test",        y_true_test,     y_pred_test),
        ("Combined",    y_true_combined, y_pred_combined),
    ]:
        print(f"\n{'='*70}")
        print(f"Detailed Classification Report — {label} Set")
        print(f"{'='*70}")
        print(classification_report(
            y_true, y_pred,
            target_names=['Non-DE', 'Up-regulated', 'Down-regulated'],
            zero_division=0
        ))

    print(f"\n{'='*70}")
    print("ERROR ANALYSIS (Combined Calib + Test)")
    print(f"{'='*70}")
    combined_keys = list(zip(combined_df['pert'], combined_df['gene']))
    run_error_analysis_from_results(
        dataset=args.dataset_name,
        test_keys=combined_keys,
        predictions=y_pred_combined,
        true_labels=y_true_combined,
        adata=None,
        task_name="3class_DE",
        class_names=['Non-DE', 'Up-regulated', 'Down-regulated'],
        output_dir=error_dir,
    )

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
