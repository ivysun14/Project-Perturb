"""
majority_transfer_full.py
=========================

Transfer experiment using the majority-vote baseline on full DE+DOC dataset files.
Each input file is a single combined CSV (label: 0=Non-DE, 1=Up, 2=Down) with a
'split_calib' column (60/20/20 train/calibration/test split).

Training uses the train+test splits of the 3 source cell lines (80% of each).
Evaluation is performed separately on the held-out cell line's calibration set,
test set, and their combination.
"""

import os
import json
import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import classification_report
from analysis_utility import (
    train_gene_majority_vote,
    predict_from_gene_majority,
    compute_3class_metrics,
    plot_data_summary
)
from error_utility import run_error_analysis_from_results


def load_de_doc_file(file_path: str) -> pd.DataFrame:
    """
    Load a combined DE+DOC CSV file.

    Args:
        file_path: Path to DE+DOC CSV file (output of DE_DOC_stats.py)

    Returns:
        Full DataFrame with split_calib column
    """
    df = pd.read_csv(file_path)

    print(f"Loaded DE+DOC file: {len(df)} perturbation-gene pairs")
    print(f"  Non-DE (0): {(df['label'] == 0).sum()}")
    print(f"  Up-regulated (1): {(df['label'] == 1).sum()}")
    print(f"  Down-regulated (2): {(df['label'] == 2).sum()}")

    return df


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

    parser = argparse.ArgumentParser(description='Gene-based majority vote transfer baseline using full DE+DOC files')
    parser.add_argument('--de_doc_file_train', type=str, required=True, nargs='+', help='Path(s) to DE+DOC CSV file(s) (training cell lines)')
    parser.add_argument('--de_doc_file_test', type=str, required=True, help='Path to DE+DOC CSV file (held-out test cell line)')
    parser.add_argument('--dataset_name_train', type=str, required=True, nargs='+', help='Train dataset name(s) (for output files)')
    parser.add_argument('--dataset_name_test', type=str, required=True, help='Test dataset name (for output files)')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory for results')
    args = parser.parse_args()

    if len(args.de_doc_file_train) != len(args.dataset_name_train):
        raise ValueError("Number of --de_doc_file_train paths must match number of --dataset_name_train names")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('error_analysis', exist_ok=True)

    print("="*70)
    print("3-Class Gene-Based Majority Vote Baseline - Transferability Experiment")
    print("="*70)

    # Load and pool train+test splits from all source cell lines
    print("\nLoading training data...")
    train_dfs = []
    for file_path, name in zip(args.de_doc_file_train, args.dataset_name_train):
        print(f"\nLoading training dataset: {name}")
        df = load_de_doc_file(file_path)
        # Use train (60%) + test (20%) splits of source cell lines for training
        source_train = df[df['split_calib'].isin(['train', 'test'])]
        print(f"  Using {len(source_train)} rows (train+test splits)")
        train_dfs.append(source_train)

    train_df = pd.concat(train_dfs, ignore_index=True)
    print(f"\nTotal training set size: {len(train_df)}")

    # Load held-out cell line and split into calib / test
    print(f"\nLoading held-out test dataset: {args.dataset_name_test}")
    test_full = load_de_doc_file(args.de_doc_file_test)
    calib_df   = test_full[test_full['split_calib'] == 'calibration'].reset_index(drop=True)
    test_df    = test_full[test_full['split_calib'] == 'test'].reset_index(drop=True)
    combined_df = pd.concat([calib_df, test_df], ignore_index=True)

    print(f"  Calib set size:    {len(calib_df)}")
    print(f"  Test set size:     {len(test_df)}")
    print(f"  Combined set size: {len(combined_df)}")

    train_name = "+".join(args.dataset_name_train)
    plot_data_summary(train_df, combined_df, args.dataset_name_test)

    print("\nTraining gene-level majority vote on source cell lines (train+test)...")
    gene_majority, gene_proba = train_gene_majority_vote(train_df, random_seed=305)
    print(f"Learned majority class for {len(gene_majority)} genes")

    # Predict on each held-out split and track unseen genes
    print("\nPredicting on calibration set...")
    y_pred_calib, unseen_calib, y_proba_calib = predict_from_gene_majority(calib_df, gene_majority, gene_proba, default_pred=0)
    y_true_calib = calib_df['label'].values

    print("\nPredicting on test set...")
    y_pred_test, unseen_test, y_proba_test = predict_from_gene_majority(test_df, gene_majority, gene_proba, default_pred=0)
    y_true_test = test_df['label'].values

    y_pred_combined  = np.concatenate([y_pred_calib, y_pred_test])
    y_proba_combined = np.concatenate([y_proba_calib, y_proba_test], axis=0)
    y_true_combined  = np.concatenate([y_true_calib, y_true_test])
    unseen_combined  = np.concatenate([unseen_calib, unseen_test])

    # Save unseen genes JSON
    unseen_genes = sorted(combined_df['gene'][unseen_combined].unique().tolist())
    unseen_info = {'unseen_genes': unseen_genes, 'n_unseen_genes': len(unseen_genes)}
    unseen_path = os.path.join(args.output_dir, f'{args.dataset_name_test}_transfer_unseen.json')
    with open(unseen_path, 'w') as f:
        json.dump(unseen_info, f, indent=2)
    print(f"\nUnseen genes ({len(unseen_genes)}) saved to {unseen_path}")

    # Compute metrics for all 3 evaluation sets
    metrics_calib    = compute_3class_metrics(y_true_calib,    y_pred_calib)
    metrics_test     = compute_3class_metrics(y_true_test,     y_pred_test)
    metrics_combined = compute_3class_metrics(y_true_combined, y_pred_combined)

    print_metrics(metrics_calib,    "RESULTS — Calibration Set (20%)")
    print_metrics(metrics_test,     "RESULTS — Test Set (20%)")
    print_metrics(metrics_combined, "RESULTS — Combined Calib + Test Set (40%)")

    # Save 3 metrics CSVs and 3 predictions CSVs (with per-class probabilities)
    for split_label, df_split, metrics, y_pred, y_true, y_proba in [
        ("calib",    calib_df,    metrics_calib,    y_pred_calib,    y_true_calib,    y_proba_calib),
        ("test",     test_df,     metrics_test,     y_pred_test,     y_true_test,     y_proba_test),
        ("combined", combined_df, metrics_combined, y_pred_combined, y_true_combined, y_proba_combined),
    ]:
        metrics_df = pd.DataFrame([metrics])
        metrics_df['train_dataset'] = train_name
        metrics_df['test_dataset']  = args.dataset_name_test
        metrics_path = os.path.join(args.output_dir, f'{args.dataset_name_test}_transfer_metrics_{split_label}.csv')
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Metrics ({split_label}) saved to {metrics_path}")

        pred_df = df_split.copy()
        pred_df['prediction']   = y_pred
        pred_df['correct']      = (y_pred == y_true)
        pred_df['prob_class_0'] = y_proba[:, 0]
        pred_df['prob_class_1'] = y_proba[:, 1]
        pred_df['prob_class_2'] = y_proba[:, 2]
        pred_path = os.path.join(args.output_dir, f'{args.dataset_name_test}_transfer_predictions_{split_label}.csv')
        pred_df.to_csv(pred_path, index=False)
        print(f"Predictions ({split_label}) saved to {pred_path}")

    # Classification reports
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

    # Error analysis on combined calib+test
    print(f"\n{'='*70}")
    print("ERROR ANALYSIS (Combined Calib + Test)")
    print(f"{'='*70}")
    combined_keys = list(zip(combined_df['pert'], combined_df['gene']))
    run_error_analysis_from_results(
        dataset=args.dataset_name_test,
        test_keys=combined_keys,
        predictions=y_pred_combined,
        true_labels=y_true_combined,
        adata=None,
        task_name="3class_DE",
        class_names=['Non-DE', 'Up-regulated', 'Down-regulated']
    )

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
