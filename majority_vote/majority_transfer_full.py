"""
majority_transfer_full.py
=========================

Transfer experiment using the majority-vote baseline on full DE+DOC dataset files.
Each input file is a single combined CSV (label: 0=Non-DE, 1=Up, 2=Down) with a
'split' column, as produced by DE_DOC_stats.py.
"""

import os
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
        DataFrame with columns: pert, gene, label (0/1/2), split
    """
    df = pd.read_csv(file_path)

    print(f"Loaded DE+DOC file: {len(df)} perturbation-gene pairs")
    print(f"  Non-DE (0): {(df['label'] == 0).sum()}")
    print(f"  Up-regulated (1): {(df['label'] == 1).sum()}")
    print(f"  Down-regulated (2): {(df['label'] == 2).sum()}")

    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Gene-based majority vote transfer baseline using full DE+DOC files')
    parser.add_argument('--de_doc_file_train', type=str, required=True, nargs='+', help='Path(s) to DE+DOC CSV file(s) (training)')
    parser.add_argument('--de_doc_file_test', type=str, required=True, help='Path to DE+DOC CSV file (testing)')
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

    print("\nLoading data...")
    train_dfs = []
    for file_path, name in zip(args.de_doc_file_train, args.dataset_name_train):
        print(f"\nLoading training dataset: {name}")
        df = load_de_doc_file(file_path)
        train_dfs.append(df[df['split'] == 'train'])

    print(f"\nLoading test dataset: {args.dataset_name_test}")
    test_full = load_de_doc_file(args.de_doc_file_test)

    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = test_full[test_full['split'] == 'test'].reset_index(drop=True)

    print(f"\nTrain set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")

    train_name = "+".join(args.dataset_name_train)
    plot_data_summary(train_df, test_df, args.dataset_name_test)

    print("Training gene-level majority vote...")
    gene_majority = train_gene_majority_vote(train_df, random_seed=0)
    print(f"Learned majority class for {len(gene_majority)} genes")

    print("Predicting on test set...")
    y_pred, _ = predict_from_gene_majority(test_df, gene_majority, default_pred=0)
    y_true = test_df['label'].values

    metrics = compute_3class_metrics(y_true, y_pred)

    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")

    print(f"\nOverall metrics:")
    print(f"  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro):    {metrics['recall_macro']:.4f}")
    print(f"  F1 (macro):        {metrics['f1_macro']:.4f}")

    print(f"\nPer-class metrics:")
    for class_idx in range(3):
        class_name = ['Non-DE', 'Up-regulated', 'Down-regulated'][class_idx]
        print(f"  {class_name} (class {class_idx}):")
        print(f"    Precision: {metrics[f'precision_class_{class_idx}']:.4f}")
        print(f"    Recall:    {metrics[f'recall_class_{class_idx}']:.4f}")
        print(f"    F1:        {metrics[f'f1_class_{class_idx}']:.4f}")

    print(f"\n{'='*70}")
    print("Detailed Classification Report")
    print(f"{'='*70}")
    print(classification_report(
        y_true,
        y_pred,
        target_names=['Non-DE', 'Up-regulated', 'Down-regulated'],
        zero_division=0
    ))

    predictions_df = test_df.copy()
    predictions_df['prediction'] = y_pred
    predictions_df['correct'] = (y_pred == y_true)
    predictions_path = os.path.join(args.output_dir, f'{args.dataset_name_test}_majority_transfer_predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)
    print(f"\nPredictions saved to {predictions_path}")

    metrics_df = pd.DataFrame([metrics])
    metrics_df['train_dataset'] = train_name
    metrics_df['test_dataset'] = args.dataset_name_test
    metrics_path = os.path.join(args.output_dir, f'{args.dataset_name_test}_majority_transfer_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")

    print(f"\n{'='*70}")
    print("ERROR ANALYSIS")
    print(f"{'='*70}")

    test_keys = list(zip(test_df['pert'], test_df['gene']))
    run_error_analysis_from_results(
        dataset=args.dataset_name_test,
        test_keys=test_keys,
        predictions=y_pred,
        true_labels=y_true,
        adata=None,
        task_name="3class_DE",
        class_names=['Non-DE', 'Up-regulated', 'Down-regulated']
    )

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
