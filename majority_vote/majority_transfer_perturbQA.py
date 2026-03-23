"""
majority_transfer_perturbQA.py
==============================
"""

import os
import numpy as np
import pandas as pd
import argparse
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

    doc_dict = {}
    for _, row in doc_df.iterrows():
        key = (row['pert'], row['gene'])
        # DOC label: 1 = up, 0 = down
        # Convert to our scheme: 1 = up, 2 = down
        doc_dict[key] = 1 if row['label'] == 1 else 2

    merged_data = []
    for _, row in de_df.iterrows():
        key = (row['pert'], row['gene'])

        if row['label'] == 0:
            final_label = 0
        else:
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

    return merged_df


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Gene-based majority vote baseline for 3-class DE prediction')
    parser.add_argument('--de_file_train', type=str, required=True, nargs='+', help='Path(s) to DE CSV file(s) (training)')
    parser.add_argument('--doc_file_train', type=str, required=True, nargs='+', help='Path(s) to DOC CSV file(s) (training)')
    parser.add_argument('--de_file_test', type=str, required=True, help='Path to DE CSV file (testing)')
    parser.add_argument('--doc_file_test', type=str, required=True, help='Path to DOC CSV file (testing)')
    parser.add_argument('--dataset_name_train', type=str, required=True, nargs='+', help='Train dataset name(s) (for output files)')
    parser.add_argument('--dataset_name_test', type=str, required=True, help='Test dataset name (for output files)')
    args = parser.parse_args()
    
    os.makedirs('error_analysis', exist_ok=True)

    print("="*70)
    print("3-Class Gene-Based Majority Vote Baseline - Transferability Experiment")
    print("="*70)

    # Build combined training data
    train_dfs = []
    for de_path, doc_path, name in zip(args.de_file_train, args.doc_file_train, args.dataset_name_train):
        print(f"\nLoading training dataset: {name}")
        merged = load_and_merge_de_doc(de_path, doc_path)
        train_dfs.append(merged[merged['split'] == 'train'])
    train_df = pd.concat(train_dfs, ignore_index=True)
    
    merged_df_test = load_and_merge_de_doc(args.de_file_test, args.doc_file_test)
    test_df = merged_df_test[merged_df_test['split'] == 'test'].reset_index(drop=True)
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

    print(f"\n{'='*70}")
    print("ERROR ANALYSIS")
    print(f"{'='*70}")

    test_keys = list(zip(test_df['pert'], test_df['gene']))
    run_error_analysis_from_results(
        dataset=args.dataset_name_test,
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
