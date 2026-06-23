"""
majority_perturbQA_DE_file.py
==============================
Gene-based majority vote baseline for PerturbQA within-cell-line evaluation.

Input: PerturbQA two-file format — {ds}-de.csv (binary DE/non-DE) and {ds}-dir.csv
(direction: 1=up, 0=down for DE pairs). Merged into 3-class labels at load time.

Training: 'train' split of the same cell line.
Evaluation: 'test' split of the same cell line.
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
    plot_data_summary,
)
from error_utility import run_error_analysis_from_results


def load_and_merge_de_dir(de_path: str, dir_path: str) -> pd.DataFrame:
    """
    Load PerturbQA DE and direction files and merge into 3-class labels.
    Returns DataFrame with columns: pert, gene, label (0=non-DE/1=up/2=down), split.
    """
    de_df  = pd.read_csv(de_path)
    dir_df = pd.read_csv(dir_path)

    dir_dict = {(r['pert'], r['gene']): (1 if r['label'] == 1 else 2)
                for _, r in dir_df.iterrows()}

    rows = []
    for _, row in de_df.iterrows():
        key = (row['pert'], row['gene'])
        rows.append({
            'pert':  row['pert'],
            'gene':  row['gene'],
            'label': 0 if row['label'] == 0 else dir_dict.get(key, 0),
            'split': row['split'],
        })

    merged = pd.DataFrame(rows)
    print(f"Loaded {os.path.basename(de_path)}: {len(merged)} pairs")
    print(f"  Non-DE (0): {(merged['label']==0).sum()}  "
          f"Up (1): {(merged['label']==1).sum()}  "
          f"Down (2): {(merged['label']==2).sum()}")
    return merged


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
    for cls in range(3):
        cname = ['Non-DE', 'Up-regulated', 'Down-regulated'][cls]
        print(f"    {cname} (class {cls}):")
        print(f"      Precision: {metrics[f'precision_class_{cls}']:.4f}")
        print(f"      Recall:    {metrics[f'recall_class_{cls}']:.4f}")
        print(f"      F1:        {metrics[f'f1_class_{cls}']:.4f}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Gene-based majority vote baseline for PerturbQA within-cell-line evaluation'
    )
    parser.add_argument('--de_file',       type=str, required=True,
                        help='Path to PerturbQA DE CSV file ({ds}-de.csv)')
    parser.add_argument('--dir_file',      type=str, required=True,
                        help='Path to PerturbQA direction CSV file ({ds}-dir.csv)')
    parser.add_argument('--dataset_name',  type=str, required=True,
                        help='Dataset name (for output files)')
    parser.add_argument('--output_dir',    type=str,
                        default='DE_PerturbQA_within_expr/results',
                        help='Output directory for results')
    args = parser.parse_args()

    error_dir = os.path.join(os.path.dirname(args.output_dir), 'error_analysis')
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(error_dir, exist_ok=True)

    print("="*70)
    print("3-Class Gene-Based Majority Vote Baseline (PerturbQA Within)")
    print("="*70)
    print(f"\nDataset: {args.dataset_name}")

    merged_df = load_and_merge_de_dir(args.de_file, args.dir_file)
    train_df  = merged_df[merged_df['split'] == 'train'].reset_index(drop=True)
    test_df   = merged_df[merged_df['split'] == 'test'].reset_index(drop=True)
    print(f"\nTrain set: {len(train_df)} pairs  ({train_df['pert'].nunique()} perts)")
    print(f"Test set:  {len(test_df)} pairs  ({test_df['pert'].nunique()} perts)")

    plot_data_summary(train_df, test_df, args.dataset_name,
                      output_dir=error_dir)

    print("\nTraining gene-level majority vote on train split...")
    gene_majority, gene_proba = train_gene_majority_vote(train_df, random_seed=305)
    print(f"Learned majority class for {len(gene_majority)} genes")

    print("\nPredicting on test split...")
    y_pred, _, y_proba = predict_from_gene_majority(
        test_df, gene_majority, gene_proba, default_pred=0
    )
    y_true = test_df['label'].values

    metrics = compute_3class_metrics(y_true, y_pred, y_proba=y_proba)
    print_metrics(metrics, f"RESULTS — {args.dataset_name} test split")

    print(f"\n{'='*70}")
    print("Detailed Classification Report")
    print(f"{'='*70}")
    print(classification_report(
        y_true, y_pred,
        target_names=['Non-DE', 'Up-regulated', 'Down-regulated'],
        zero_division=0
    ))

    # Save metrics CSV
    pd.DataFrame([metrics]).to_csv(
        os.path.join(args.output_dir, f'{args.dataset_name}_majority_metrics.csv'),
        index=False
    )

    # Save predictions CSV
    pred_df = test_df.copy()
    pred_df['prediction']   = y_pred
    pred_df['correct']      = (y_pred == y_true)
    pred_df['prob_class_0'] = y_proba[:, 0]
    pred_df['prob_class_1'] = y_proba[:, 1]
    pred_df['prob_class_2'] = y_proba[:, 2]
    pred_df.to_csv(
        os.path.join(args.output_dir, f'{args.dataset_name}_majority_predictions.csv'),
        index=False
    )
    print(f"\nMetrics and predictions saved to {args.output_dir}/")

    print(f"\n{'='*70}")
    print("ERROR ANALYSIS")
    print(f"{'='*70}")
    run_error_analysis_from_results(
        dataset=args.dataset_name,
        test_keys=list(zip(test_df['pert'], test_df['gene'])),
        predictions=y_pred,
        true_labels=y_true,
        adata=None,
        task_name="3class_MV_perturbQA",
        class_names=['Non-DE', 'Up-regulated', 'Down-regulated'],
        output_dir=error_dir,
    )

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
