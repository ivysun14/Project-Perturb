"""
majority_transfer_perturbQA.py
================================
Gene-based majority vote baseline for PerturbQA cross-cell-line evaluation.

Training: ALL (perturbation, gene) pairs from 3 source cell lines — both 'train'
and 'test' splits — restricted to the common gene universe across all 4 datasets.
Evaluation: 'test' split of the held-out cell line, restricted to the same gene universe.

When the same (perturbation, gene) pair appears in multiple source cell lines,
the majority vote is trained on the combined pool; the per-gene majority class
is computed from all source observations together.
"""

import os
import sys
import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import classification_report
from analysis_utility import (
    train_gene_majority_vote,
    predict_from_gene_majority,
    plot_data_summary,
)
from error_utility import run_error_analysis_from_results
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics.class_metrics import compute_3class_metrics


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
    print(f"  Loaded {os.path.basename(de_path)}: {len(merged)} pairs  "
          f"Non-DE={( merged['label']==0).sum()}  "
          f"Up={(merged['label']==1).sum()}  "
          f"Down={(merged['label']==2).sum()}")
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
        description='Gene-based majority vote baseline for PerturbQA cross-cell-line evaluation'
    )
    parser.add_argument('--de_file_train',  type=str, required=True, nargs='+',
                        help='DE CSV file(s) for source cell lines')
    parser.add_argument('--dir_file_train', type=str, required=True, nargs='+',
                        help='Direction CSV file(s) for source cell lines, same order as --de_file_train')
    parser.add_argument('--de_file_test',   type=str, required=True,
                        help='DE CSV file for the held-out cell line')
    parser.add_argument('--dir_file_test',  type=str, required=True,
                        help='Direction CSV file for the held-out cell line')
    parser.add_argument('--gene_universe_file', type=str, required=True,
                        help='CSV with column "gene_name" listing the common gene universe')
    parser.add_argument('--dataset_name_train', type=str, required=True, nargs='+',
                        help='Source dataset name(s)')
    parser.add_argument('--dataset_name_test',  type=str, required=True,
                        help='Held-out dataset name')
    parser.add_argument('--output_dir', type=str,
                        default='DE_PerturbQA_cross_expr/results',
                        help='Output directory for results')
    args = parser.parse_args()

    if len(args.de_file_train) != len(args.dir_file_train):
        raise ValueError("--de_file_train and --dir_file_train must have the same length")
    if len(args.de_file_train) != len(args.dataset_name_train):
        raise ValueError("--de_file_train and --dataset_name_train must have the same length")

    error_dir = os.path.join(os.path.dirname(args.output_dir), 'error_analysis')
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(error_dir, exist_ok=True)

    print("="*70)
    print("3-Class Gene-Based Majority Vote Baseline (PerturbQA Cross-Cell-Line)")
    print("="*70)

    # Load common gene universe
    gene_universe = set(pd.read_csv(args.gene_universe_file)['gene_name'])
    print(f"\nCommon gene universe: {len(gene_universe)} genes")

    # Load and pool ALL pairs from source cell lines, restricted to common genes
    print(f"\n{'='*70}")
    print("Loading source datasets (all splits)")
    print(f"{'='*70}")
    train_dfs = []
    for de_fp, dir_fp, name in zip(args.de_file_train, args.dir_file_train,
                                   args.dataset_name_train):
        print(f"\n[{name}]")
        df = load_and_merge_de_dir(de_fp, dir_fp)
        df = df[df['gene'].isin(gene_universe)].reset_index(drop=True)
        print(f"  After gene filter: {len(df)} pairs  "
              f"({df['pert'].nunique()} perts, {df['gene'].nunique()} genes)")
        train_dfs.append(df)

    train_df = pd.concat(train_dfs, ignore_index=True)
    print(f"\nTotal training set: {len(train_df)} pairs  "
          f"({train_df['pert'].nunique()} perts, {train_df['gene'].nunique()} genes)")

    # Load held-out cell line test split, restricted to common genes
    print(f"\n{'='*70}")
    print(f"Loading held-out dataset: {args.dataset_name_test} (test split)")
    print(f"{'='*70}")
    test_full = load_and_merge_de_dir(args.de_file_test, args.dir_file_test)
    test_df   = test_full[test_full['split'] == 'test'].reset_index(drop=True)
    test_df   = test_df[test_df['gene'].isin(gene_universe)].reset_index(drop=True)
    print(f"After gene filter: {len(test_df)} pairs  "
          f"({test_df['pert'].nunique()} perts, {test_df['gene'].nunique()} genes)")

    train_name = "+".join(args.dataset_name_train)
    plot_data_summary(train_df, test_df, args.dataset_name_test,
                      output_dir=error_dir)

    print("\nTraining gene-level majority vote on source cell lines (all pairs)...")
    gene_majority, gene_proba = train_gene_majority_vote(train_df, random_seed=305)
    print(f"Learned majority class for {len(gene_majority)} genes")

    print("\nPredicting on held-out test split...")
    y_pred, _, y_proba = predict_from_gene_majority(
        test_df, gene_majority, gene_proba, default_pred=0
    )
    y_true = test_df['label'].values

    metrics = compute_3class_metrics(y_true, y_pred, y_proba=y_proba)
    print_metrics(metrics, f"RESULTS — {args.dataset_name_test} (transfer: {train_name})")

    print(f"\n{'='*70}")
    print("Detailed Classification Report")
    print(f"{'='*70}")
    print(classification_report(
        y_true, y_pred,
        target_names=['Non-DE', 'Up-regulated', 'Down-regulated'],
        zero_division=0
    ))

    # Save metrics CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df['train_dataset'] = train_name
    metrics_df['test_dataset']  = args.dataset_name_test
    metrics_df.to_csv(
        os.path.join(args.output_dir,
                     f'{args.dataset_name_test}_majority_transfer_metrics.csv'),
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
        os.path.join(args.output_dir,
                     f'{args.dataset_name_test}_majority_transfer_predictions.csv'),
        index=False
    )
    print(f"\nMetrics and predictions saved to {args.output_dir}/")

    print(f"\n{'='*70}")
    print("ERROR ANALYSIS")
    print(f"{'='*70}")
    run_error_analysis_from_results(
        dataset=args.dataset_name_test,
        test_keys=list(zip(test_df['pert'], test_df['gene'])),
        predictions=y_pred,
        true_labels=y_true,
        adata=None,
        task_name="3class_MV_transfer_perturbQA",
        class_names=['Non-DE', 'Up-regulated', 'Down-regulated'],
        output_dir=error_dir,
    )

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
