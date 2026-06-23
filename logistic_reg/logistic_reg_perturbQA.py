"""
logistic_reg_perturbQA.py
=========================
Per-gene logistic regression for 3-class perturbation-gene prediction on
PerturbQA data (sparse two-file format).

Modes:
  Within-cell-line:
    --de_file {ds}-de.csv --dir_file {ds}-dir.csv --dataset_name DS
    Train on 'train' split; evaluate on 'test' split.
    No calibration split → single metrics/predictions file per dataset.

  Cross-cell-line:
    --de_file_train <f1> [f2 f3] --dir_file_train <f1> [f2 f3]
    --de_file_test <f> --dir_file_test <f>
    --gene_universe_file <csv>  --dataset_name_test DS
    Train on ALL (pert, gene) pairs from 3 source cell lines (both splits),
    restricted to the common gene universe. Duplicate (pert, gene) pairs
    across cell lines are resolved by majority vote (same as full cross).
    Evaluate on 'test' split of held-out cell line, restricted to gene universe.
    No unseen-gene JSON (common universe guarantees coverage).

Shared helpers (load_embeddings, build_feature_matrix, build_label_matrix,
merge_train_sets, fit_per_gene_models, predict_pairs) are imported from
logistic_reg.py in the same directory.
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logistic_reg import (
    load_embeddings,
    build_feature_matrix,
    build_label_matrix,
    merge_train_sets,
    fit_per_gene_models,
    predict_pairs,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from majority_vote.analysis_utility import compute_3class_metrics, plot_data_summary
from majority_vote.error_utility import run_error_analysis_from_results


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
    if 'auroc_ovr_macro' in metrics:
        print(f"  AUROC OvR (macro): {metrics['auroc_ovr_macro']:.4f}")
    print(f"\n  Per-class metrics:")
    for cls in range(3):
        cname = ['Non-DE', 'Up-regulated', 'Down-regulated'][cls]
        print(f"  {cname} (class {cls}):")
        print(f"    Precision: {metrics[f'precision_class_{cls}']:.4f}")
        print(f"    Recall:    {metrics[f'recall_class_{cls}']:.4f}")
        print(f"    F1:        {metrics[f'f1_class_{cls}']:.4f}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Per-gene logistic regression for PerturbQA 3-class DE prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Within-cell-line mode
    parser.add_argument('--de_file',  type=str, default=None,
                        help='(within) PerturbQA DE CSV ({ds}-de.csv)')
    parser.add_argument('--dir_file', type=str, default=None,
                        help='(within) PerturbQA direction CSV ({ds}-dir.csv)')
    parser.add_argument('--dataset_name', type=str, default=None,
                        help='(within) Dataset name for output file naming')
    # Cross-cell-line mode
    parser.add_argument('--de_file_train',  type=str, nargs='+', default=None,
                        help='(cross) DE CSV file(s) for source cell lines')
    parser.add_argument('--dir_file_train', type=str, nargs='+', default=None,
                        help='(cross) Direction CSV file(s) for source cell lines')
    parser.add_argument('--de_file_test',   type=str, default=None,
                        help='(cross) DE CSV file for held-out cell line')
    parser.add_argument('--dir_file_test',  type=str, default=None,
                        help='(cross) Direction CSV file for held-out cell line')
    parser.add_argument('--gene_universe_file', type=str, default=None,
                        help='(cross) CSV with column "gene_name" for common gene universe')
    parser.add_argument('--dataset_name_test', type=str, default=None,
                        help='(cross) Held-out dataset name for output file naming')
    # Shared
    parser.add_argument('--embedding_file',  type=str, required=True,
                        help='Path to primary GenePT embedding pkl')
    parser.add_argument('--embedding_file2', type=str, default=None,
                        help='Path to secondary GenePT embedding pkl (merged into primary)')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for predictions and metrics')
    parser.add_argument('--Cs',       type=str, default='0.01',
                        help='Comma-separated C values (C=1/lambda); single value skips CV')
    parser.add_argument('--cv',       type=int, default=5,
                        help='CV folds for C selection (ignored if single C)')
    parser.add_argument('--max_iter', type=int, default=2000,
                        help='Max solver iterations')
    parser.add_argument('--random_seed', type=int, default=305)

    args = parser.parse_args()

    within_mode = args.de_file is not None
    cross_mode  = args.de_file_train is not None

    if within_mode == cross_mode:
        parser.error("Specify either --de_file (within) or --de_file_train (cross), not both/neither.")
    if within_mode and args.dataset_name is None:
        parser.error("--dataset_name is required in within mode.")
    if cross_mode:
        for flag in ('--dir_file_train', '--de_file_test', '--dir_file_test',
                     '--gene_universe_file', '--dataset_name_test'):
            attr = flag.lstrip('-').replace('-', '_')
            if getattr(args, attr) is None:
                parser.error(f"{flag} is required in cross mode.")
        if len(args.de_file_train) != len(args.dir_file_train):
            parser.error("--de_file_train and --dir_file_train must have the same length.")

    error_dir = os.path.join(os.path.dirname(args.output_dir), 'error_analysis')
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(error_dir,       exist_ok=True)

    Cs = [float(c) for c in args.Cs.split(',')]

    print("=" * 70)
    print("Per-Gene Logistic Regression: PerturbQA")
    print("=" * 70)

    # ------------------------------------------------------------------ #
    # Data loading                                                         #
    # ------------------------------------------------------------------ #
    if within_mode:
        print(f"\nMode:    within-cell-line  |  Dataset: {args.dataset_name}")

        merged = load_and_merge_de_dir(args.de_file, args.dir_file)
        train_df = merged[merged['split'] == 'train'].reset_index(drop=True)
        test_df  = merged[merged['split'] == 'test'].reset_index(drop=True)
        print(f"\nTrain: {len(train_df)} pairs ({train_df['pert'].nunique()} perts, "
              f"{train_df['gene'].nunique()} genes)")
        print(f"Test:  {len(test_df)} pairs ({test_df['pert'].nunique()} perts, "
              f"{test_df['gene'].nunique()} genes)")

        ds_name   = args.dataset_name
        task_name = "3class_logreg_perturbQA_within"

    else:
        ds_name = args.dataset_name_test
        print(f"\nMode:    cross-cell-line  |  Held-out: {ds_name}")

        gene_universe = set(pd.read_csv(args.gene_universe_file)['gene_name'])
        print(f"Common gene universe: {len(gene_universe)} genes")

        print(f"\n{'='*70}")
        print("Loading source datasets (all splits)")
        print(f"{'='*70}")
        src_dfs = []
        for de_fp, dir_fp in zip(args.de_file_train, args.dir_file_train):
            df = load_and_merge_de_dir(de_fp, dir_fp)
            df = df[df['gene'].isin(gene_universe)].reset_index(drop=True)
            print(f"    After gene filter: {len(df)} pairs  "
                  f"({df['pert'].nunique()} perts, {df['gene'].nunique()} genes)")
            src_dfs.append(df)

        print(f"\nMerging source pairs via majority vote on duplicate (pert, gene) pairs ...")
        train_df = merge_train_sets(src_dfs, args.random_seed)
        print(f"  Train set after merge: {len(train_df)} unique pairs  "
              f"({train_df['pert'].nunique()} perts, {train_df['gene'].nunique()} genes)")

        print(f"\n{'='*70}")
        print(f"Loading held-out dataset: {ds_name} (test split)")
        print(f"{'='*70}")
        held_out = load_and_merge_de_dir(args.de_file_test, args.dir_file_test)
        test_df  = held_out[held_out['split'] == 'test'].reset_index(drop=True)
        test_df  = test_df[test_df['gene'].isin(gene_universe)].reset_index(drop=True)
        print(f"  After gene filter: {len(test_df)} pairs  "
              f"({test_df['pert'].nunique()} perts, {test_df['gene'].nunique()} genes)")

        task_name = "3class_logreg_perturbQA_cross"

    plot_data_summary(train_df, test_df, ds_name, output_dir=error_dir)

    # ------------------------------------------------------------------ #
    # Build training matrices and fit models                               #
    # ------------------------------------------------------------------ #
    embedding_dict = load_embeddings(args.embedding_file, args.embedding_file2)

    train_perts = sorted(train_df['pert'].unique())
    train_genes = sorted(train_df['gene'].unique())
    print(f"\nBuilding feature matrix X ({len(train_perts)} train perts × 3072) ...")
    X_train, _, valid_train_perts = build_feature_matrix(train_perts, embedding_dict)
    print(f"  X_train shape: {X_train.shape}")

    print(f"Building label matrix Y ({len(valid_train_perts)} perts × {len(train_genes)} genes) ...")
    Y_train = build_label_matrix(train_df, valid_train_perts, train_genes)
    n_obs = (~np.isnan(Y_train)).sum()
    print(f"  Y_train shape: {Y_train.shape}  ({n_obs} observed labels, "
          f"{np.isnan(Y_train).sum()} NaN)")

    print(f"\nFitting per-gene models (Cs={Cs}, cv={args.cv}, max_iter={args.max_iter}) ...")
    models = fit_per_gene_models(X_train, Y_train, train_genes, Cs, args.cv,
                                 args.max_iter, args.random_seed)

    # ------------------------------------------------------------------ #
    # Predict on test split                                                #
    # ------------------------------------------------------------------ #
    test_perts = sorted(test_df['pert'].unique())
    print(f"\nBuilding test embedding matrix ({len(test_perts)} perts) ...")
    X_test, _, valid_test_perts = build_feature_matrix(test_perts, embedding_dict)
    pert_to_row = {p: i for i, p in enumerate(valid_test_perts)}

    print(f"\nPredicting on test set ({len(test_df)} pairs) ...")
    y_pred, y_proba, valid_mask, no_model_genes = predict_pairs(
        test_df, models, pert_to_row, X_test
    )
    test_clean = test_df[valid_mask].reset_index(drop=True)
    y_true = test_clean['label'].values
    print(f"  Valid predictions: {valid_mask.sum()}/{len(test_df)}  "
          f"(defaulted genes: {len(no_model_genes)})")

    # ------------------------------------------------------------------ #
    # Metrics                                                              #
    # ------------------------------------------------------------------ #
    metrics = compute_3class_metrics(y_true, y_pred, y_proba=y_proba)
    print_metrics(metrics, f"RESULTS — {ds_name} test split")

    print(f"\n{'='*70}")
    print("Detailed Classification Report")
    print(f"{'='*70}")
    print(classification_report(
        y_true, y_pred,
        target_names=['Non-DE', 'Up-regulated', 'Down-regulated'],
        zero_division=0
    ))

    # ------------------------------------------------------------------ #
    # Save outputs                                                         #
    # ------------------------------------------------------------------ #
    gene_to_best_C = {
        gene: float(np.mean(m.C_)) if hasattr(m, 'C_') else np.nan
        for gene, m in models.items()
    }

    if within_mode:
        metrics_fname = f'{ds_name}_logreg_metrics.csv'
        pred_fname    = f'{ds_name}_logreg_predictions.csv'
    else:
        metrics_fname = f'{ds_name}_logreg_transfer_metrics.csv'
        pred_fname    = f'{ds_name}_logreg_transfer_predictions.csv'

    metrics_df = pd.DataFrame([metrics])
    metrics_df['dataset'] = ds_name
    metrics_path = os.path.join(args.output_dir, metrics_fname)
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nMetrics saved to {metrics_path}")

    pred_df = test_clean.copy()
    pred_df['prediction']   = y_pred
    pred_df['correct']      = (y_pred == y_true)
    pred_df['prob_class_0'] = y_proba[:, 0]
    pred_df['prob_class_1'] = y_proba[:, 1]
    pred_df['prob_class_2'] = y_proba[:, 2]
    pred_df['best_C']       = pred_df['gene'].map(gene_to_best_C)
    pred_path = os.path.join(args.output_dir, pred_fname)
    pred_df.to_csv(pred_path, index=False)
    print(f"Predictions saved to {pred_path}")

    # ------------------------------------------------------------------ #
    # Error analysis                                                       #
    # ------------------------------------------------------------------ #
    print(f"\n{'='*70}")
    print("ERROR ANALYSIS")
    print(f"{'='*70}")
    run_error_analysis_from_results(
        dataset=ds_name,
        test_keys=list(zip(test_clean['pert'], test_clean['gene'])),
        predictions=y_pred,
        true_labels=y_true,
        adata=None,
        task_name=task_name,
        class_names=['Non-DE', 'Up-regulated', 'Down-regulated'],
        output_dir=error_dir,
    )

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
