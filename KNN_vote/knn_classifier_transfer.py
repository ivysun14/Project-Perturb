"""
knn_classifier_transfer.py
===========================

Weighted KNN classifier (GenePT embeddings) for cross-cell-line transfer prediction.
Trains on the train splits of N source datasets and predicts on the test split of one
held-out target dataset.

When the same (perturbation, gene) pair appears in multiple source datasets, the labels
are resolved by majority vote *before* building the training label dictionary.  This
keeps the lookup compact and unambiguous.
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import classification_report

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baseline.analysis_utility import compute_3class_metrics, plot_data_summary
from baseline.error_utility import run_error_analysis_from_results
from analysis_utility import (
    load_embedding_from_pkl,
    build_embedding_matrix,
    find_k_nearest_neighbors,
)


def load_de_doc_file(file_path: str, split: str) -> pd.DataFrame:
    """
    Load a DE_DOC CSV and return only the requested split.
    """
    df = pd.read_csv(file_path)
    print(f"\nLoaded: {file_path}")
    print(f"  Total pairs: {len(df)}")
    print(f"  Non-DE(0): {(df['label']==0).sum()}  "
          f"Up(1): {(df['label']==1).sum()}  "
          f"Down(2): {(df['label']==2).sum()}")
    if 'split' not in df.columns:
        raise ValueError(f"CSV file does not have a 'split' column: {file_path}")
    split_df = df[df['split'] == split].copy()
    print(f"  '{split}' split: {len(split_df)} pairs "
          f"({split_df['pert'].nunique()} perts, {split_df['gene'].nunique()} genes)")
    return split_df


def merge_train_datasets(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenate training DataFrames and resolve (pert, gene) duplicates via majority vote.
    When the same perturbation-gene pair appears in more than one source dataset,
    the label with the most votes wins.

    Args:
        dfs: List of training DataFrames, each with columns [pert, gene, label].

    Returns:
        Merged DataFrame with unique (pert, gene) rows and majority-voted labels.
    """
    combined = pd.concat(dfs, ignore_index=True)[['pert', 'gene', 'label']]

    n_before = len(combined)
    n_unique = combined[['pert', 'gene']].drop_duplicates().shape[0]
    print(f"\nMerging {len(dfs)} training dataset(s):")
    print(f"  Total pairs (before merge): {n_before}")
    print(f"  Unique (pert, gene) pairs:  {n_unique}")
    print(f"  Overlapping pairs resolved via majority vote: {n_before - n_unique}")

    n_ties = 0

    def _majority(labels):
        nonlocal n_ties
        counts = np.bincount(labels.values.astype(int), minlength=3)
        max_count = counts.max()
        tied = np.where(counts == max_count)[0]
        if len(tied) > 1:
            n_ties += 1
            return int(np.random.choice(tied))
        return int(tied[0])

    merged = (combined
              .groupby(['pert', 'gene'], sort=False)['label']
              .agg(_majority)
              .reset_index())

    pct_ties = 100 * n_ties / len(merged) if len(merged) > 0 else 0.0
    print(f"  Tied label pairs (randomly resolved): {n_ties}/{len(merged)} ({pct_ties:.2f}%)")
    print(f"  Merged training set: {len(merged)} pairs "
          f"({merged['pert'].nunique()} perts, {merged['gene'].nunique()} genes)")
    return merged


def predict_with_precomputed_neighbors(target_gene: str,
                                       neighbor_pert_names: List[str],
                                       neighbor_distances: np.ndarray,
                                       train_label_dict: Dict[Tuple[str, str], int],
                                       k: int,
                                       weights: str = 'distance') -> Tuple[int, np.ndarray, int]:
    """
    Predict label and class probabilities for one (pert, gene) pair. Scans through all 
    neighbors in ascending distance order, collecting up to K neighbors who have a label 
    for target_gene in train_label_dict. Uses however many were found if the full list
    is exhausted before reaching K.  Falls back to class 0 with uniform probabilities
    if no valid neighbor exists anywhere in the training set.

    Args:
        target_gene         : Name of the gene to predict.
        neighbor_pert_names : All training perturbations sorted by ascending distance.
        neighbor_distances  : Corresponding cosine dissimilarities.
        train_label_dict    : {(pert, gene): label} from merged training data.
        k                   : Desired number of valid neighbors.
        weights             : 'distance' → cosine-similarity weight; 'uniform' → equal.

    Returns:
        (predicted_label, class_probabilities, n_valid)
        n_valid is the number of valid neighbors actually used (may be < k).
    """
    valid_labels: List[int]   = []
    valid_dists:  List[float] = []

    for pert_name, dist in zip(neighbor_pert_names, neighbor_distances):
        if len(valid_labels) == k:
            break
        key = (pert_name, target_gene)
        if key in train_label_dict:
            valid_labels.append(train_label_dict[key])
            valid_dists.append(dist)

    if len(valid_labels) == 0:
        return 0, np.array([1/3, 1/3, 1/3]), 0

    nb_labels = np.array(valid_labels, dtype=int)
    nb_dists  = np.array(valid_dists)

    if weights == 'uniform':
        class_votes = np.bincount(nb_labels, minlength=3).astype(float)
    else:
        # weight = cosine similarity = 1 − cosine dissimilarity
        nb_weights  = 1.0 - nb_dists
        class_votes = np.zeros(3)
        for lbl, w in zip(nb_labels, nb_weights):
            class_votes[lbl] += w

    class_proba = class_votes / class_votes.sum()
    max_vote    = class_votes.max()
    tied        = np.where(np.abs(class_votes - max_vote) < 1e-10)[0]
    if len(tied) > 1:
        prediction = int(np.random.choice(tied))
        print(f"    Vote tie: classes {tied} tied with vote={max_vote:.4f}. "
              f"Randomly choosing class {prediction}.")
    else:
        prediction = int(tied[0])

    return prediction, class_proba, len(valid_labels)


def predict_knn_batch(test_df: pd.DataFrame,
                      train_df: pd.DataFrame,
                      train_pert_names: List[str],
                      embedding_dict: Dict[str, Union[list, np.ndarray]],
                      k: int,
                      weights: str = 'distance',
                      return_proba: bool = False,
                      verbose: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Batch KNN prediction for all (pert, gene) pairs in test_df.

    Args:
        test_df          : Test pairs with columns [pert, gene, label].
        train_df         : Merged training DataFrame with columns [pert, gene, label].
        train_pert_names : Names of training perturbations (the neighbor pool).
        embedding_dict   : {pert_name: embedding_vector} for all perts.
        k                : Desired number of valid neighbors per prediction.
        weights          : 'distance' or 'uniform'.
        return_proba     : Whether to return per-class probabilities.
        verbose          : Print progress every 10 000 pairs.

    Returns:
        predictions   : (n_test,) int array.
        probabilities : (n_test, 3) float array if return_proba=True, else None.
        uncertain_mask: (n_test,) bool — True where no valid neighbor was found at all.
    """
    # Build label lookup from merged training data
    train_label_dict = {(row['pert'], row['gene']): int(row['label'])
                        for _, row in train_df.iterrows()}

    unique_query_perts = test_df['pert'].unique().tolist()
    n_train            = len(train_pert_names)
    query_emb = build_embedding_matrix(unique_query_perts, embedding_dict)
    train_emb = build_embedding_matrix(train_pert_names,   embedding_dict)

    # Retrieve ALL training neighbors sorted by cosine distance (expandable search)
    if verbose:
        print(f"  Retrieving all {n_train} neighbors sorted by cosine distance "
              f"for {len(unique_query_perts)} query perturbations (target k={k})...")
    distances, indices = find_k_nearest_neighbors(
        query_emb, train_emb, n_train, check_ties=False
    )
    train_pert_arr    = np.array(train_pert_names)
    pert_to_neighbors = {
        p: (train_pert_arr[indices[i]].tolist(), distances[i])
        for i, p in enumerate(unique_query_perts)
    }

    predictions   = []
    probabilities = [] if return_proba else None
    uncertain     = []
    n_total       = len(test_df)
    n_zero_valid  = 0
    n_expanded    = 0
    expanded_j    = []

    for i, (_, row) in enumerate(test_df.iterrows()):
        if verbose and (i + 1) % 10_000 == 0:
            print(f"  Processed {i+1}/{n_total} pairs...")

        pert = row['pert']
        gene = row['gene']
        neighbor_names, neighbor_dists = pert_to_neighbors[pert]

        pred, proba, n_valid = predict_with_precomputed_neighbors(
            target_gene=gene,
            neighbor_pert_names=neighbor_names,
            neighbor_distances=neighbor_dists,
            train_label_dict=train_label_dict,
            k=k,
            weights=weights
        )

        predictions.append(pred)
        if return_proba:
            probabilities.append(proba)

        if n_valid == 0:
            n_zero_valid += 1
            uncertain.append(True)
        else:
            uncertain.append(False)
            if n_valid < k:
                n_expanded += 1
                expanded_j.append(n_valid)

    if n_zero_valid > 0:
        warnings.warn(
            f"{n_zero_valid}/{n_total} (pert, gene) pairs had no valid neighbor anywhere "
            f"in the training set; defaulted to class 0 (non-DE)."
        )
    if n_expanded > 0:
        j_arr = np.array(expanded_j)
        warnings.warn(
            f"{n_expanded}/{n_total} (pert, gene) pairs needed search expansion past k={k} "
            f"but found < k valid neighbors. "
            f"Valid neighbors used — min: {j_arr.min()}, "
            f"median: {np.median(j_arr):.0f}, max: {j_arr.max()}."
        )

    predictions    = np.array(predictions)
    uncertain_mask = np.array(uncertain, dtype=bool)
    if return_proba:
        probabilities = np.array(probabilities)

    return predictions, probabilities, uncertain_mask


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='GenePT KNN transfer: train on N cell lines, test on 1.')
    parser.add_argument('--train_de_doc_file', type=str, required=True, nargs='+', help='DE_DOC CSV file(s) for source datasets (train split used)')
    parser.add_argument('--test_de_doc_file', type=str, required=True, help='DE_DOC CSV file for the target dataset (test split used)')
    parser.add_argument('--embedding_file', type=str, required=True, help='Path to primary GenePT embedding pkl file')
    parser.add_argument('--embedding_file2', type=str, default=None, help='(optional) Path to secondary GenePT embedding pkl to merge in')
    parser.add_argument('--train_dataset_name', type=str, required=True, nargs='+', help='Source dataset name(s), e.g. K562 HEPG2 Jurkat')
    parser.add_argument('--test_dataset_name', type=str, required=True, help='Target dataset name, e.g. RPE1')
    parser.add_argument('--k', type=int, default=10, help='Number of neighbors')
    parser.add_argument('--weights', type=str, default='distance', choices=['distance', 'uniform'], help='KNN vote weighting')
    parser.add_argument('--random_seed', type=int, default=305, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--subsample', type=int, default=None, help='Subsample N perts per dataset for quick testing')

    args = parser.parse_args()

    if len(args.train_de_doc_file) != len(args.train_dataset_name):
        raise ValueError("--train_de_doc_file and --train_dataset_name must have the same length")

    np.random.seed(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)

    train_name    = "+".join(args.train_dataset_name)
    transfer_name = args.test_dataset_name   # target cell line name used for all output files

    print("="*70)
    print("GenePT KNN Classifier — Cross-Cell-Line Transfer")
    print("="*70)
    print(f"\nTrain: {train_name}")
    print(f"Test: {args.test_dataset_name}")
    print(f"K: {args.k}   Weights: {args.weights}   Seed: {args.random_seed}")
    print(f"Output: {args.output_dir}")

    print(f"\n{'='*70}")
    print("Loading source datasets (train splits)")
    print(f"{'='*70}")
    train_dfs = []
    for fp, name in zip(args.train_de_doc_file, args.train_dataset_name):
        print(f"\n  [{name}]")
        train_dfs.append(load_de_doc_file(fp, split='train'))

    print(f"\n{'='*70}")
    print(f"Loading target dataset: {args.test_dataset_name} (test split)")
    print(f"{'='*70}")
    test_df = load_de_doc_file(args.test_de_doc_file, split='test')

    # TESTING PIRPOSE ONLY
    if args.subsample is not None:
        train_dfs = [df[df['pert'].isin(sorted(df['pert'].unique())[:args.subsample])]
                     for df in train_dfs]
        test_df = test_df[
            test_df['pert'].isin(sorted(test_df['pert'].unique())[:args.subsample])
        ]
        print(f"\nSubsampled to {args.subsample} perts per dataset.")

    train_df = merge_train_datasets(train_dfs)
    train_perts = set(train_df['pert'].unique())
    test_perts  = set(test_df['pert'].unique())
    train_genes = set(train_df['gene'].unique())
    test_genes  = set(test_df['gene'].unique())

    print(f"\n{'='*70}")
    print("Cross-Dataset Overlap Statistics")
    print(f"{'='*70}")
    print(f"  Train perts: {len(train_perts)}   Test perts: {len(test_perts)}   "
          f"Shared: {len(train_perts & test_perts)}")
    print(f"  Train genes: {len(train_genes)}   Test genes: {len(test_genes)}   "
          f"Shared: {len(train_genes & test_genes)}")
    print(f"  Test-only genes (no training label available): "
          f"{len(test_genes - train_genes)}")

    print(f"\nLoading GenePT embeddings from {args.embedding_file}...")
    primary_dict = load_embedding_from_pkl(args.embedding_file)
    if args.embedding_file2 is not None:
        print(f"Loading secondary GenePT embeddings from {args.embedding_file2}...")
        secondary_dict = load_embedding_from_pkl(args.embedding_file2)
        embedding_dict = {**primary_dict, **secondary_dict}
        print(f"  {len(primary_dict)} (primary) + {len(secondary_dict)} (secondary) "
              f"= {len(embedding_dict)} unique entries")
    else:
        embedding_dict = primary_dict
        print(f"  Loaded {len(embedding_dict)} entries")

    plot_data_summary(train_df, test_df, transfer_name, output_dir=args.output_dir)

    train_pert_names = sorted(train_df['pert'].unique().tolist())

    print(f"\n{'='*70}")
    print(f"Predicting (K = {args.k}, expandable search)")
    print(f"  Train: {len(train_perts)} perts, {len(train_df)} merged pairs")
    print(f"  Test:  {len(test_perts)} perts, {len(test_df)} pairs")
    print(f"{'='*70}")

    y_pred, y_proba, uncertain_mask = predict_knn_batch(
        test_df=test_df,
        train_df=train_df,
        train_pert_names=train_pert_names,
        embedding_dict=embedding_dict,
        k=args.k,
        weights=args.weights,
        return_proba=True,
        verbose=True
    )
    y_true = test_df['label'].values
    metrics = compute_3class_metrics(y_true, y_pred, y_proba)

    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"\nTransfer: {train_name} → {args.test_dataset_name}   K={args.k}")
    print(f"\nOverall metrics:")
    print(f"  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro):    {metrics['recall_macro']:.4f}")
    print(f"  F1 (macro):        {metrics['f1_macro']:.4f}")
    if 'auroc_ovr_macro' in metrics:
        print(f"  AUROC OvR (macro): {metrics['auroc_ovr_macro']:.4f}")

    print(f"\nPer-class metrics:")
    for cls in range(3):
        name = ['Non-DE', 'Up-regulated', 'Down-regulated'][cls]
        print(f"  {name} (class {cls}):")
        print(f"    Precision: {metrics[f'precision_class_{cls}']:.4f}")
        print(f"    Recall:    {metrics[f'recall_class_{cls}']:.4f}")
        print(f"    F1:        {metrics[f'f1_class_{cls}']:.4f}")
        if f'auroc_ovr_class_{cls}' in metrics:
            print(f"    AUROC OvR: {metrics[f'auroc_ovr_class_{cls}']:.4f}")

    print(f"\n{'='*70}")
    print("Detailed Classification Report")
    print(f"{'='*70}")
    print(classification_report(
        y_true, y_pred,
        target_names=['Non-DE', 'Up-regulated', 'Down-regulated'],
        zero_division=0
    ))

    predictions_df = test_df.copy()
    predictions_df['prediction'] = y_pred
    predictions_df['correct']    = (y_pred == y_true)
    if y_proba is not None:
        predictions_df['prob_class_0'] = y_proba[:, 0]
        predictions_df['prob_class_1'] = y_proba[:, 1]
        predictions_df['prob_class_2'] = y_proba[:, 2]
    pred_path = os.path.join(
        args.output_dir, f'{transfer_name}_knn_transfer_predictions.csv'
    )
    predictions_df.to_csv(pred_path, index=False)
    print(f"\nPredictions saved to {pred_path}")

    metrics_df = pd.DataFrame([metrics])
    metrics_df['k']              = args.k
    metrics_df['train_dataset']  = train_name
    metrics_df['test_dataset']   = args.test_dataset_name
    metrics_df['n_train_perts']  = len(train_perts)
    metrics_df['n_test_perts']   = len(test_perts)
    metrics_df['n_shared_perts'] = len(train_perts & test_perts)
    metrics_df['n_shared_genes'] = len(train_genes & test_genes)
    metrics_path = os.path.join(
        args.output_dir, f'{transfer_name}_knn_transfer_metrics.csv'
    )
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")

    print(f"\n{'='*70}")
    print("ERROR ANALYSIS")
    print(f"{'='*70}")
    test_keys = list(zip(test_df['pert'], test_df['gene']))
    run_error_analysis_from_results(
        dataset=transfer_name,
        test_keys=test_keys,
        predictions=y_pred,
        true_labels=y_true,
        adata=None,
        task_name="3class_KNN_transfer",
        class_names=['Non-DE', 'Up-regulated', 'Down-regulated']
    )

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
