"""
knn_classifier_transfer_perturbQA.py
=====================================
Cross-cell-line KNN transfer for PerturbQA data.

Training: ALL (pert, gene) pairs from 3 source cell lines (both 'train' and 'test' splits),
          restricted to the common gene universe across all 4 PerturbQA datasets.
Test:     'test'-split pairs from the held-out cell line, restricted to the same gene universe.

When the same (perturbation, gene) pair appears in multiple source datasets, labels are
resolved by majority vote before building the training dictionary.
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
from majority_vote.analysis_utility import compute_3class_metrics, plot_data_summary
from majority_vote.error_utility import run_error_analysis_from_results
from analysis_utility import (
    PERTS_NO_ESM2_EMBEDDING,
    PERTS_NO_WIKICROW_EMBEDDING,
    PERTS_NO_COPATH_NEIGHBORS,
    load_embedding_from_pkl,
    load_copath_neighbors,
    build_embedding_matrix,
    find_k_nearest_neighbors,
)


def load_perturbqa_de_doc(de_path: str, dir_path: str,
                          splits: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load PerturbQA DE (binary 0/1) and direction (0=down/1=up) files and merge into
    3-class labels (0=non-DE, 1=up, 2=down). Optionally filter to requested splits.
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

    df = pd.DataFrame(rows)
    if splits is not None:
        df = df[df['split'].isin(splits)].reset_index(drop=True)

    print(f"  Loaded: {os.path.basename(de_path)}")
    print(f"  Pairs: {len(df)}  "
          f"Non-DE(0): {(df['label']==0).sum()}  "
          f"Up(1): {(df['label']==1).sum()}  "
          f"Down(2): {(df['label']==2).sum()}")
    return df


def merge_train_datasets(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenate source DataFrames and resolve (pert, gene) duplicates via majority vote.
    """
    combined = pd.concat(dfs, ignore_index=True)[['pert', 'gene', 'label']]
    n_before = len(combined)
    n_unique = combined[['pert', 'gene']].drop_duplicates().shape[0]
    print(f"\nMerging {len(dfs)} source dataset(s):")
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
    print(f"  Merged training set: {len(merged)} pairs  "
          f"({merged['pert'].nunique()} perts, {merged['gene'].nunique()} genes)")
    return merged


def predict_with_precomputed_neighbors(target_gene: str,
                                       neighbor_pert_names: List[str],
                                       neighbor_distances: np.ndarray,
                                       train_label_dict: Dict[Tuple[str, str], int],
                                       k: int,
                                       weights: str = 'distance') -> Tuple[int, np.ndarray, int]:
    """
    Scan neighbors in ascending-distance order, collecting up to k that have a label for
    target_gene. Falls back to class 0 with uniform probabilities if none found.

    Returns (predicted_label, class_probabilities, n_valid).
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
        nb_weights  = 1.0 - nb_dists
        class_votes = np.zeros(3)
        for lbl, w in zip(nb_labels, nb_weights):
            class_votes[lbl] += w

    class_proba = class_votes / class_votes.sum()
    max_vote    = class_votes.max()
    tied        = np.where(np.abs(class_votes - max_vote) < 1e-10)[0]
    if len(tied) > 1:
        prediction = int(np.random.choice(tied))
        print(f"    Vote tie: classes {tied} tied. Randomly choosing class {prediction}.")
    else:
        prediction = int(tied[0])

    return prediction, class_proba, len(valid_labels)


def predict_knn_batch(test_df: pd.DataFrame,
                      train_df: pd.DataFrame,
                      train_pert_names: List[str],
                      embedding_dict: Optional[Dict[str, Union[list, np.ndarray]]],
                      k: int,
                      weights: str = 'distance',
                      return_proba: bool = False,
                      verbose: bool = True,
                      copath_neighbors: Optional[Dict[str, list]] = None,
                      ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Batch KNN prediction for all (pert, gene) pairs in test_df.
    Returns (predictions, probabilities, uncertain_mask).
    uncertain_mask is True for pairs where no valid neighbor was found (class-0 fallback).
    """
    if embedding_dict is None and copath_neighbors is None:
        raise ValueError("Provide either embedding_dict or copath_neighbors.")
    if embedding_dict is not None and copath_neighbors is not None:
        raise ValueError("Provide only one of embedding_dict or copath_neighbors, not both.")

    train_label_dict = {(row['pert'], row['gene']): int(row['label'])
                        for _, row in train_df.iterrows()}

    unique_query_perts = test_df['pert'].unique().tolist()
    n_train            = len(train_pert_names)

    if copath_neighbors is not None:
        if verbose:
            print(f"  Using co-pathway neighbors for {len(unique_query_perts)} "
                  f"query perturbations (target k={k})...")
        train_pert_set = set(train_pert_names)
        pert_to_neighbors: Dict[str, Tuple[List[str], np.ndarray]] = {}
        no_copath: List[str] = []
        for p in unique_query_perts:
            if p not in copath_neighbors:
                no_copath.append(p)
                pert_to_neighbors[p] = ([], np.array([]))
            else:
                train_nb = [(n, s) for n, s in copath_neighbors[p] if n in train_pert_set]
                pert_to_neighbors[p] = (
                    [n for n, _ in train_nb],
                    np.array([1.0 - s for _, s in train_nb]),
                )
        if no_copath:
            warnings.warn(
                f"{len(no_copath)} query perturbations not found in co-path neighbors "
                f"and will default to class 0: {sorted(no_copath)}"
            )
    else:
        query_emb = build_embedding_matrix(unique_query_perts, embedding_dict)
        train_emb = build_embedding_matrix(train_pert_names,   embedding_dict)
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
            f"{n_zero_valid}/{n_total} (pert, gene) pairs had no valid neighbor; "
            f"defaulted to class 0 (non-DE)."
        )
    if n_expanded > 0:
        j_arr = np.array(expanded_j)
        warnings.warn(
            f"{n_expanded}/{n_total} pairs needed search expansion past k={k} "
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

    parser = argparse.ArgumentParser(
        description='KNN cross-cell-line transfer for PerturbQA data.'
    )
    parser.add_argument('--train_de_file', type=str, required=True, nargs='+',
                        help='DE CSV file(s) for source datasets (all splits used for training)')
    parser.add_argument('--train_dir_file', type=str, required=True, nargs='+',
                        help='Direction CSV file(s) for source datasets, same order as --train_de_file')
    parser.add_argument('--test_de_file', type=str, required=True,
                        help='DE CSV file for the held-out target dataset')
    parser.add_argument('--test_dir_file', type=str, required=True,
                        help='Direction CSV file for the held-out target dataset')
    parser.add_argument('--gene_universe_file', type=str, required=True,
                        help='CSV with column "gene_name" listing the common gene universe')
    parser.add_argument('--embedding_file', type=str, default=None,
                        help='Primary embedding pkl file (required for genept/esm2/wikicrow)')
    parser.add_argument('--embedding_file2', type=str, default=None,
                        help='Optional secondary embedding pkl to merge in')
    parser.add_argument('--embedding_mode', type=str, required=True,
                        choices=['genept', 'esm2', 'wikicrow', 'copath'],
                        help='Embedding type')
    parser.add_argument('--copath_file', type=str, default=None,
                        help='Path to gene_copath_neighbors.json (required for copath mode)')
    parser.add_argument('--coverage_filter', action='store_true',
                        help='Drop perts missing from any of the 4 embedding types')
    parser.add_argument('--train_dataset_name', type=str, required=True, nargs='+',
                        help='Source dataset name(s), e.g. K562 HepG2 Jurkat')
    parser.add_argument('--test_dataset_name', type=str, required=True,
                        help='Held-out target dataset name, e.g. RPE1')
    parser.add_argument('--k', type=int, default=10, help='Number of neighbors')
    parser.add_argument('--weights', type=str, default='distance',
                        choices=['distance', 'uniform'])
    parser.add_argument('--random_seed', type=int, default=305)
    parser.add_argument('--output_dir', type=str, default='results')

    args = parser.parse_args()

    if len(args.train_de_file) != len(args.train_dir_file):
        parser.error("--train_de_file and --train_dir_file must have the same length")
    if len(args.train_de_file) != len(args.train_dataset_name):
        parser.error("--train_de_file and --train_dataset_name must have the same length")

    use_copath = args.embedding_mode == 'copath'
    if use_copath and args.copath_file is None:
        parser.error("--copath_file is required when --embedding_mode is 'copath'")
    if not use_copath and args.embedding_file is None:
        parser.error("--embedding_file is required for genept/esm2/wikicrow modes")

    if args.embedding_mode == 'esm2':
        mode_drop_set = PERTS_NO_ESM2_EMBEDDING
        mode_label    = "ESM2 embedding"
    elif args.embedding_mode == 'wikicrow':
        mode_drop_set = PERTS_NO_WIKICROW_EMBEDDING
        mode_label    = "WikiCrow embedding"
    elif args.embedding_mode == 'copath':
        mode_drop_set = PERTS_NO_COPATH_NEIGHBORS
        mode_label    = "co-path neighbors"
    else:
        mode_drop_set = frozenset()
        mode_label    = "GenePT embedding"

    all_drop_set = (PERTS_NO_ESM2_EMBEDDING
                    | PERTS_NO_WIKICROW_EMBEDDING
                    | PERTS_NO_COPATH_NEIGHBORS)
    effective_drop_set = all_drop_set if args.coverage_filter else mode_drop_set

    np.random.seed(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)
    error_dir = os.path.join(os.path.dirname(args.output_dir) or ".", "error_analysis")
    os.makedirs(error_dir, exist_ok=True)

    train_name    = "+".join(args.train_dataset_name)
    mode_tag      = args.embedding_mode
    transfer_name = f"{args.test_dataset_name}_{mode_tag}_k{args.k}"
    if args.coverage_filter:
        transfer_name += "_covfilt"

    print("="*70)
    print("KNN Classifier — Cross-Cell-Line Transfer (PerturbQA)")
    print("="*70)
    print(f"\nEmbedding mode : {mode_label}")
    print(f"Train          : {train_name}")
    print(f"Test           : {args.test_dataset_name}")
    print(f"K              : {args.k}   Weights: {args.weights}   Seed: {args.random_seed}")
    print(f"Coverage filter: {args.coverage_filter}")
    print(f"Output dir     : {args.output_dir}")

    # Load common gene universe
    gene_universe = set(pd.read_csv(args.gene_universe_file)['gene_name'])
    print(f"\nCommon gene universe: {len(gene_universe)} genes")

    # Load all pairs from source cell lines (both 'train' and 'test' splits)
    print(f"\n{'='*70}")
    print("Loading source datasets (all splits)")
    print(f"{'='*70}")
    train_dfs = []
    for de_fp, dir_fp, name in zip(args.train_de_file, args.train_dir_file,
                                   args.train_dataset_name):
        print(f"\n  [{name}]")
        df = load_perturbqa_de_doc(de_fp, dir_fp, splits=None)
        train_dfs.append(df)

    # Load test split of held-out cell line
    print(f"\n{'='*70}")
    print(f"Loading target dataset: {args.test_dataset_name} (test split)")
    print(f"{'='*70}")
    test_df = load_perturbqa_de_doc(args.test_de_file, args.test_dir_file, splits=['test'])

    # Restrict to common gene universe
    print(f"\n[gene universe] Restricting all data to {len(gene_universe)} common genes")
    train_dfs = [df[df['gene'].isin(gene_universe)].reset_index(drop=True)
                 for df in train_dfs]
    test_df   = test_df[test_df['gene'].isin(gene_universe)].reset_index(drop=True)
    print(f"  After gene filter: "
          + ", ".join(f"{n}={df['pert'].nunique()}p/{df['gene'].nunique()}g"
                      for n, df in zip(args.train_dataset_name, train_dfs))
          + f", test={test_df['pert'].nunique()}p/{test_df['gene'].nunique()}g")

    # Drop perturbations without embedding/co-path coverage
    for split_name, df in ([("train", pd.concat(train_dfs, ignore_index=True)),
                             ("test",  test_df)]):
        dropped = set(df['pert'].unique()) & effective_drop_set
        if dropped:
            print(f"\nDropping {len(dropped)} {split_name} perturbations "
                  f"with no {mode_label}: {sorted(dropped)}")
    train_dfs = [df[~df['pert'].isin(effective_drop_set)].reset_index(drop=True)
                 for df in train_dfs]
    test_df   = test_df[~test_df['pert'].isin(effective_drop_set)].reset_index(drop=True)
    print(f"\nAfter pert filter: "
          + ", ".join(f"{n}={df['pert'].nunique()}p"
                      for n, df in zip(args.train_dataset_name, train_dfs))
          + f", test={test_df['pert'].nunique()}p")

    train_df = merge_train_datasets(train_dfs)

    train_perts = set(train_df['pert'].unique())
    train_genes = set(train_df['gene'].unique())
    test_perts  = set(test_df['pert'].unique())
    test_genes  = set(test_df['gene'].unique())

    print(f"\n{'='*70}")
    print("Cross-Dataset Overlap Statistics")
    print(f"{'='*70}")
    print(f"  Train perts: {len(train_perts)}   Test perts: {len(test_perts)}   "
          f"Shared: {len(train_perts & test_perts)}")
    print(f"  Train genes: {len(train_genes)}   Test genes: {len(test_genes)}")

    # Load embedding / co-path neighbors
    if use_copath:
        print(f"\nLoading co-pathway neighbors from {args.copath_file}...")
        copath_neighbors = load_copath_neighbors(args.copath_file)
        embedding_dict   = None
    else:
        copath_neighbors = None
        print(f"\nLoading {mode_label} embeddings from {args.embedding_file}...")
        primary_dict = load_embedding_from_pkl(args.embedding_file)
        if args.embedding_file2 is not None:
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
    print(f"Predicting on test set (K={args.k}, expandable search)")
    print(f"  {len(test_df)} pairs, {test_df['pert'].nunique()} perts")
    print(f"{'='*70}")

    y_pred, y_proba, uncertain_mask = predict_knn_batch(
        test_df=test_df,
        train_df=train_df,
        train_pert_names=train_pert_names,
        embedding_dict=embedding_dict,
        k=args.k,
        weights=args.weights,
        return_proba=True,
        verbose=True,
        copath_neighbors=copath_neighbors,
    )
    y_true = test_df['label'].values

    metrics = compute_3class_metrics(y_true, y_pred, y_proba)

    print(f"\n{'='*70}")
    print(f"RESULTS — {args.test_dataset_name} (transfer: {train_name} → {transfer_name})")
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

    # Save metrics CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df['k']             = args.k
    metrics_df['train_dataset'] = train_name
    metrics_df['test_dataset']  = args.test_dataset_name
    metrics_df['n_train_perts'] = len(train_perts)
    metrics_df['n_test_perts']  = len(test_perts)
    metrics_df['n_shared_perts'] = len(train_perts & test_perts)
    metrics_df['n_genes']       = len(gene_universe)
    metrics_path = os.path.join(
        args.output_dir, f'{transfer_name}_knn_transfer_metrics.csv'
    )
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nMetrics saved to {metrics_path}")

    # Save predictions CSV (includes zero_neighbor flag)
    pred_df = test_df.copy()
    pred_df['prediction']   = y_pred
    pred_df['correct']      = (y_pred == y_true)
    pred_df['prob_class_0'] = y_proba[:, 0]
    pred_df['prob_class_1'] = y_proba[:, 1]
    pred_df['prob_class_2'] = y_proba[:, 2]
    pred_df['zero_neighbor'] = uncertain_mask
    pred_path = os.path.join(
        args.output_dir, f'{transfer_name}_knn_transfer_predictions.csv'
    )
    pred_df.to_csv(pred_path, index=False)
    print(f"Predictions saved to {pred_path}")

    print(f"\n{'='*70}")
    print("ERROR ANALYSIS")
    print(f"{'='*70}")
    run_error_analysis_from_results(
        dataset=transfer_name,
        test_keys=list(zip(test_df['pert'], test_df['gene'])),
        predictions=y_pred,
        true_labels=y_true,
        adata=None,
        task_name="3class_KNN_transfer",
        class_names=['Non-DE', 'Up-regulated', 'Down-regulated'],
        output_dir=error_dir,
    )

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
