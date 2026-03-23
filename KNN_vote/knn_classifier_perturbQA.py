"""
knn_classifier_perturbQA.py
============================
Weighted KNN classifier for perturbation-gene pair prediction. Builds a 3-class KNN classifier:
- Class 0: Non-DE
- Class 1: Up-regulated
- Class 2: Down-regulated

Embedding-based (GenePT, ESM2):
  - Distance metric: cosine dissimilarity (1 - cos)
  - Neighbors: found at run time via sklearn NearestNeighbors
  - Voting weight ('distance'): cosine similarity (1 - cosine dissimilarity)

Co-pathway (co-path):
  - No embedding; neighbors are pre-computed from Jaccard similarity on
    shared MSigDB/KEGG/Reactome pathway memberships (see precompute_pathways.py)
  - Input: gene_copath_neighbors.json  {gene: [[neighbor, jaccard_sim], ...]}
  - Voting weight ('distance'): Jaccard similarity used directly as vote weight

Input CSV columns:
- DE file: pert, gene, label, split
- DOC file: pert, gene, label, split
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baseline.analysis_utility import compute_3class_metrics, plot_data_summary
from baseline.error_utility import run_error_analysis_from_results
from analysis_utility import (
    PERTS_NO_ESM2_EMBEDDING,
    PERTS_NO_COPATH_NEIGHBORS,
    load_embedding_from_pkl,
    load_copath_neighbors,
    build_embedding_matrix,
    find_k_nearest_neighbors,
    run_bootstrap,
)


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

    # DOC uses 1=up/0=down; our 3-class scheme uses 1=up/2=down
    doc_dict = {}
    for _, row in doc_df.iterrows():
        key = (row['pert'], row['gene'])
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
                # Shouldn't happen if DE/DOC files are consistent; default conservatively to non-DE
                warnings.warn(f"DE pair {key} not found in DOC file; defaulting to non-DE (label 0).")
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



def predict_with_precomputed_neighbors(target_gene: str,
                                       neighbor_pert_names: List[str],
                                       neighbor_distances: np.ndarray,
                                       train_label_dict: Dict[Tuple[str, str], int],
                                       k: int,
                                       weights: str = 'distance') -> Tuple[int, np.ndarray, int]:
    """
    Given a target gene and the full sorted neighbor list of a perturbation, predict
    label and class probabilities using weighted KNN. Keep going down the list of
    neighbors until we can find K neighbors with genes under query.  If the full neighbor
    list is exhausted before K are found, the available j < K valid neighbors are used.
    If no valid neighbors exist at all, falls back to class 0 (last resort).

    For 'distance' weighting, neighbor_distances must already encode the weighting:
      embedding path : distance = cosine_dissimilarity → weight = 1 - distance (cosine sim)
      co-path        : distance = 1 - jaccard_sim      → weight = 1 - distance (jaccard sim)

    Args:
        target_gene: Name of the target gene
        neighbor_pert_names: All neighbor perturbations sorted by ascending distance
        neighbor_distances: Corresponding distances, same order
        train_label_dict: Dict mapping (pert, gene) -> label from training data
        k: Desired number of valid neighbors to collect
        weights: 'distance' for similarity weighting, 'uniform' for equal weights

    Returns:
        Tuple of (predicted_label, class_probabilities, n_valid):
            - predicted_label: int (0, 1, or 2)
            - class_probabilities: np.ndarray of shape (3,) with probabilities for each class
            - n_valid: int, number of valid neighbors actually used (may be < k)
    """
    valid_labels = []
    valid_distances = []
    for pert_name, dist in zip(neighbor_pert_names, neighbor_distances):
        if len(valid_labels) == k:
            break
        key = (pert_name, target_gene)
        if key in train_label_dict:
            valid_labels.append(train_label_dict[key])
            valid_distances.append(dist)

    if len(valid_labels) == 0:
        # No neighbor in the entire training set has a label for this gene.
        # Fallback: predict class 0 (non-DE).
        return 0, np.array([1/3, 1/3, 1/3]), 0

    neighbor_labels = np.array(valid_labels, dtype=int)
    neighbor_dists = np.array(valid_distances)

    if weights == 'uniform':
        class_votes = np.bincount(neighbor_labels, minlength=3).astype(float)
    else:
        neighbor_weights = 1 - neighbor_dists
        class_votes = np.zeros(3)
        for label, weight in zip(neighbor_labels, neighbor_weights):
            class_votes[label] += weight

    class_probabilities = class_votes / class_votes.sum()

    max_vote = class_votes.max()
    tied_classes = np.where(np.abs(class_votes - max_vote) < 1e-10)[0]
    if len(tied_classes) > 1:
        prediction = np.random.choice(tied_classes)
        print(f"    Vote tie: classes {tied_classes} tied with vote={max_vote:.4f}. Randomly choosing class {prediction}.")
    else:
        prediction = tied_classes[0]

    return prediction, class_probabilities, len(valid_labels)


def predict_knn_batch(df: pd.DataFrame,
                      train_df: pd.DataFrame,
                      train_pert_names: List[str],
                      embedding_dict: Optional[Dict[str, Union[list, np.ndarray]]],
                      k: int,
                      weights: str = 'distance',
                      return_proba: bool = False,
                      verbose: bool = True,
                      copath_neighbors: Optional[Dict[str, list]] = None,
                      return_uncertain: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Predict labels for a batch of (perturbation, gene) pairs using KNN.

    Exactly one neighbor source must be provided:
      embedding_dict   -- uses cosine dissimilarity to find neighbors at run time
      copath_neighbors -- uses pre-computed Jaccard-similarity neighbors from
                          precompute_pathways.py; no embedding dict needed

    For 'distance' weighting:
      embedding path : weight = 1 - cosine_dissimilarity  (= cosine similarity)
      co-path        : weight = Jaccard similarity

    Args:
        df: DataFrame with pert-gene pairs to predict for
        train_df: Training DataFrame with columns [pert, gene, label]
        train_pert_names: List of unique training perturbation names
        embedding_dict: Dictionary mapping perturbation name to embedding vector; None for co-path
        k: Number of neighbors
        weights: 'distance' for similarity weighting, 'uniform' for equal weights
        return_proba: Whether to return class probabilities
        verbose: Whether to print progress
        copath_neighbors: Pre-loaded {gene: [[neighbor, jaccard_sim], ...]}; None for embedding

    Returns:
        predictions: int array of shape (n_samples,)
        probabilities: float array of shape (n_samples, 3) if return_proba=True, else None
        uncertain_mask: bool array of shape (n_samples,) marking pairs where no valid
          neighbor was found anywhere in the training set (class-0 fallback)
    """
    if embedding_dict is None and copath_neighbors is None:
        raise ValueError("Provide either embedding_dict or copath_neighbors.")
    if embedding_dict is not None and copath_neighbors is not None:
        raise ValueError("Provide only one of embedding_dict or copath_neighbors, not both.")

    train_label_dict = {(row['pert'], row['gene']): row['label'] for _, row in train_df.iterrows()}
    unique_query_perts = df['pert'].unique().tolist()

    # Build pert_to_neighbors: {pert: (neighbor_names, distances)}
    # For co-path, distance = 1 - jaccard_sim so weight = 1 - distance = jaccard_sim,
    # matching the cosine similarity weighting convention used by the embedding path.
    if copath_neighbors is not None:
        if verbose:
            print(f"  Using co-pathway neighbors for {len(unique_query_perts)} query perturbations...")
        train_pert_set = set(train_pert_names)
        pert_to_neighbors = {}
        zero_nb_perts = []
        n_under_k = 0
        for p in unique_query_perts:
            if p not in copath_neighbors:
                raise ValueError(f"Perturbation '{p}' not found in co-path neighbors.")
            # Keep ALL training neighbors sorted by descending Jaccard similarity
            # so predict_with_precomputed_neighbors can expand the search as needed.
            train_nb = [(n, s) for n, s in copath_neighbors[p] if n in train_pert_set]
            if len(train_nb) == 0:
                # Perturbation belongs to niche pathways with no overlap to the training set;
                # record it and fall back to class 0 (non-DE) at prediction time.
                zero_nb_perts.append(p)
                continue
            if len(train_nb) < k:
                n_under_k += 1
            neighbor_names = [n for n, _ in train_nb]
            distances = np.array([1.0 - s for _, s in train_nb])
            pert_to_neighbors[p] = (neighbor_names, distances)
        if zero_nb_perts:
            warnings.warn(
                f"{len(zero_nb_perts)}/{len(unique_query_perts)} query perturbations have 0 "
                f"co-path neighbors in the current training set and will be predicted as class 0 "
                f"(non-DE): {sorted(zero_nb_perts)}"
            )
        zero_nb_set = set(zero_nb_perts)
        if n_under_k > 0:
            warnings.warn(
                f"{n_under_k}/{len(unique_query_perts)} query perturbations have fewer than "
                f"k={k} co-path neighbors total in the current training set."
            )
    else:
        query_emb = build_embedding_matrix(unique_query_perts, embedding_dict)
        train_emb = build_embedding_matrix(train_pert_names, embedding_dict)
        # Find ALL training neighbors sorted by distance so predict_with_precomputed_neighbors
        # can expand the search beyond the initial K when a gene is missing from close neighbors.
        n_train = len(train_pert_names)
        if verbose:
            print(f"  Finding all {n_train} neighbors (sorted) for {len(unique_query_perts)} "
                  f"query perturbations (expandable search, target k={k})...")
        distances, indices = find_k_nearest_neighbors(query_emb, train_emb, n_train, check_ties=False)
        train_pert_arr = np.array(train_pert_names)
        pert_to_neighbors = {
            p: (train_pert_arr[indices[i]].tolist(), distances[i])
            for i, p in enumerate(unique_query_perts)
        }

    predictions = []
    probabilities = [] if return_proba else None
    uncertain = []   # True only for pairs with zero valid neighbors anywhere in the training set
    n_total = len(df)
    # zero_nb_set only populated in co-path mode; empty set for embedding path
    if copath_neighbors is None:
        zero_nb_set = set()
    # Track pairs that needed search expansion and the j values reached
    n_zero_valid = 0        # no valid neighbor found anywhere in training set
    n_expanded = 0          # needed to look past K but found at least one valid neighbor
    expanded_j_vals = []    # j values for expanded-but-short pairs (for log summary)

    for i, (_, row) in enumerate(df.iterrows()):
        if verbose and (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1}/{n_total} pairs...")

        pert = row['pert']
        gene = row['gene']

        if pert in zero_nb_set:
            predictions.append(0)
            uncertain.append(True)
            if return_proba:
                probabilities.append(np.array([1/3, 1/3, 1/3]))
            continue

        neighbor_names, neighbor_dist = pert_to_neighbors[pert]

        pred, proba, n_valid = predict_with_precomputed_neighbors(
            target_gene=gene,
            neighbor_pert_names=neighbor_names,
            neighbor_distances=neighbor_dist,
            train_label_dict=train_label_dict,
            k=k,
            weights=weights
        )
        if n_valid == 0:
            n_zero_valid += 1
            uncertain.append(True)
        else:
            uncertain.append(False)
            if n_valid < k:
                n_expanded += 1
                expanded_j_vals.append(n_valid)

        predictions.append(pred)
        if return_proba:
            probabilities.append(proba)

    if n_zero_valid > 0:
        warnings.warn(
            f"{n_zero_valid}/{n_total} (pert, gene) pairs had no valid neighbor anywhere in "
            f"the training set; predictions defaulted to class 0 (last resort)."
        )
    if n_expanded > 0:
        j_arr = np.array(expanded_j_vals)
        warnings.warn(
            f"{n_expanded}/{n_total} (pert, gene) pairs required search expansion beyond the "
            f"initial k={k} neighbors to find valid labels for the queried gene, but could not "
            f"reach k. Valid neighbors used — min: {j_arr.min()}, "
            f"median: {np.median(j_arr):.0f}, max: {j_arr.max()}."
        )

    predictions = np.array(predictions)
    uncertain_mask = np.array(uncertain, dtype=bool)
    if return_proba:
        probabilities = np.array(probabilities)

    if return_uncertain:
        return predictions, probabilities, uncertain_mask
    return predictions, probabilities


def cross_validate_k(train_df: pd.DataFrame,
                     embedding_dict: Optional[Dict[str, Union[list, np.ndarray]]],
                     k_values: List[int] = [1, 3, 5, 7, 9],
                     n_folds: int = 5,
                     weights: str = 'distance',
                     random_seed: int = 305,
                     copath_neighbors: Optional[Dict[str, list]] = None) -> Tuple[int, Dict[int, Dict[str, float]]]:
    """
    Perform K-fold cross-validation to select optimal K.

    Args:
        train_df: Training DataFrame with training pert-gene pairs
        embedding_dict: Dictionary mapping perturbation name to embedding vector; None for co-path
        k_values: List of K values to try
        n_folds: Number of folds
        weights: 'distance' for similarity weighting, 'uniform' for equal weights
        random_seed: Random seed for reproducibility
        copath_neighbors: Pre-loaded co-path neighbors; None for embedding path

    Returns:
        best_k: Optimal K value
        cv_results: Dict mapping K -> metrics dict
    """
    print(f"\n{'='*70}")
    print(f"Cross-Validation for K Selection")
    print(f"{'='*70}")
    print(f"K values to try: {k_values}")
    print(f"Number of folds: {n_folds}")

    train_perts = train_df['pert'].unique()
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    cv_results = {k: defaultdict(list) for k in k_values}

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_perts)):
        print(f"\nFold {fold + 1}/{n_folds}")

        fold_train_perts = train_perts[train_idx]
        fold_val_perts = train_perts[val_idx]
        fold_train_df = train_df[train_df['pert'].isin(fold_train_perts)].reset_index(drop=True)
        val_df = train_df[train_df['pert'].isin(fold_val_perts)]
        y_true = val_df['label'].values

        for k in k_values:
            print(f"  K={k}: predicting...", end='', flush=True)

            y_pred, _ = predict_knn_batch(
                df=val_df,
                train_df=fold_train_df,
                train_pert_names=fold_train_perts.tolist(),
                embedding_dict=embedding_dict,
                k=k,
                weights=weights,
                return_proba=False,
                verbose=False,
                copath_neighbors=copath_neighbors,
            )

            metrics = compute_3class_metrics(y_true, y_pred, y_proba=None)
            for metric_name, value in metrics.items():
                cv_results[k][metric_name].append(value)
            print(f" balanced_acc={metrics['balanced_accuracy']:.4f}")

    print(f"\n{'='*70}")
    print("Cross-Validation Results (Mean ± Std)")
    print(f"{'='*70}")

    avg_results = {}
    for k in k_values:
        avg_results[k] = {}
        print(f"\nK = {k}:")
        for metric_name in ['accuracy', 'balanced_accuracy', 'f1_macro']:
            values = cv_results[k][metric_name]
            mean = np.mean(values)
            std = np.std(values)
            avg_results[k][metric_name] = mean
            print(f"  {metric_name}: {mean:.4f} ± {std:.4f}")

    best_k = max(k_values, key=lambda k: avg_results[k]['balanced_accuracy'])
    print(f"\nBest K (by balanced accuracy): {best_k}")

    return best_k, avg_results


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Weighted KNN classifier for 3-class perturbation-gene prediction (PerturbQA data file)')
    parser.add_argument('--de_file', type=str, required=True, help='Path to PerturbQA DE CSV file (must have split column)')
    parser.add_argument('--doc_file', type=str, required=True, help='Path to PerturbQA DOC CSV file (must have split column)')
    parser.add_argument('--embedding_file', type=str, default=None, help='Path to primary GenePT embedding pkl file')
    parser.add_argument('--embedding_file2', type=str, default=None, help='(optional) Path to a second pkl to merge into the primary GenePT embedding')
    parser.add_argument('--esm2_file', type=str, default=None, help='Path to ESM2 embedding pkl file; mutually exclusive with --embedding_file and --copath_file')
    parser.add_argument('--copath_file', type=str, default=None, help='Path to gene_copath_neighbors.json produced by precompute_pathways.py; mutually exclusive with --embedding_file and --esm2_file')
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name (for output files)')
    parser.add_argument('--k_values', type=str, default='1,3,5,7,9', help='Comma-separated K values to try (default: 1,3,5,7,9)')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds for cross-validation (default: 5)')
    parser.add_argument('--skip_cv', action='store_true', help='Skip cross-validation and use specified K')
    parser.add_argument('--k', type=int, default=5, help='K value to use if skipping CV (default: 5)')
    parser.add_argument('--weights', type=str, default='distance', choices=['distance', 'uniform'], help="Weight function: 'distance' uses similarity as vote weight, 'uniform' gives equal weight (default: distance)")
    parser.add_argument('--random_seed', type=int, default=305, help='Random seed for reproducibility (default: 305)')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory for results (default: results)')
    parser.add_argument('--subsample', type=int, default=None, help='Subsample perturbations for a quick algorithm check (default: None = use all)')
    parser.add_argument('--n_bootstrap', type=int, default=0, help='Number of bootstrap trials for uncertainty estimation on sparse pairs (default: 0 = disabled)')

    args = parser.parse_args()

    # Validate mode: exactly one embedding source must be provided
    n_sources = sum([
        args.embedding_file is not None,
        args.esm2_file is not None,
        args.copath_file is not None,
    ])
    if n_sources == 0:
        parser.error("One of --embedding_file, --esm2_file, or --copath_file must be provided.")
    if n_sources > 1:
        parser.error("--embedding_file, --esm2_file, and --copath_file are mutually exclusive.")

    use_copath = args.copath_file is not None
    use_esm2 = args.esm2_file is not None

    print("="*70)
    print("Weighted KNN Classifier for Perturbation-Gene Prediction")
    print("="*70)
    print(f"\nDataset: {args.dataset_name}")
    if use_copath:
        print(f"Mode: co-pathway (Jaccard similarity)")
    elif use_esm2:
        print(f"Mode: ESM2 embedding (cosine similarity)")
    else:
        print(f"Mode: GenePT embedding (cosine similarity)")
    print(f"Weights: {args.weights}")
    print(f"Random seed: {args.random_seed}")
    print(f"Output results at: {args.output_dir}")
    print(f"Skip CV: {args.skip_cv}")

    os.makedirs(args.output_dir, exist_ok=True)

    merged_df = load_and_merge_de_doc(args.de_file, args.doc_file)
    train_df = merged_df[merged_df['split'] == 'train'].reset_index(drop=True)
    test_df = merged_df[merged_df['split'] == 'test'].reset_index(drop=True)

    # TESTING PURPOSE ONLY: subsample for a quick check of the algorithm
    if args.subsample is not None:
        train_pert = sorted(train_df['pert'].unique())
        test_pert = sorted(test_df['pert'].unique())
        train_df = train_df[train_df['pert'].isin(train_pert[:args.subsample])]
        test_df = test_df[test_df['pert'].isin(test_pert[:args.subsample])]

    # Drop perturbations that have no representation in the chosen embedding/neighbor source.
    # The drop sets are pre-computed from logs/check_coverage.txt (union across all datasets).
    drop_set = PERTS_NO_COPATH_NEIGHBORS if use_copath else (PERTS_NO_ESM2_EMBEDDING if use_esm2 else frozenset())
    mode_label = "co-path neighbors" if use_copath else ("ESM2 embedding" if use_esm2 else None)
    if drop_set:
        for split_name, df in [("train", train_df), ("test", test_df)]:
            dropped = set(df['pert'].unique()) & drop_set
            if dropped:
                print(f"\nDropping {len(dropped)} {split_name} perturbations with no {mode_label}: "
                      f"{sorted(dropped)}")
        train_df = train_df[~train_df['pert'].isin(drop_set)]
        test_df  = test_df[~test_df['pert'].isin(drop_set)]
        print(f"\nAfter filtering: train={train_df['pert'].nunique()} perts, "
              f"test={test_df['pert'].nunique()} perts")

    print(f"\nTraining data: {len(train_df)} pairs, {train_df['pert'].nunique()} unique perturbations, {train_df['gene'].nunique()} unique genes")
    print(f"Test data: {len(test_df)} pairs, {test_df['pert'].nunique()} unique perturbations, {test_df['gene'].nunique()} unique genes")

    if use_copath:
        print(f"\nLoading co-pathway neighbors from {args.copath_file}...")
        copath_neighbors = load_copath_neighbors(args.copath_file)
        embedding_dict = None
    elif use_esm2:
        print(f"\nLoading ESM2 embeddings from {args.esm2_file}...")
        embedding_dict = load_embedding_from_pkl(args.esm2_file)
        print(f"Loaded embeddings for {len(embedding_dict)} genes")
        copath_neighbors = None
    else:
        print(f"\nLoading GenePT embeddings from {args.embedding_file}...")
        primary_dict = load_embedding_from_pkl(args.embedding_file)
        if args.embedding_file2 is not None:
            secondary_dict = load_embedding_from_pkl(args.embedding_file2)
            embedding_dict = {**primary_dict, **secondary_dict}
            print(f"Loaded embeddings: {len(primary_dict)} (primary) + "
                  f"{len(secondary_dict)} (secondary) = {len(embedding_dict)} unique genes")
        else:
            embedding_dict = primary_dict
            print(f"Loaded embeddings for {len(embedding_dict)} genes")
        copath_neighbors = None

    plot_data_summary(train_df, test_df, args.dataset_name, output_dir=args.output_dir)

    if args.skip_cv:
        best_k = args.k
        print(f"\nSkipping cross-validation, using K = {best_k}")
        cv_results = None
    else:
        k_values = [int(k) for k in args.k_values.split(',')]
        best_k, cv_results = cross_validate_k(
            train_df=train_df,
            embedding_dict=embedding_dict,
            k_values=k_values,
            n_folds=args.n_folds,
            weights=args.weights,
            random_seed=args.random_seed,
            copath_neighbors=copath_neighbors,
        )
        cv_results_df = pd.DataFrame(cv_results).T
        cv_results_df.index.name = 'k'
        cv_results_path = os.path.join(args.output_dir, f'{args.dataset_name}_cv_results.csv')
        cv_results_df.to_csv(cv_results_path)
        print(f"\nCV results saved to {cv_results_path}")

    print(f"\n{'='*70}")
    print(f"Final Prediction on Test Set (K = {best_k})")
    print(f"{'='*70}")

    train_perts = train_df['pert'].unique().tolist()

    print(f"\nPredicting on {len(test_df)} test pairs (with probabilities for AUROC)...")
    y_pred, y_proba, uncertain_mask = predict_knn_batch(
        df=test_df,
        train_df=train_df,
        train_pert_names=train_perts,
        embedding_dict=embedding_dict,
        k=best_k,
        weights=args.weights,
        return_proba=True,
        verbose=True,
        copath_neighbors=copath_neighbors,
        return_uncertain=True,
    )
    y_true = test_df['label'].values

    metrics = compute_3class_metrics(y_true, y_pred, y_proba)

    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"\nBest K: {best_k}")
    print(f"\nOverall metrics:")
    print(f"  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro):    {metrics['recall_macro']:.4f}")
    print(f"  F1 (macro):        {metrics['f1_macro']:.4f}")
    if 'auroc_ovr_macro' in metrics:
        print(f"  AUROC OvR (macro): {metrics['auroc_ovr_macro']:.4f}")

    print(f"\nPer-class metrics:")
    for class_idx in range(3):
        class_name = ['Non-DE', 'Up-regulated', 'Down-regulated'][class_idx]
        print(f"  {class_name} (class {class_idx}):")
        print(f"    Precision: {metrics[f'precision_class_{class_idx}']:.4f}")
        print(f"    Recall:    {metrics[f'recall_class_{class_idx}']:.4f}")
        print(f"    F1:        {metrics[f'f1_class_{class_idx}']:.4f}")
        if f'auroc_ovr_class_{class_idx}' in metrics:
            print(f"    AUROC OvR:      {metrics[f'auroc_ovr_class_{class_idx}']:.4f}")

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
    predictions_df['prob_class_0'] = y_proba[:, 0]
    predictions_df['prob_class_1'] = y_proba[:, 1]
    predictions_df['prob_class_2'] = y_proba[:, 2]
    predictions_path = os.path.join(args.output_dir, f'{args.dataset_name}_knn_predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)
    print(f"\nPredictions saved to {predictions_path}")

    metrics_df = pd.DataFrame([metrics])
    metrics_df['best_k'] = best_k
    metrics_df['dataset'] = args.dataset_name
    metrics_path = os.path.join(args.output_dir, f'{args.dataset_name}_knn_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")

    # Bootstrap uncertainty estimation: re-sample uncertain pairs (those with no valid
    # neighbors) uniformly at random across N trials to measure how much the sparse
    # coverage inflates or deflates each metric.  Deterministic predictions are held fixed.
    # AUROC is excluded because it depends on probabilities (which are always [1/3,1/3,1/3]
    # for uncertain pairs and thus constant across trials).
    n_uncertain = uncertain_mask.sum()
    if args.n_bootstrap > 0 and n_uncertain > 0:
        run_bootstrap(y_true, y_pred, uncertain_mask, args.n_bootstrap,
                      args.random_seed, args.output_dir, args.dataset_name)
    elif args.n_bootstrap > 0 and n_uncertain == 0:
        print(f"\nNo uncertain pairs found; bootstrap skipped.")

    print(f"\n{'='*70}")
    print("ERROR ANALYSIS")
    print(f"{'='*70}")

    test_keys = list(zip(test_df['pert'], test_df['gene']))
    run_error_analysis_from_results(
        dataset=args.dataset_name,
        test_keys=test_keys,
        predictions=y_pred,
        true_labels=y_true,
        adata=None,
        task_name="3class_KNN",
        class_names=['Non-DE', 'Up-regulated', 'Down-regulated']
    )

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
