"""
knn_classifier.py
=================
Weighted KNN classifier for perturbation-gene pair prediction. Builds a 3-class KNN classifier:
- Class 0: Non-DE
- Class 1: Up-regulated
- Class 2: Down-regulated

Embedding-based (GenePT, ESM2, WikiCrow):
  - Distance metric: cosine dissimilarity (1 - cos)
  - Neighbors: found at run time via sklearn NearestNeighbors
  - Voting weight ('distance'): cosine similarity (1 - cosine dissimilarity)

Co-pathway (co-path):
  - No embedding; neighbors are pre-computed from Jaccard similarity on
    shared MSigDB/KEGG/Reactome pathway memberships (see precompute_pathways.py)
  - Input: gene_copath_neighbors.json  {gene: [[neighbor, jaccard_sim], ...]}
  - Voting weight ('distance'): Jaccard similarity used directly as vote weight

For both modes, 'uniform' weighting gives every neighbor an equal vote.

Input CSV format (from DE_DOC_stats.py):
- pert: perturbation name
- gene: gene name
- label: 0 (non-DE), 1 (up-regulation), 2 (down-regulation)
- split: 'train' or 'test'
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import KFold
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
    run_sensitivity_analysis,
)


def load_de_doc_file(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load DE_DOC CSV file and return the 60/20/20 train/calib/test splits
    using the 'split_calib' column.

    Args:
        file_path: Path to DE_DOC CSV file with 'split_calib' column

    Returns:
        Tuple of (train_df, calib_df, test_df)
    """
    df = pd.read_csv(file_path)

    print(f"\nLoaded DE+DOC file: {len(df)} perturbation-gene pairs")
    print(f"3-Class label distribution:")
    print(f"  Non-DE (0): {(df['label'] == 0).sum()}")
    print(f"  Up-regulated (1): {(df['label'] == 1).sum()}")
    print(f"  Down-regulated (2): {(df['label'] == 2).sum()}")

    if 'split_calib' not in df.columns:
        raise ValueError("CSV file does not have 'split_calib' column!")

    train_df = df[df['split_calib'] == 'train'].copy()
    calib_df = df[df['split_calib'] == 'calibration'].copy()
    test_df  = df[df['split_calib'] == 'test'].copy()

    print(f"Train set:       {len(train_df)} pairs ({train_df['pert'].nunique()} perturbations)")
    print(f"Calibration set: {len(calib_df)} pairs ({calib_df['pert'].nunique()} perturbations)")
    print(f"Test set:        {len(test_df)} pairs ({test_df['pert'].nunique()} perturbations)")

    return train_df, calib_df, test_df


def build_log2fc_and_label_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], Dict[str, int]]:
    """
    Build 1). a perturbation × gene matrix of log2FC values
          2). a perturbation × gene matrix of labels
          3). perturbation to index mapping
          4). gene to index mapping

    Args:
        df: DataFrame with columns [pert, gene, log2FC, label]

    Returns:
        log2fc_matrix: numpy array of shape (n_perturbations, n_genes)
        label_matrix: numpy array of shape (n_perturbations, n_genes)
        pert_to_idx: dict mapping perturbation name to row index
        gene_to_idx: dict mapping gene name to column index
    """
    perturbations = sorted(df['pert'].unique())
    genes = sorted(df['gene'].unique())
    pert_to_idx = {p: i for i, p in enumerate(perturbations)}
    gene_to_idx = {g: i for i, g in enumerate(genes)}

    n_perts = len(perturbations)
    n_genes = len(genes)
    log2fc_matrix = np.full((n_perts, n_genes), np.nan, dtype=np.float64)
    label_matrix = np.full((n_perts, n_genes), np.nan, dtype=np.float64)

    for _, row in df.iterrows():
        pert_idx = pert_to_idx[row['pert']]
        gene_idx = gene_to_idx[row['gene']]
        log2fc_matrix[pert_idx, gene_idx] = row['log2FC']
        label_matrix[pert_idx, gene_idx] = row['label']

    n_missing_log2fc = np.isnan(log2fc_matrix).sum()
    n_missing_label = np.isnan(label_matrix).sum()
    if n_missing_log2fc > 0:
        warnings.warn(
            f"{n_missing_log2fc} missing values in log2FC matrix "
            f"({100*n_missing_log2fc/(n_perts*n_genes):.2f}%)"
        )
    if n_missing_label > 0:
        warnings.warn(
            f"{n_missing_label} missing values in label matrix "
            f"({100*n_missing_label/(n_perts*n_genes):.2f}%)"
        )

    return log2fc_matrix, label_matrix, pert_to_idx, gene_to_idx



def predict_with_precomputed_neighbors(target_gene_idx: int,
                                       neighbor_indices: np.ndarray,
                                       neighbor_distances: np.ndarray,
                                       label_matrix: np.ndarray,
                                       train_pert_indices: np.ndarray,
                                       weights: str = 'distance') -> Tuple[int, np.ndarray]:
    """
    Given a target gene and pre-computed neighbors of a specific perturbation, predict
    label and class probabilities using weighted KNN.

    Args:
        target_gene_idx: Index of the target gene
        neighbor_indices: Indices of K nearest neighbors (in train_pert_indices)
        neighbor_distances: Distances to K nearest neighbors
        label_matrix: Perturbation × gene matrix of labels
        train_pert_indices: Indices of training perturbations
        weights: 'distance' for cosine similarity weighting, 'uniform' for equal weights

    Returns:
        Tuple of (predicted_label, class_probabilities):
            - predicted_label: int (0, 1, or 2)
            - class_probabilities: np.ndarray of shape (3,) with probabilities for each class
    """
    neighbor_labels = label_matrix[train_pert_indices[neighbor_indices], target_gene_idx]
    neighbor_labels = neighbor_labels.astype(int)

    if weights == 'uniform':
        class_votes = np.bincount(neighbor_labels, minlength=3).astype(float)
    else:
        neighbor_weights = 1 - neighbor_distances
        class_votes = np.zeros(3)
        for label, weight in zip(neighbor_labels, neighbor_weights):
            class_votes[label] += weight

    class_probabilities = class_votes / class_votes.sum()
    max_vote = class_votes.max()
    # flag as caveats if voting tie occurs
    tied_classes = np.where(np.abs(class_votes - max_vote) < 1e-10)[0]
    if len(tied_classes) > 1:
        prediction = np.random.choice(tied_classes)
        print(f"    Vote tie: classes {tied_classes} tied with vote={max_vote:.4f}. Randomly choosing class {prediction}.")
    else:
        prediction = tied_classes[0]

    return prediction, class_probabilities


def predict_knn_batch(df: pd.DataFrame,
                      label_matrix: np.ndarray,
                      pert_to_idx: Dict[str, int],
                      gene_to_idx: Dict[str, int],
                      train_pert_names: List[str],
                      embedding_matrix: Optional[np.ndarray],
                      k: int,
                      weights: str = 'distance',
                      return_proba: bool = False,
                      verbose: bool = True,
                      copath_neighbors: Optional[Dict[str, list]] = None,
                      return_uncertain: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Predict labels for a batch of (perturbation, gene) pairs using KNN.

    Exactly one neighbor source must be provided:
      embedding_matrix   -- uses cosine dissimilarity to find neighbors at run time
      copath_neighbors   -- uses pre-computed Jaccard-similarity neighbors from
                            precompute_pathways.py; no embedding matrix needed

    For 'distance' weighting:
      embedding path : weight = 1 - cosine_dissimilarity  (= cosine similarity)
      co-path        : weight = Jaccard similarity

    Args:
        df: DataFrame with pert-gene pairs to predict for
        label_matrix: Perturbation × gene matrix of labels
        pert_to_idx: Dict mapping perturbation name to row index in matrices
        gene_to_idx: Dict mapping gene name to column index in matrices
        train_pert_names: List of training perturbation names
        embedding_matrix: Embedding matrix (n_perts, embed_dim); None for co-path
        k: Number of neighbors
        weights: 'distance' for similarity weighting, 'uniform' for equal weights
        return_proba: Whether to return class probabilities
        verbose: Whether to print progress
        copath_neighbors: Pre-loaded {gene: [[neighbor, jaccard_sim], ...]}; None for embedding

    Returns:
        If return_proba=False and return_uncertain=False: (predictions, None)
        If return_proba=True:  (predictions, probabilities) shape (n_samples, 3)
        If return_uncertain=True: adds uncertain_mask as third return value
    """
    if embedding_matrix is None and copath_neighbors is None:
        raise ValueError("Provide either embedding_matrix or copath_neighbors")
    if embedding_matrix is not None and copath_neighbors is not None:
        raise ValueError("Provide only one of embedding_matrix or copath_neighbors, not both")

    train_pert_indices = np.array([pert_to_idx[p] for p in train_pert_names])
    unique_query_perts = df['pert'].unique()

    if copath_neighbors is not None:
        # Co-path: neighbors come from precomputed Jaccard similarity
        if verbose:
            print(f"  Using co-pathway neighbors for {len(unique_query_perts)} query perturbations...")
        train_name_to_local = {n: i for i, n in enumerate(train_pert_names)}
        train_pert_set = set(train_pert_names)
        pert_to_neighbors = {}
        zero_nb_perts = []
        n_under_k = 0
        for p in unique_query_perts:
            if p not in copath_neighbors:
                raise ValueError(f"Perturbation '{p}' not found in co-path neighbors.")
            # Filter to training perturbations (already sorted desc by Jaccard sim)
            train_nb = [(n, s) for n, s in copath_neighbors[p] if n in train_pert_set]
            if len(train_nb) == 0:
                # Gene belongs to niche pathways with no overlap to the current training set;
                # record it and fall back to class 0 (non-DE) at prediction time.
                zero_nb_perts.append(p)
                continue
            if len(train_nb) < k:
                n_under_k += 1
            train_nb = train_nb[:k]
            local_idx = np.array([train_name_to_local[n] for n, _ in train_nb])
            # store as distance = 1 - jaccard_sim so predict_with_precomputed_neighbors
            # computes weight = 1 - distance = jaccard_sim
            distances = np.array([1.0 - s for _, s in train_nb])
            pert_to_neighbors[p] = (local_idx, distances)
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
                f"k={k} co-path neighbors in the current training set."
            )
    else:
        # Embedding path: find neighbors via cosine dissimilarity
        query_pert_indices = np.array([pert_to_idx[p] for p in unique_query_perts])
        query_emb = embedding_matrix[query_pert_indices]
        train_emb = embedding_matrix[train_pert_indices]
        if verbose:
            print(f"  Finding {k} nearest neighbors for {len(unique_query_perts)} query perturbations...")
        distances, indices = find_k_nearest_neighbors(query_emb, train_emb, k)
        pert_to_neighbors = {p: (indices[i], distances[i]) for i, p in enumerate(unique_query_perts)}

    predictions = []
    probabilities = [] if return_proba else None
    uncertain = [] if return_uncertain else None
    n_total = len(df)
    # zero_nb_set only populated in co-path mode; empty set for embedding path
    if copath_neighbors is None:
        zero_nb_set = set()
    for i, (_, row) in enumerate(df.iterrows()):
        if verbose and (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1}/{n_total} pairs...")

        pert = row['pert']
        gene = row['gene']
        gene_idx = gene_to_idx[gene]

        if pert in zero_nb_set:
            predictions.append(0)
            if return_proba:
                probabilities.append(np.array([1/3, 1/3, 1/3]))
            if return_uncertain:
                uncertain.append(True)
            continue

        neighbor_idx, neighbor_dist = pert_to_neighbors[pert]
        pred, proba = predict_with_precomputed_neighbors(
            target_gene_idx=gene_idx,
            neighbor_indices=neighbor_idx,
            neighbor_distances=neighbor_dist,
            label_matrix=label_matrix,
            train_pert_indices=train_pert_indices,
            weights=weights
        )
        predictions.append(pred)
        if return_proba:
            probabilities.append(proba)
        if return_uncertain:
            uncertain.append(False)

    predictions = np.array(predictions)
    if return_proba:
        probabilities = np.array(probabilities)
    if return_uncertain:
        uncertain_mask = np.array(uncertain, dtype=bool)
        return predictions, probabilities, uncertain_mask

    return predictions, probabilities


def cross_validate_k(train_df: pd.DataFrame,
                     label_matrix: np.ndarray,
                     embedding_matrix: Optional[np.ndarray],
                     pert_to_idx: Dict[str, int],
                     gene_to_idx: Dict[str, int],
                     k_values: List[int] = [1, 3, 5, 7, 9],
                     n_folds: int = 5,
                     weights: str = 'distance',
                     random_seed: int = 305,
                     copath_neighbors: Optional[Dict[str, list]] = None) -> Tuple[int, Dict[int, Dict[str, float]]]:
    """
    Perform K-fold cross-validation to select optimal K.

    Args:
        train_df: Training DataFrame with training pert-gene pairs
        label_matrix: Perturbation × gene matrix of labels
        embedding_matrix: Perturbation embedding matrix; None when using co-path
        pert_to_idx: Dict mapping perturbation name to row index in matrices
        gene_to_idx: Dict mapping gene name to column index in matrices
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
        val_df = train_df[train_df['pert'].isin(fold_val_perts)]
        y_true = val_df['label'].values

        for k in k_values:
            print(f"  K={k}: predicting...", end='', flush=True)

            y_pred, _ = predict_knn_batch(
                df=val_df,
                label_matrix=label_matrix,
                pert_to_idx=pert_to_idx,
                gene_to_idx=gene_to_idx,
                train_pert_names=fold_train_perts.tolist(),
                embedding_matrix=embedding_matrix,
                k=k,
                weights=weights,
                return_proba=False,
                verbose=False,
                copath_neighbors=copath_neighbors,
            )

            # Compute metrics (without AUROC during CV)
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

    # best K is selected based on balanced accuracy
    best_k = max(k_values, key=lambda k: avg_results[k]['balanced_accuracy'])
    print(f"\nBest K (by balanced accuracy): {best_k}")

    return best_k, avg_results


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Weighted KNN classifier for 3-class perturbation-gene prediction')
    parser.add_argument('--de_doc_file', type=str, required=True, help='Path to DE_DOC CSV file (must have split column)')
    parser.add_argument('--embedding_file', type=str, default=None, help='Path to primary embedding pkl file (required unless --copath_file is provided)')
    parser.add_argument('--embedding_file2', type=str, default=None, help='(optional) Path to a second pkl to merge into the primary embedding')
    parser.add_argument('--embedding_mode', type=str, default=None, choices=['genept', 'esm2', 'wikicrow', 'copath'],
                        help="Embedding type: 'genept' (no drop set), 'esm2', 'wikicrow', or 'copath'. Required when --embedding_file is provided.")
    parser.add_argument('--copath_file', type=str, default=None, help='Path to gene_copath_neighbors.json produced by precompute_pathways.py; mutually exclusive with --embedding_file')
    parser.add_argument('--coverage_filter', action='store_true',
                        help='Restrict all splits to perturbations covered by ALL 4 embedding types '
                             '(intersection of genept/esm2/wikicrow/copath). Use for sensitivity analysis '
                             'to ensure apples-to-apples comparison across embedding modes.')
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name (for output files)')
    parser.add_argument('--k_values', type=str, default='1,3,5,7,9', help='Comma-separated K values to try (default: 1,3,5,7,9)')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds for cross-validation (default: 5)')
    parser.add_argument('--skip_cv', action='store_true', help='Skip cross-validation and use specified K')
    parser.add_argument('--k', type=int, default=5, help='K value to use if skipping CV (default: 5)')
    parser.add_argument('--weights', type=str, default='distance', choices=['distance', 'uniform'], help="Weight function: 'distance' uses cosine similarity as vote weight, 'uniform' gives equal weight to all neighbors (default: distance)")
    parser.add_argument('--random_seed', type=int, default=305, help='Random seed for reproducibility (default: 305)')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory for results (default: results)')
    parser.add_argument('--subsample', type=int, default=None, help='Subsample perturbations per train and test data for quick check of the algorithm (default: None = use all)')
    parser.add_argument('--sensitivity_analysis', action='store_true', help='Run sensitivity analysis for co-path zero-neighbor uncertain pairs (sets uncertain pairs to class 1 and class 2 respectively)')

    args = parser.parse_args()

    # Validate: exactly one of --embedding_file or --copath_file must be provided
    if args.copath_file is not None and args.embedding_file is not None:
        parser.error("--copath_file and --embedding_file are mutually exclusive")
    if args.copath_file is None and args.embedding_file is None:
        parser.error("One of --embedding_file or --copath_file must be provided")
    if args.embedding_file is not None and args.embedding_mode is None:
        parser.error("--embedding_mode (genept or esm2) is required when --embedding_file is provided")

    use_copath = args.copath_file is not None

    print("="*70)
    print("Weighted KNN Classifier for Perturbation-Gene Prediction")
    print("="*70)
    print(f"\nDataset: {args.dataset_name}")
    print(f"Mode: {'co-pathway (Jaccard similarity)' if use_copath else 'embedding (cosine similarity)'}")
    print(f"Weights: {args.weights}")
    print(f"Random seed: {args.random_seed}")
    print(f"Output results at: {args.output_dir}")
    print(f"Skip CV: {args.skip_cv}")

    os.makedirs(args.output_dir, exist_ok=True)
    error_dir = os.path.join(os.path.dirname(args.output_dir) or ".", "error_analysis")
    os.makedirs(error_dir, exist_ok=True)
    train_df, calib_df, test_df = load_de_doc_file(args.de_doc_file)

    # TESTING PURPOSE ONLY: subsample a small amount of train and test perturbations for quick check
    if args.subsample is not None:
        train_pert = sorted(train_df['pert'].unique())
        calib_pert = sorted(calib_df['pert'].unique())
        test_pert  = sorted(test_df['pert'].unique())
        train_df = train_df[train_df['pert'].isin(train_pert[:args.subsample])]
        calib_df = calib_df[calib_df['pert'].isin(calib_pert[:args.subsample])]
        test_df  = test_df[test_df['pert'].isin(test_pert[:args.subsample])]

    # Drop perturbations with no embedding / no co-path neighbors for the chosen mode.
    # Co-path, ESM2, and WikiCrow use pre-computed static sets; GenePT has full coverage.
    if use_copath:
        mode_drop_set = PERTS_NO_COPATH_NEIGHBORS
        mode_label    = "co-path neighbors"
    elif args.embedding_mode == 'esm2':
        mode_drop_set = PERTS_NO_ESM2_EMBEDDING
        mode_label    = "ESM2 embedding"
    elif args.embedding_mode == 'wikicrow':
        mode_drop_set = PERTS_NO_WIKICROW_EMBEDDING
        mode_label    = "WikiCrow embedding"
    else:  # genept
        mode_drop_set = frozenset()
        mode_label    = "GenePT embedding"

    # With --coverage_filter also drop perts missing from ANY type so all 4
    # modes are evaluated on the same perturbation set (apples-to-apples).
    all_drop_set = (PERTS_NO_ESM2_EMBEDDING
                    | PERTS_NO_WIKICROW_EMBEDDING
                    | PERTS_NO_COPATH_NEIGHBORS)
    effective_drop_set = all_drop_set if args.coverage_filter else mode_drop_set

    for split_name, df in [("train", train_df), ("calib", calib_df), ("test", test_df)]:
        dropped = set(df['pert'].unique()) & effective_drop_set
        if dropped:
            print(f"\nDropping {len(dropped)} {split_name} perturbations with no {mode_label}: "
                  f"{sorted(dropped)}")
    train_df = train_df[~train_df['pert'].isin(effective_drop_set)]
    calib_df = calib_df[~calib_df['pert'].isin(effective_drop_set)]
    test_df  = test_df[~test_df['pert'].isin(effective_drop_set)]
    print(f"\nAfter filtering: train={train_df['pert'].nunique()} perts, "
          f"calib={calib_df['pert'].nunique()} perts, "
          f"test={test_df['pert'].nunique()} perts")

    print("\nBuilding log2fc and label matrix from train data...")
    _, label_matrix, pert_to_idx, gene_to_idx = build_log2fc_and_label_matrix(train_df)
    print(f"    Unique perturbations: {len(pert_to_idx)}")
    print(f"    Unique genes: {len(gene_to_idx)}")
    print(f"    Label matrix shape: {label_matrix.shape} (perturbations × genes)")

    if use_copath:
        print(f"\nLoading co-pathway neighbors from {args.copath_file}...")
        copath_neighbors = load_copath_neighbors(args.copath_file)
        embedding_matrix = None
        embedding_dict = None
    else:
        copath_neighbors = None
        print(f"\nLoading embeddings from {args.embedding_file}...")
        primary_dict = load_embedding_from_pkl(args.embedding_file)
        if args.embedding_file2 is not None:
            secondary_dict = load_embedding_from_pkl(args.embedding_file2)
            embedding_dict = {**primary_dict, **secondary_dict}
            print(f"Loaded embeddings: {len(primary_dict)} (primary) + "
                  f"{len(secondary_dict)} (secondary) = {len(embedding_dict)} unique genes")
        else:
            embedding_dict = primary_dict
            print(f"Loaded embeddings for {len(embedding_dict)} genes")

        print("\nBuilding embedding matrix...")
        embedding_matrix = build_embedding_matrix(list(pert_to_idx.keys()), embedding_dict)
        print(f"Embedding matrix shape: {embedding_matrix.shape}")

    eval_combined_df = pd.concat([calib_df, test_df], ignore_index=True)
    plot_data_summary(train_df, eval_combined_df, args.dataset_name, output_dir=args.output_dir)

    if args.skip_cv:
        best_k = args.k
        print(f"\nSkipping cross-validation, using K = {best_k}")
        cv_results = None
    else:
        k_values = [int(k) for k in args.k_values.split(',')]
        best_k, cv_results = cross_validate_k(
            train_df=train_df,
            label_matrix=label_matrix,
            embedding_matrix=embedding_matrix,
            pert_to_idx=pert_to_idx,
            gene_to_idx=gene_to_idx,
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
    print(f"Final Prediction (K = {best_k})")
    print(f"{'='*70}")

    train_perts = train_df['pert'].unique().tolist()

    # Build combined label/embedding matrix across all splits for neighbor lookup
    all_df = pd.concat([train_df, calib_df, test_df], ignore_index=True)
    print("\nBuilding combined log2fc and label matrix from all splits...")
    _, combined_label_matrix, combined_pert_to_idx, combined_gene_to_idx = build_log2fc_and_label_matrix(all_df)
    print(f"Unique perturbations: {len(combined_pert_to_idx)}")
    print(f"Unique genes: {len(combined_gene_to_idx)}")
    print(f"Label matrix shape: {combined_label_matrix.shape} (perturbations × genes)")

    if use_copath:
        combined_embedding_matrix = None
    else:
        print("\nBuilding combined embedding matrix...")
        combined_embedding_matrix = build_embedding_matrix(list(combined_pert_to_idx.keys()), embedding_dict)
        print(f"Combined embedding matrix shape: {combined_embedding_matrix.shape}")

    def predict_split(df, split_label):
        print(f"\nPredicting on {len(df)} {split_label} pairs (with probabilities for AUROC)...")
        return predict_knn_batch(
            df=df,
            label_matrix=combined_label_matrix,
            pert_to_idx=combined_pert_to_idx,
            gene_to_idx=combined_gene_to_idx,
            train_pert_names=train_perts,
            embedding_matrix=combined_embedding_matrix,
            k=best_k,
            weights=args.weights,
            return_proba=True,
            verbose=False,
            copath_neighbors=copath_neighbors,
            return_uncertain=True,
        )

    y_pred_calib, y_proba_calib, uncertain_calib = predict_split(calib_df, "calibration")
    y_pred_test,  y_proba_test,  uncertain_test  = predict_split(test_df,  "test")

    y_true_calib    = calib_df['label'].values
    y_true_test     = test_df['label'].values
    y_pred_combined = np.concatenate([y_pred_calib, y_pred_test])
    y_proba_combined = np.concatenate([y_proba_calib, y_proba_test], axis=0)
    y_true_combined = np.concatenate([y_true_calib, y_true_test])
    uncertain_combined = np.concatenate([uncertain_calib, uncertain_test])

    metrics_calib    = compute_3class_metrics(y_true_calib,    y_pred_calib,    y_proba_calib)
    metrics_test     = compute_3class_metrics(y_true_test,     y_pred_test,     y_proba_test)
    metrics_combined = compute_3class_metrics(y_true_combined, y_pred_combined, y_proba_combined)

    def print_metrics(metrics, header):
        print(f"\n{'='*70}")
        print(header)
        print(f"{'='*70}")
        print(f"\nBest K: {best_k}")
        print(f"  Accuracy:          {metrics['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"  Precision (macro): {metrics['precision_macro']:.4f}")
        print(f"  Recall (macro):    {metrics['recall_macro']:.4f}")
        print(f"  F1 (macro):        {metrics['f1_macro']:.4f}")
        if 'auroc_ovr_macro' in metrics:
            print(f"  AUROC OvR (macro): {metrics['auroc_ovr_macro']:.4f}")
        print(f"\n  Per-class metrics:")
        for class_idx in range(3):
            class_name = ['Non-DE', 'Up-regulated', 'Down-regulated'][class_idx]
            print(f"  {class_name} (class {class_idx}):")
            print(f"    Precision: {metrics[f'precision_class_{class_idx}']:.4f}")
            print(f"    Recall:    {metrics[f'recall_class_{class_idx}']:.4f}")
            print(f"    F1:        {metrics[f'f1_class_{class_idx}']:.4f}")
            if f'auroc_ovr_class_{class_idx}' in metrics:
                print(f"    AUROC OvR: {metrics[f'auroc_ovr_class_{class_idx}']:.4f}")

    print_metrics(metrics_calib,    "RESULTS — Calibration Set (20%)")
    print_metrics(metrics_test,     "RESULTS — Test Set (20%)")
    print_metrics(metrics_combined, "RESULTS — Combined Calib + Test Set (40%)")

    for split_label, y_true, y_pred in [
        ("calib",    y_true_calib,    y_pred_calib),
        ("test",     y_true_test,     y_pred_test),
        ("combined", y_true_combined, y_pred_combined),
    ]:
        print(f"\n{'='*70}")
        print(f"Detailed Classification Report — {split_label} Set")
        print(f"{'='*70}")
        print(classification_report(
            y_true, y_pred,
            target_names=['Non-DE', 'Up-regulated', 'Down-regulated'],
            zero_division=0
        ))

    # Save 3 metrics CSVs and 3 predictions CSVs (with per-class probabilities)
    for split_label, df_split, metrics, y_pred, y_true, y_proba in [
        ("calib",    calib_df,         metrics_calib,    y_pred_calib,    y_true_calib,    y_proba_calib),
        ("test",     test_df,          metrics_test,     y_pred_test,     y_true_test,     y_proba_test),
        ("combined", eval_combined_df, metrics_combined, y_pred_combined, y_true_combined, y_proba_combined),
    ]:
        metrics_df = pd.DataFrame([metrics])
        metrics_df['best_k'] = best_k
        metrics_df['dataset'] = args.dataset_name
        metrics_path = os.path.join(args.output_dir, f'{args.dataset_name}_knn_metrics_{split_label}.csv')
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Metrics ({split_label}) saved to {metrics_path}")

        pred_df = df_split.copy()
        pred_df['prediction']  = y_pred
        pred_df['correct']     = (y_pred == y_true)
        pred_df['prob_class_0'] = y_proba[:, 0]
        pred_df['prob_class_1'] = y_proba[:, 1]
        pred_df['prob_class_2'] = y_proba[:, 2]
        pred_path = os.path.join(args.output_dir, f'{args.dataset_name}_knn_predictions_{split_label}.csv')
        pred_df.to_csv(pred_path, index=False)
        print(f"Predictions ({split_label}) saved to {pred_path}")

    # Sensitivity analysis on combined set for uncertain co-path pairs
    n_uncertain = uncertain_combined.sum()
    if args.sensitivity_analysis and n_uncertain > 0:
        run_sensitivity_analysis(y_true_combined, y_pred_combined, uncertain_combined,
                                 args.output_dir, args.dataset_name, split_label='combined')
    elif args.sensitivity_analysis and n_uncertain == 0:
        print(f"\nUncertain pairs (co-path zero-neighbor fallbacks): 0/{len(y_true_combined)}")
        print("No uncertain pairs found; sensitivity analysis skipped.")

    print(f"\n{'='*70}")
    print("ERROR ANALYSIS (Combined Calib + Test)")
    print(f"{'='*70}")
    combined_keys = list(zip(eval_combined_df['pert'], eval_combined_df['gene']))
    run_error_analysis_from_results(
        dataset=args.dataset_name,
        test_keys=combined_keys,
        predictions=y_pred_combined,
        true_labels=y_true_combined,
        adata=None,
        task_name="3class_KNN",
        class_names=['Non-DE', 'Up-regulated', 'Down-regulated'],
        output_dir=error_dir,
    )
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE\n")
    print(f"{'='*70}")