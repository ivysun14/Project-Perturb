"""
knn_classifier.py
=================

Weighted KNN classifier for perturbation-gene pair prediction. Builds a 3-class KNN classifier:
- Class 0: Non-DE
- Class 1: Up-regulated
- Class 2: Down-regulated

The classifier predicts the effect of a perturbation on a gene based on:
- Distance: euclidean distance between perturbations represented by GenePT embeddings
- K nearest neighbors: K determined via 5-fold cross-validation
- Weighted voting: weight = inverse distance

Input CSV format (from DE_DOC_stats.py):
- pert: perturbation name
- gene: gene name
- label: 0 (non-DE), 1 (up-regulation), 2 (down-regulation)
- split: 'train' or 'test'
"""

import os
import sys
import argparse
import zipfile
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import (accuracy_score,
                             balanced_accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score,
                             roc_auc_score,
                             classification_report)

# Add parent directory to path for importing utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baseline.analysis_utility import compute_3class_metrics, plot_data_summary
from baseline.error_utility import run_error_analysis_from_results


def load_de_doc_file(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load DE_DOC CSV file and split by pre-existing train/test split.

    Args:
        file_path: Path to DE_DOC CSV file with 'split' column

    Returns:
        Tuple of (train_df, test_df)
    """

    df = pd.read_csv(file_path)

    print(f"\nLoaded DE+DOC file: {len(df)} perturbation-gene pairs")
    print(f"3-Class label distribution:")
    print(f"  Non-DE (0): {(df['label'] == 0).sum()}")
    print(f"  Up-regulated (1): {(df['label'] == 1).sum()}")
    print(f"  Down-regulated (2): {(df['label'] == 2).sum()}")

    if 'split' not in df.columns:
        raise ValueError("CSV file does not have train test split!")

    train_df = df[df['split'] == 'train'].copy()
    test_df = df[df['split'] == 'test'].copy()

    print(f"Train set: {len(train_df)} pairs ({train_df['pert'].nunique()} perturbations)")
    print(f"Test set: {len(test_df)} pairs ({test_df['pert'].nunique()} perturbations)")

    return train_df, test_df


def load_genePT_embedding(file_path: str) -> Dict[str, list]:
    """
    Load in zipped GenePT embeddings for genes.

    Args:
        file_path: path to zipped GenePT outputs
    
    Returns:
        Dictionary of {"gene": [embedding]}
    """

    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        pickle_path = f'GenePT_emebdding_v2/GenePT_gene_protein_embedding_model_3_text.pickle.'
        with zip_ref.open(pickle_path) as f:
            gene_embeddings_NCBI_UniProt = pickle.load(f)
    
    return gene_embeddings_NCBI_UniProt


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
    # Get unique perturbations and genes
    perturbations = sorted(df['pert'].unique())
    genes = sorted(df['gene'].unique())

    # Create mapping for perturbations and genes
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

    # Check for missing values
    n_missing_log2fc = np.isnan(log2fc_matrix).sum()
    n_missing_label = np.isnan(label_matrix).sum()
    if n_missing_log2fc > 0:
        print(f"    WARNING: {n_missing_log2fc} missing values in log2FC matrix "
              f"    ({100*n_missing_log2fc/(n_perts*n_genes):.2f}%)")
    if n_missing_label > 0:
        print(f"    WARNING: {n_missing_label} missing values in label matrix "
              f"    ({100*n_missing_label/(n_perts*n_genes):.2f}%)")

    return log2fc_matrix, label_matrix, pert_to_idx, gene_to_idx


def build_embedding_matrix(pert_to_idx: Dict[str, int],
                           embedding_dict: Dict[str, list]) -> np.ndarray:
    """
    Build an embedding matrix for perturbations provided.

    Args:
        pert_to_idx: Dict mapping perturbation name to row index in matrices
        embedding_dict: Dictionary of perturbation gene name to their embedding in a list

    Returns:
        Numpy array of shape (n_perturbations, embed_dim)
    """
    n_pert = len(pert_to_idx)
    embed_dim = len(next(iter(embedding_dict.values())))
    embed_matrix = np.full((n_pert, embed_dim), np.nan, dtype=np.float64)
    pert_available = embedding_dict.keys()
    n_pert_unavail = 0

    for pert, ind in pert_to_idx.items():
        if pert not in pert_available:
            n_pert_unavail += 1
            print(f"    WARNING: Perturbation {pert} does not have pre-computed embedding from GenePT!")
        embed_matrix[ind] = np.array(embedding_dict[pert], dtype=np.float64)

    # raise error if no matching embedding from GenePT
    if n_pert_unavail > 0:
        raise ValueError(f"\nIn total {n_pert_unavail} perturbations do not have matching embeddings from GenePT!")

    # Check for any missing entry in the embedding matrix
    n_missing_embedding = np.isnan(embed_matrix).sum()
    if n_missing_embedding > 0:
        print(f"    WARNING: {n_missing_embedding} missing values in embedding matrix "
              f"    ({100*n_missing_embedding/(n_pert*embed_dim):.2f}%)")

    return embed_matrix


def find_k_nearest_neighbors(query_pert_embeddings: np.ndarray,
                             train_pert_embeddings: np.ndarray,
                             k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find K nearest neighbors for query perturbations based on embeddings.

    Args:
        query_pert_embeddings: Query perturbation embeddings (n_query, embed_dim)
        train_pert_embeddings: Training perturbation embeddings (n_train, embed_dim)
        k: Number of neighbors

    Returns:
        distances: Array of shape (n_query, k) with distances to neighbors
        indices: Array of shape (n_query, k) with indices of neighbors in train set
    """
    # Make sure enough training perturbations to find k neighbors
    uniq_pert = train_pert_embeddings.shape[0]
    use_k = k
    if k > uniq_pert:
        use_k = uniq_pert
        print(f"    WARNING: Only {uniq_pert} training perturbations, reduce k down to {use_k}")

    # Fit k nearest neighbors model on training data
    nbrs = NearestNeighbors(n_neighbors=use_k, algorithm='auto', metric='euclidean')
    nbrs.fit(train_pert_embeddings)

    # Find k-nearest neighbors for all queries at once
    distances, indices = nbrs.kneighbors(query_pert_embeddings)

    # For each query pert, check for distance ties (consecutive neighbors have same distance)
    n_queries = len(query_pert_embeddings)
    n_ties = 0
    for i in range(n_queries):
        if use_k > 1:
            distance_diffs = np.diff(distances[i])
            if np.any(distance_diffs < 1e-10):  # Allow floating point tolerance
                print(f"    \nDistances: {distances[i]}")
                n_ties += 1

    if n_ties > 0:
        print(f"  WARNING: {n_ties}/{n_queries} queries have distance ties among {use_k} neighbors")

    return distances, indices


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
        weights: 'distance' for inverse distance weighting, 'uniform' for equal weights

    Returns:
        Tuple of (predicted_label, class_probabilities):
            - predicted_label: int (0, 1, or 2)
            - class_probabilities: np.ndarray of shape (3,) with probabilities for each class
    """

    # Get labels from neighbors for the target gene
    neighbor_labels = label_matrix[train_pert_indices[neighbor_indices], target_gene_idx]
    neighbor_labels = neighbor_labels.astype(int)

    # Compute class votes based on weighting scheme
    if weights == 'uniform':
        class_votes = np.bincount(neighbor_labels, minlength=3).astype(float)
    else:
        # Inverse distance weighting (add small epsilon to avoid division by zero)
        neighbor_weights = 1.0 / (neighbor_distances + 1e-10)
        class_votes = np.zeros(3)
        for label, weight in zip(neighbor_labels, neighbor_weights):
            class_votes[label] += weight

    # Compute normalized probabilities
    class_probabilities = class_votes / class_votes.sum()

    # Determine predicted class
    max_vote = class_votes.max()
    tied_classes = np.where(np.abs(class_votes - max_vote) < 1e-10)[0]

    if len(tied_classes) > 1:
        prediction = np.random.choice(tied_classes)
        print(f"Vote tie: classes {tied_classes} tied with vote={max_vote:.4f}. Randomly choosing class {prediction}.")
    else:
        prediction = tied_classes[0]

    return prediction, class_probabilities


def predict_knn_batch(df: pd.DataFrame,
                      label_matrix: np.ndarray,
                      pert_to_idx: Dict[str, int],
                      gene_to_idx: Dict[str, int],
                      train_pert_names: List[str],
                      embedding_matrix: np.ndarray,
                      k: int,
                      weights: str = 'distance',
                      return_proba: bool = False,
                      verbose: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Predict labels for a batch of (perturbation, gene) pairs using embedding-based KNN.

    Args:
        df: DataFrame with pert-gene pairs to predict for
        label_matrix: Perturbation × gene matrix of labels
        pert_to_idx: Dict mapping perturbation name to row index in matrices
        gene_to_idx: Dict mapping gene name to column index in matrices
        train_pert_names: List of training perturbation names
        embedding_matrix: Embedding matrix for all perturbations
        k: Number of neighbors
        weights: 'distance' for inverse distance weighting
        return_proba: Whether to return both predictions and probabilities
        verbose: Whether to print progress

    Returns:
        If return_proba=False: (predictions, None) where predictions is array of shape (n_samples,)
        If return_proba=True: (predictions, probabilities) where probabilities is array of shape (n_samples, 3)
    """
    # Get train perturbation indices
    train_pert_indices = np.array([pert_to_idx[p] for p in train_pert_names])

    # Get query perturbation indices
    unique_query_perts = df['pert'].unique()
    query_pert_indices = np.array([pert_to_idx[p] for p in unique_query_perts])

    # Build query and train embedding matrices
    query_emb = embedding_matrix[query_pert_indices]
    train_emb = embedding_matrix[train_pert_indices]

    # Precompute neighbors for all query perturbations
    if verbose:
        print(f"  Finding {k} nearest neighbors for {len(unique_query_perts)} query perturbations...")
    distances, indices = find_k_nearest_neighbors(query_emb, train_emb, k)

    # Create mapping from perturbation to its neighbors and corresponding distances
    pert_to_neighbors = {p: (indices[i], distances[i]) for i, p in enumerate(unique_query_perts)}

    # Predict for each (pert, gene) pair
    predictions = []
    probabilities = [] if return_proba else None
    n_total = len(df)

    for i, (_, row) in enumerate(df.iterrows()):
        if verbose and (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1}/{n_total} pairs...")

        pert = row['pert']
        gene = row['gene']
        gene_idx = gene_to_idx[gene]

        # Get precomputed neighbors for this perturbation
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

    predictions = np.array(predictions)
    if return_proba:
        probabilities = np.array(probabilities)

    return predictions, probabilities


def cross_validate_k(train_df: pd.DataFrame,
                     label_matrix: np.ndarray,
                     embedding_matrix: np.ndarray,
                     pert_to_idx: Dict[str, int],
                     gene_to_idx: Dict[str, int],
                     k_values: List[int] = [1, 3, 5, 7, 9],
                     n_folds: int = 5,
                     weights: str = 'distance',
                     random_seed: int = 305) -> Tuple[int, Dict[int, Dict[str, float]]]:
    """
    Perform K-fold cross-validation to select optimal K using embedding-based KNN.

    Args:
        train_df: Training DataFrame with training pert-gene pairs
        label_matrix: Perturbation × gene matrix of labels
        embedding_matrix: Perturbation embedding matrix
        pert_to_idx: Dict mapping perturbation name to row index in matrices
        gene_to_idx: Dict mapping gene name to column index in matrices
        k_values: List of K values to try
        n_folds: Number of folds
        weights: 'distance' for inverse distance weighting
        random_seed: Random seed for reproducibility

    Returns:
        best_k: Optimal K value
        cv_results: Dict mapping K -> metrics dict
    """
    print(f"\n{'='*70}")
    print(f"Cross-Validation for K Selection")
    print(f"{'='*70}")
    print(f"K values to try: {k_values}")
    print(f"Number of folds: {n_folds}")

    # Get unique training perturbations
    train_perts = train_df['pert'].unique()

    # Set up K-fold by perturbations
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    cv_results = {k: defaultdict(list) for k in k_values}

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_perts)):
        print(f"\nFold {fold + 1}/{n_folds}")

        # Split perturbations
        fold_train_perts = train_perts[train_idx]
        fold_val_perts = train_perts[val_idx]

        # Get validation data
        val_df = train_df[train_df['pert'].isin(fold_val_perts)]

        # Get true labels
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
                return_proba=False,  # don't compute probabilities during CV
                verbose=False        # this will only be true during actual test set prediction
            )

            # Compute metrics (without AUROC during CV)
            metrics = compute_3class_metrics(y_true, y_pred, y_proba=None)

            for metric_name, value in metrics.items():
                cv_results[k][metric_name].append(value)

            print(f" balanced_acc={metrics['balanced_accuracy']:.4f}")

    # Average results across folds
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

    # Select best K based on balanced accuracy
    best_k = max(k_values, key=lambda k: avg_results[k]['balanced_accuracy'])
    print(f"\nBest K (by balanced accuracy): {best_k}")

    return best_k, avg_results


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Weighted KNN classifier for 3-class perturbation-gene prediction')
    parser.add_argument('--de_doc_file', type=str, required=True, help='Path to DE_DOC CSV file (must have split column)')
    parser.add_argument('--embedding_file', type=str, required=True, help='Path to GenePT embedding zip file')
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name (for output files)')
    parser.add_argument('--k_values', type=str, default='1,3,5,7,9', help='Comma-separated K values to try (default: 1,3,5,7,9)')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds for cross-validation (default: 5)')
    parser.add_argument('--skip_cv', action='store_true', help='Skip cross-validation and use specified K')
    parser.add_argument('--k', type=int, default=5, help='K value to use if skipping CV (default: 5)')
    parser.add_argument('--weights', type=str, default='distance', choices=['distance', 'uniform'], help='Weight function for KNN (default: distance)')
    parser.add_argument('--random_seed', type=int, default=305, help='Random seed for reproducibility (default: 305)')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory for results (default: results)')
    parser.add_argument('--subsample', type=int, default=None, help='Subsample perturbations per train and test data for quick check of the algorithm (default: None = use all)')

    args = parser.parse_args()

    print("="*70)
    print("Weighted KNN Classifier for Perturbation-Gene Prediction")
    print("="*70)
    print(f"\nDataset: {args.dataset_name}")
    print(f"Weights: {args.weights}")
    print(f"Random seed: {args.random_seed}")
    print(f"Output results at: {args.output_dir}")
    print(f"Skip CV: {args.skip_cv}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load DE+DOC data
    train_df, test_df = load_de_doc_file(args.de_doc_file)

    # TESTING PURPOSE ONLY: subsample a small amount of train and test perturbations for quick check
    if args.subsample is not None:
        train_pert = sorted(train_df['pert'].unique())
        test_pert = sorted(test_df['pert'].unique())
        train_pert_subsample = train_pert[:args.subsample]
        test_pert_subsample = test_pert[:args.subsample]
        train_df = train_df[train_df['pert'].isin(train_pert_subsample)]
        test_df = test_df[test_df['pert'].isin(test_pert_subsample)]

    print("\nBuilding log2fc and label matrix from train data...")
    _, label_matrix, pert_to_idx, gene_to_idx = build_log2fc_and_label_matrix(train_df)
    print(f"    Unique perturbations: {len(pert_to_idx)}")
    print(f"    Unique genes: {len(gene_to_idx)}")
    print(f"    Label matrix shape: {label_matrix.shape} (perturbations × genes)")

    # Load perturbation embeddings
    embedding_dict = load_genePT_embedding(args.embedding_file)
    print(f"\nLoaded embeddings for {len(embedding_dict)} genes")

    # Build embedding matrix for training perturbations
    print("\nBuilding embedding matrix...")
    embedding_matrix = build_embedding_matrix(pert_to_idx, embedding_dict)
    print(f"Embedding matrix shape: {embedding_matrix.shape}")

    # Plot train/test data summary
    plot_data_summary(train_df, test_df, args.dataset_name, output_dir=args.output_dir)

    # Cross-validation for K selection if not skipped
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
            random_seed=args.random_seed
        )

        # Save CV results
        cv_results_df = pd.DataFrame(cv_results).T
        cv_results_df.index.name = 'k'
        cv_results_path = os.path.join(args.output_dir, f'{args.dataset_name}_cv_results.csv')
        cv_results_df.to_csv(cv_results_path)
        print(f"\nCV results saved to {cv_results_path}")

    # Final prediction on test set
    print(f"\n{'='*70}")
    print(f"Final Prediction on Test Set (K = {best_k})")
    print(f"{'='*70}")

    # Get training perturbations
    train_perts = train_df['pert'].unique().tolist()

    # Build combined matrices including test perturbations
    combined_df = pd.concat([train_df, test_df])

    print("\nBuilding combined log2fc and label matrix from train+test df...")
    _, combined_label_matrix, combined_pert_to_idx, combined_gene_to_idx = build_log2fc_and_label_matrix(combined_df)
    print(f"Unique perturbations: {len(combined_pert_to_idx)}")
    print(f"Unique genes: {len(combined_gene_to_idx)}")
    print(f"Label matrix shape: {combined_label_matrix.shape} (perturbations × genes)")

    # Build combined embedding matrix
    print("\nBuilding combined embedding matrix...")
    combined_embedding_matrix = build_embedding_matrix(combined_pert_to_idx, embedding_dict)
    print(f"Combined embedding matrix shape: {combined_embedding_matrix.shape}")

    print(f"\nPredicting on {len(test_df)} test pairs (with probabilities for AUROC)...")
    y_pred, y_proba = predict_knn_batch(
        df=test_df,
        label_matrix=combined_label_matrix,
        pert_to_idx=combined_pert_to_idx,
        gene_to_idx=combined_gene_to_idx,
        train_pert_names=train_perts,
        embedding_matrix=combined_embedding_matrix,
        k=best_k,
        weights=args.weights,
        return_proba=True,
        verbose=True
    )
    y_true = test_df['label'].values

    # Compute metrics including AUROC
    metrics = compute_3class_metrics(y_true, y_pred, y_proba)

    # Print results
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
            print(f"    AUROC OvR: {metrics[f'auroc_ovr_class_{class_idx}']:.4f}")

    # Detailed classification report
    print(f"\n{'='*70}")
    print("Detailed Classification Report")
    print(f"{'='*70}")
    print(classification_report(
        y_true,
        y_pred,
        target_names=['Non-DE', 'Up-regulated', 'Down-regulated'],
        zero_division=0
    ))

    # Save predictions
    predictions_df = test_df.copy()
    predictions_df['prediction'] = y_pred
    predictions_df['correct'] = (y_pred == y_true)
    # Add predicted probabilities for each class
    predictions_df['prob_class_0'] = y_proba[:, 0]
    predictions_df['prob_class_1'] = y_proba[:, 1]
    predictions_df['prob_class_2'] = y_proba[:, 2]
    predictions_path = os.path.join(args.output_dir, f'{args.dataset_name}_knn_predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)
    print(f"\nPredictions saved to {predictions_path}")

    # Save final metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df['best_k'] = best_k
    metrics_df['dataset'] = args.dataset_name
    metrics_path = os.path.join(args.output_dir, f'{args.dataset_name}_knn_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")

    # Run error analysis
    print(f"\n{'='*70}")
    print("ERROR ANALYSIS")
    print(f"{'='*70}")

    # Create test_keys as list of (pert, gene) tuples
    test_keys = list(zip(test_df['pert'], test_df['gene']))

    analyzer, error_results = run_error_analysis_from_results(
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
