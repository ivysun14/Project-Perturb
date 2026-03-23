"""
analysis_utility.py
===================
Shared utilities for KNN classifiers.
"""

import json
import pickle
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union
from sklearn.neighbors import NearestNeighbors

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baseline.analysis_utility import compute_3class_metrics


# Perturbations with no ESM2 embedding (union across all 4 datasets).
PERTS_NO_ESM2_EMBEDDING: frozenset = frozenset({
    'AARS', 'AC118549.1', 'ALG1L', 'ATP5MD', 'C12orf45', 'C14orf178',
    'C5orf30', 'C9orf16', 'CARS', 'CCDC130', 'CCDC144NL', 'CCDC84',
    'CD3EAP', 'DARS', 'EPRS', 'FAM207A', 'FGFR1OP', 'GARS', 'H2AFX',
    'H2AFZ', 'H3F3A', 'HARS', 'HIST1H2AB', 'HIST1H2AE', 'HIST1H2AI',
    'HIST1H2BB', 'HIST1H2BC', 'HIST1H2BE', 'HIST1H2BJ', 'HIST1H2BL',
    'HIST1H2BM', 'HIST1H2BN', 'HIST2H2AA3', 'HIST2H2AC', 'HIST2H2BE',
    'HIST2H2BF', 'HIST2H3A', 'HIST2H3D', 'IARS', 'KARS', 'LARS', 'MARS',
    'NARS', 'QARS', 'RARS', 'SARS', 'TARS', 'TAZ', 'TWISTNB', 'VARS',
    'WARS', 'WDR92', 'YARS', 'ZNRD1', 'ADSS', 'ATP5MPL', 'FAM122A', 'HIST2H2AB',
})


# Perturbations with no co-path neighbors (union across all 4 datasets).
PERTS_NO_COPATH_NEIGHBORS: frozenset = frozenset({
    'AARS', 'ABT1', 'AC118549.1', 'ALG1L', 'ARGLU1', 'ARL6IP6', 'ARMC7',
    'ATP5MD', 'BTF3L4', 'C12orf45', 'C12orf60', 'C14orf178', 'C16orf86',
    'C17orf58', 'C18orf21', 'C19orf25', 'C19orf53', 'C1orf109', 'C1orf131',
    'C5orf30', 'C6orf15', 'C7orf26', 'C7orf50', 'C9orf16', 'C9orf78',
    'CARF', 'CARS', 'CCDC130', 'CCDC137', 'CCDC144A', 'CCDC144NL',
    'CCDC174', 'CCDC84', 'CCDC86', 'CD3EAP', 'CLCC1', 'COMTD1', 'CT45A5',
    'DARS', 'DDN', 'DDX51', 'DDX55', 'DPPA2', 'EIF1AD', 'EPRS', 'FAM102B',
    'FAM133B', 'FAM136A', 'FAM207A', 'FAM229A', 'FAM32A', 'FAM72D',
    'FBRSL1', 'FBXO42', 'FGFR1OP', 'FNBP4', 'FRG2', 'GARS', 'GCOM1',
    'GOLGA6L1', 'GPN1', 'GPN2', 'GPN3', 'H2AFX', 'H2AFZ', 'H3F3A',
    'HARS', 'HIST1H2AB', 'HIST1H2AE', 'HIST1H2AI', 'HIST1H2BB', 'HIST1H2BC',
    'HIST1H2BE', 'HIST1H2BJ', 'HIST1H2BL', 'HIST1H2BM', 'HIST1H2BN',
    'HIST2H2AA3', 'HIST2H2AC', 'HIST2H2BE', 'HIST2H2BF', 'HIST2H3A',
    'HIST2H3D', 'IARS', 'KARS', 'KIAA1143', 'KRTAP4-7', 'LARS', 'LCN10',
    'LENG8', 'LRRC37A3', 'LRRC37B', 'LSM12', 'MARS', 'MEIS3', 'MEX3A',
    'MOB4', 'MOK', 'NAF1', 'NARS', 'NBPF15', 'NBPF3', 'NEDD8-MDP1',
    'NKAPD1', 'NOP9', 'PCNX3', 'PNISR', 'POGZ', 'POLR2M', 'PPP1R37',
    'PRRC2A', 'PSMG4', 'QARS', 'RARS', 'RBM33', 'RBM48', 'RCCD1', 'RGPD6',
    'RPAP1', 'RPF1', 'RRP1', 'RSRC2', 'SARS', 'SCAF1', 'SDR39U1',
    'SLC35A4', 'SLC35G2', 'SLC35G6', 'SLC7A6OS', 'SMAGP', 'SNTB2',
    'SPAG7', 'SRA1', 'SS18L2', 'SSX4', 'TARS', 'TBC1D28', 'TEN1', 'TIGD1',
    'TMEM214', 'TMEM240', 'TOR2A', 'TTC1', 'TTC27', 'TWISTNB', 'TXNL4B',
    'UBL5', 'UTP23', 'VARS', 'VEZT', 'WARS', 'WBP1', 'WBP1L', 'WDR25',
    'WDR7', 'WDR70', 'WDR92', 'YARS', 'ZBTB44', 'ZC3H18', 'ZCCHC9',
    'ZFC3H1', 'ZNF574', 'ZNF687', 'ZNF720', 'ZNHIT2', 'ZRANB2',
})


def load_embedding_from_pkl(file_path: str) -> Dict[str, np.ndarray]:
    """
    Load embeddings from a pickle file.

    Args:
        file_path: Path to pkl file

    Returns:
        Dictionary of {"gene": np.array[embedding]}
    """
    with open(file_path, "rb") as f:
        embeddings = pickle.load(f)
    return embeddings


def load_copath_neighbors(file_path: str) -> Dict[str, list]:
    """
    Load pre-computed co-pathway neighbors from a JSON file produced by
    precompute_pathways.py.

    Expected format:  {gene: [[neighbor, jaccard_sim], ...]}
    Neighbors are sorted by descending Jaccard similarity.

    Returns:
        Dict mapping gene_name -> [[neighbor_name, jaccard_sim], ...]
    """
    with open(file_path, 'r') as f:
        raw = json.load(f)
    print(f"  Loaded co-path neighbors for {len(raw)} genes from {file_path}")
    return raw


def build_embedding_matrix(pert_names: List[str],
                           embedding_dict: Dict[str, Union[list, np.ndarray]]) -> np.ndarray:
    """
    Build an embedding matrix for perturbations, ordered by pert_names.

    Args:
        pert_names: List of perturbation names (order determines row order)
        embedding_dict: Dictionary mapping perturbation name to embedding vector

    Returns:
        Numpy array of shape (n_perturbations, embed_dim)
    """
    n_pert = len(pert_names)
    embed_dim = len(next(iter(embedding_dict.values())))
    embed_matrix = np.full((n_pert, embed_dim), np.nan, dtype=np.float64)
    n_pert_unavail = 0

    for i, pert in enumerate(pert_names):
        if pert not in embedding_dict:
            n_pert_unavail += 1
            warnings.warn(f"Perturbation '{pert}' does not have a pre-computed embedding.")
            embed_matrix[i] = np.zeros(embed_dim, dtype=np.float64)
        else:
            embed_matrix[i] = np.asarray(embedding_dict[pert], dtype=np.float64)

    if n_pert_unavail > 0:
        raise ValueError(f"{n_pert_unavail} perturbations do not have embeddings.")

    n_missing = np.isnan(embed_matrix).sum()
    if n_missing > 0:
        warnings.warn(
            f"{n_missing} missing values in embedding matrix "
            f"({100*n_missing/(n_pert*embed_dim):.2f}%)"
        )

    return embed_matrix


def find_k_nearest_neighbors(query_pert_embeddings: np.ndarray,
                             train_pert_embeddings: np.ndarray,
                             k: int,
                             check_ties: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find K nearest neighbors for query perturbations using cosine dissimilarity.

    Args:
        query_pert_embeddings: Query perturbation embeddings (n_query, embed_dim)
        train_pert_embeddings: Training perturbation embeddings (n_train, embed_dim)
        k: Number of neighbors; pass len(train) to retrieve all neighbors sorted
        check_ties: Whether to warn about distance ties among the returned neighbors.
                    Set to False when retrieving the full sorted neighbor list (k=n_train),
                    as ties anywhere in the list are expected and not meaningful.

    Returns:
        distances: Array of shape (n_query, k) with cosine dissimilarities to neighbors
        indices: Array of shape (n_query, k) with indices of neighbors in train set
    """
    # sklearn requires n_neighbors <= n_samples_fit
    uniq_pert = train_pert_embeddings.shape[0]
    use_k = k
    if k > uniq_pert:
        use_k = uniq_pert
        warnings.warn(
            f"Only {uniq_pert} training perturbations available; reducing k from {k} to {use_k}."
        )

    nbrs = NearestNeighbors(n_neighbors=use_k, algorithm='brute', metric='cosine')
    nbrs.fit(train_pert_embeddings)
    distances, indices = nbrs.kneighbors(query_pert_embeddings)

    if check_ties:
        n_queries = len(query_pert_embeddings)
        n_ties = 0
        for i in range(n_queries):
            if use_k > 1 and np.any(np.diff(distances[i]) < 1e-10):
                print(f"        Distances: {distances[i]}")
                n_ties += 1
        if n_ties > 0:
            warnings.warn(
                f"{n_ties}/{n_queries} queries have distance ties among {use_k} neighbors."
            )

    return distances, indices


# TODO: Instead of sampling into random classes, do sensitivity analysis
# such that all uncertain pert-gene pairs are assigned class 1 and 2.
def run_bootstrap(y_true: np.ndarray,
                  y_pred: np.ndarray,
                  uncertain_mask: np.ndarray,
                  n_bootstrap: int,
                  random_seed: int,
                  output_dir: str,
                  dataset_name: str) -> None:
    """
    Bootstrap uncertainty estimation for predictions with uncertain pairs.

    Uncertain pairs (those that fell back to class 0 (non-DE) because no
    valid neighbors were available) are re-sampled uniformly from {0, 1, 2}
    across n_bootstrap trials.  All other predictions are held fixed.

    A large std indicates the reported metrics are sensitive to how those
    fallback pairs are resolved; a small std means the uncertain pairs are
    too few or too evenly distributed to meaningfully affect the outcome.

    AUROC is excluded because uncertain pairs always have [1/3, 1/3, 1/3]
    probabilities, which are constant across trials and do not affect AUROC.

    Args:
        y_true: Ground-truth labels, shape (n_samples,)
        y_pred: Deterministic predictions (uncertain pairs set to class 0), shape (n_samples,)
        uncertain_mask: Boolean mask marking uncertain pairs, shape (n_samples,)
        n_bootstrap: Number of Monte Carlo trials
        random_seed: Seed for the RNG (for reproducibility)
        output_dir: Directory to save the trial-level CSV
        dataset_name: Used to name the output file
    """
    n_uncertain = uncertain_mask.sum()
    n_total = len(y_true)

    print(f"\n{'='*70}")
    print(f"Bootstrap Uncertainty Estimation")
    print(f"{'='*70}")
    print(f"  Uncertain pairs: {n_uncertain}/{n_total} ({100*n_uncertain/n_total:.1f}%)")
    print(f"  Bootstrap trials: {n_bootstrap}")

    rng = np.random.default_rng(random_seed)
    bootstrap_metrics = defaultdict(list)
    for _ in range(n_bootstrap):
        y_pred_trial = y_pred.copy()
        y_pred_trial[uncertain_mask] = rng.integers(0, 3, size=n_uncertain)
        m = compute_3class_metrics(y_true, y_pred_trial, y_proba=None)
        for metric_name, val in m.items():
            bootstrap_metrics[metric_name].append(val)

    overall_metrics = ['accuracy', 'balanced_accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
    per_class_metrics = [
        f'{stat}_class_{i}'
        for i in range(3)
        for stat in ['precision', 'recall', 'f1']
    ]

    print(f"\n  Overall metrics (mean ± std over {n_bootstrap} trials):")
    for metric_name in overall_metrics:
        vals = bootstrap_metrics[metric_name]
        print(f"    {metric_name}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    class_names = ['Non-DE', 'Up-regulated', 'Down-regulated']
    print(f"\n  Per-class metrics (mean ± std over {n_bootstrap} trials):")
    for i in range(3):
        print(f"    {class_names[i]} (class {i}):")
        for stat in ['precision', 'recall', 'f1']:
            vals = bootstrap_metrics[f'{stat}_class_{i}']
            print(f"      {stat}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # Save trial-level results so downstream analysis can compute any summary stat
    all_summary_metrics = overall_metrics + per_class_metrics
    bootstrap_df = pd.DataFrame({m: bootstrap_metrics[m] for m in all_summary_metrics})
    bootstrap_df.index.name = 'trial'
    bootstrap_path = os.path.join(output_dir, f'{dataset_name}_bootstrap_metrics.csv')
    bootstrap_df.to_csv(bootstrap_path)
    print(f"\n  Full bootstrap results saved to {bootstrap_path}")
