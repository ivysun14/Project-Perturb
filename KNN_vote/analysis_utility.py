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
from majority_vote.analysis_utility import compute_3class_metrics


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


# Perturbations with no WikiCrow embedding (union across all 4 datasets).
PERTS_NO_WIKICROW_EMBEDDING: frozenset = frozenset({
    'AARS', 'AC118549.1', 'ALG1L', 'C12orf45', 'C1orf109', 'C7orf26',
    'C9orf16', 'CCDC130', 'CCDC144NL', 'CCDC84', 'CD3EAP', 'DARS', 'EPRS',
    'FAM102B', 'FAM207A', 'FGFR1OP', 'GARS', 'GCOM1', 'H2AFX', 'H2AFZ',
    'H3F3A', 'HARS', 'HIST1H2AB', 'HIST1H2AE', 'HIST1H2AI', 'HIST1H2BB',
    'HIST1H2BC', 'HIST1H2BE', 'HIST1H2BJ', 'HIST1H2BL', 'HIST1H2BM',
    'HIST1H2BN', 'HIST2H2AA3', 'HIST2H2AC', 'HIST2H2BE', 'HIST2H2BF',
    'HIST2H3A', 'HIST2H3D', 'IARS', 'KARS', 'LARS', 'MRPL1', 'MRPL10',
    'MRPL11', 'MRPL13', 'MRPL15', 'MRPL18', 'MRPL20', 'MRPL22', 'MRPL23',
    'MRPL24', 'MRPL28', 'MRPL3', 'MRPL32', 'MRPL33', 'MRPL37', 'MRPL39',
    'MRPL4', 'MRPL41', 'MRPL42', 'MSRB1', 'MT2A', 'MVD', 'MVK', 'MYC',
    'NARS', 'NBEAL1', 'NDUFAF5', 'NEDD8-MDP1', 'NELFE', 'NFATC2IP',
    'NHLRC2', 'NIFK', 'NMT1', 'NOLC1', 'NPB', 'NUMA1', 'NUP107', 'NUP133',
    'NUP155', 'NUP160', 'NUP205', 'NUP35', 'NUP43', 'NUP54', 'NUP85',
    'NUP88', 'NUP93', 'OLFML3', 'ORC3', 'PARN', 'PEF1', 'PFDN6', 'PHB',
    'PLK1', 'PMF1', 'POLA1', 'POLA2', 'PPA1', 'PPME1', 'PRPF38B', 'PRRC2A',
    'PRSS50', 'PSTK', 'PYROXD1', 'QARS', 'RAD51D', 'RARS', 'RBM14-RBM4',
    'RGP1', 'RNF113A', 'RNMT', 'RPL10', 'RPL10A', 'RPL11', 'RPL13', 'RPL14',
    'RPL15', 'RPL17', 'RPL18', 'RPL21', 'RPL23', 'RPL23A', 'RPL24', 'RPL26',
    'RPL27', 'RPL28', 'RPL29', 'RPL30', 'RPL31', 'RPL32', 'RPL34', 'RPL35',
    'RPL35A', 'RPL36', 'RPL36A', 'RPL37', 'RPL37A', 'RPL38', 'RPL39',
    'RPL4', 'RPL41', 'RPL6', 'RPL7', 'RPL7A', 'RPL8', 'RPL9', 'RPLP0',
    'RPLP1', 'RPS10', 'RPS10-NUDT3', 'RPS11', 'RPS12', 'RPS13', 'RPS14',
    'RPS15', 'RPS15A', 'RPS17', 'RPS18', 'RPS2', 'RPS20', 'RPS21', 'RPS3',
    'RPS3A', 'RPS4X', 'RPS7', 'RPS8', 'RPS9', 'SARS', 'SPATA5', 'SPATA5L1',
    'SRBD1', 'STAT5A', 'TAF3', 'TAZ', 'TBCA', 'TBCC', 'TBP', 'TDGF1',
    'TEAD3', 'TEN1', 'TFIP11', 'TMEM127', 'TONSL', 'TOP2A', 'TRIM49C',
    'TRMT6', 'TRRAP', 'TTF2', 'TUBG1', 'TUBGCP4', 'TVP23C', 'TWISTNB',
    'TXN', 'U2SURP', 'UQCRFS1', 'VARS', 'WARS', 'WDR61', 'WDR92', 'XRCC1',
    'ZCRB1', 'ZNF720',
    'ATP5MD', 'C5orf30', 'CARS', 'MARS', 'MRPL17', 'MRPL19', 'MTRNR2L1',
    'MYB', 'POLR2D', 'RPA3', 'RPL12', 'RPL13A', 'RPL18A', 'RPL19', 'RPS16',
    'RPS5', 'SUPT20H', 'TARS', 'TFPT', 'YARS', 'ZNRD1',
    # Additional entries from PerturbQA K562 dataset not covered above.
    'HSPB11', 'METRNL', 'MTF1', 'NR2E1', 'PAICS', 'PDCL', 'PITPNC1', 'PYGB',
    'RFK', 'RIF1', 'ST20-MTHFS', 'TLK2', 'TOMM70', 'TRIM28', 'UHRF1BP1',
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


def run_sensitivity_analysis(y_true: np.ndarray,
                              y_pred: np.ndarray,
                              uncertain_mask: np.ndarray,
                              output_dir: str,
                              dataset_name: str,
                              split_label: str = 'combined') -> None:
    """
    Sensitivity analysis for uncertain pairs (co-path zero-neighbor fallbacks).

    Sets all uncertain pairs to class 1 (Up-regulated) and class 2 (Down-regulated)
    respectively, computing metrics for each scenario. This bounds the range of
    possible metric values given the uncertain fallbacks, in contrast to random
    resampling which estimates an average. A large spread between the two scenarios
    means the metrics are sensitive to how those fallbacks are resolved.

    Args:
        y_true: Ground-truth labels, shape (n_samples,)
        y_pred: Predictions with uncertain pairs set to class 0, shape (n_samples,)
        uncertain_mask: Boolean mask marking uncertain pairs, shape (n_samples,)
        output_dir: Directory to save the results CSV
        dataset_name: Used to name the output file
        split_label: Identifier for which eval split this is (e.g. 'calib', 'test', 'combined')
    """
    n_uncertain = uncertain_mask.sum()
    n_total = len(y_true)

    print(f"\n{'='*70}")
    print(f"Sensitivity Analysis for Uncertain Pairs ({split_label})")
    print(f"{'='*70}")
    print(f"  Uncertain pairs: {n_uncertain}/{n_total} ({100*n_uncertain/n_total:.1f}%)")

    overall_metrics = ['accuracy', 'balanced_accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
    class_names = ['Non-DE', 'Up-regulated', 'Down-regulated']
    rows = []

    for assigned_class in [1, 2]:
        y_pred_trial = y_pred.copy()
        y_pred_trial[uncertain_mask] = assigned_class
        m = compute_3class_metrics(y_true, y_pred_trial, y_proba=None)

        print(f"\n  All uncertain → class {assigned_class} ({class_names[assigned_class]}):")
        for metric_name in overall_metrics:
            print(f"    {metric_name}: {m[metric_name]:.4f}")
        print(f"  Per-class metrics:")
        for i in range(3):
            print(f"    {class_names[i]}: "
                  f"P={m[f'precision_class_{i}']:.4f}  "
                  f"R={m[f'recall_class_{i}']:.4f}  "
                  f"F1={m[f'f1_class_{i}']:.4f}")

        row = dict(m)
        row['uncertain_assigned_class'] = assigned_class
        rows.append(row)

    sens_df = pd.DataFrame(rows)
    sens_path = os.path.join(output_dir, f'{dataset_name}_sensitivity_{split_label}.csv')
    sens_df.to_csv(sens_path, index=False)
    print(f"\n  Sensitivity analysis saved to {sens_path}")
