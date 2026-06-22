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
import json
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


def load_de_doc_file(file_path: str, splits: List[str]) -> pd.DataFrame:
    """
    Load a DE_DOC CSV and return rows matching any of the requested split_calib values.

    Args:
        file_path: Path to DE_DOC CSV with 'split_calib' column
        splits: List of split labels to include, e.g. ['train', 'test'] or ['calibration']
    """
    df = pd.read_csv(file_path)
    print(f"\nLoaded: {file_path}")
    print(f"  Total pairs: {len(df)}")
    print(f"  Non-DE(0): {(df['label']==0).sum()}  "
          f"Up(1): {(df['label']==1).sum()}  "
          f"Down(2): {(df['label']==2).sum()}")
    if 'split_calib' not in df.columns:
        raise ValueError(f"CSV file does not have a 'split_calib' column: {file_path}")
    split_df = df[df['split_calib'].isin(splits)].copy()
    print(f"  {splits} split(s): {len(split_df)} pairs "
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
                      embedding_dict: Optional[Dict[str, Union[list, np.ndarray]]],
                      k: int,
                      weights: str = 'distance',
                      return_proba: bool = False,
                      verbose: bool = True,
                      copath_neighbors: Optional[Dict[str, list]] = None,
                      ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Batch KNN prediction for all (pert, gene) pairs in test_df.

    Exactly one neighbor source must be supplied:
      embedding_dict   -- uses cosine dissimilarity at run time
      copath_neighbors -- uses pre-computed Jaccard-similarity neighbors
                         ({pert: [[neighbor, jaccard_sim], ...]})

    Args:
        test_df          : Test pairs with columns [pert, gene, label].
        train_df         : Merged training DataFrame with columns [pert, gene, label].
        train_pert_names : Names of training perturbations (the neighbor pool).
        embedding_dict   : {pert_name: embedding_vector}; None when using co-path.
        k                : Desired number of valid neighbors per prediction.
        weights          : 'distance' or 'uniform'.
        return_proba     : Whether to return per-class probabilities.
        verbose          : Print progress every 10 000 pairs.
        copath_neighbors : Pre-loaded co-path neighbor dict; None for embedding modes.

    Returns:
        predictions   : (n_test,) int array.
        probabilities : (n_test, 3) float array if return_proba=True, else None.
        uncertain_mask: (n_test,) bool — True where no valid neighbor was found at all.
    """
    if embedding_dict is None and copath_neighbors is None:
        raise ValueError("Provide either embedding_dict or copath_neighbors.")
    if embedding_dict is not None and copath_neighbors is not None:
        raise ValueError("Provide only one of embedding_dict or copath_neighbors, not both.")

    # Build label lookup from merged training data
    train_label_dict = {(row['pert'], row['gene']): int(row['label'])
                        for _, row in train_df.iterrows()}

    unique_query_perts = test_df['pert'].unique().tolist()
    n_train            = len(train_pert_names)

    if copath_neighbors is not None:
        # Co-path path: neighbors from pre-computed Jaccard similarity
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
                # filter to training perts (already sorted desc by Jaccard sim)
                train_nb = [(n, s) for n, s in copath_neighbors[p] if n in train_pert_set]
                pert_to_neighbors[p] = (
                    [n for n, _ in train_nb],
                    np.array([1.0 - s for _, s in train_nb]),  # dist = 1 − jaccard
                )
        if no_copath:
            warnings.warn(
                f"{len(no_copath)} query perturbations not found in co-path neighbors "
                f"and will default to class 0 (non-DE): {sorted(no_copath)}"
            )
    else:
        # Embedding path: retrieve ALL training neighbors sorted by cosine distance
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

    parser = argparse.ArgumentParser(description='KNN transfer: train on N cell lines, test on 1 (GenePT / ESM2 / WikiCrow / co-path).')
    parser.add_argument('--train_de_doc_file', type=str, required=True, nargs='+', help='DE_DOC CSV file(s) for source datasets (train split used)')
    parser.add_argument('--test_de_doc_file', type=str, required=True, help='DE_DOC CSV file for the target dataset (test split used)')
    parser.add_argument('--embedding_file', type=str, default=None, help='Path to primary embedding pkl file (required for genept/esm2/wikicrow modes)')
    parser.add_argument('--embedding_file2', type=str, default=None, help='(optional) Path to secondary embedding pkl to merge in')
    parser.add_argument('--embedding_mode', type=str, required=True,
                        choices=['genept', 'esm2', 'wikicrow', 'copath'],
                        help="Embedding type: genept (full coverage), esm2, wikicrow, or copath")
    parser.add_argument('--copath_file', type=str, default=None,
                        help='Path to gene_copath_neighbors.json (required for copath mode)')
    parser.add_argument('--coverage_filter', action='store_true',
                        help='Restrict all splits to perturbations covered by ALL 4 embedding types '
                             '(intersection of genept/esm2/wikicrow/copath). Use for sensitivity analysis '
                             'to ensure apples-to-apples comparison across embedding modes.')
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

    use_copath = args.embedding_mode == 'copath'
    if use_copath and args.copath_file is None:
        parser.error("--copath_file is required when --embedding_mode is 'copath'")
    if not use_copath and args.embedding_file is None:
        parser.error("--embedding_file is required for genept/esm2/wikicrow modes")

    # Drop set for the chosen embedding mode
    if args.embedding_mode == 'esm2':
        mode_drop_set   = PERTS_NO_ESM2_EMBEDDING
        mode_label      = "ESM2 embedding"
    elif args.embedding_mode == 'wikicrow':
        mode_drop_set   = PERTS_NO_WIKICROW_EMBEDDING
        mode_label      = "WikiCrow embedding"
    elif args.embedding_mode == 'copath':
        mode_drop_set   = PERTS_NO_COPATH_NEIGHBORS
        mode_label      = "co-path neighbors"
    else:  # genept
        mode_drop_set   = frozenset()
        mode_label      = "GenePT embedding"

    # For --coverage_filter: drop perts missing from ANY embedding type so all 4
    # modes are evaluated on the same perturbation set (apples-to-apples comparison).
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
    print("KNN Classifier — Cross-Cell-Line Transfer")
    print("="*70)
    print(f"\nEmbedding mode: {mode_label}")
    print(f"Train: {train_name}")
    print(f"Test: {args.test_dataset_name}")
    print(f"K: {args.k}   Weights: {args.weights}   Seed: {args.random_seed}")
    print(f"Coverage filter (intersection of all 4 types): {args.coverage_filter}")
    print(f"Output: {args.output_dir}")

    # Load source cell lines: train (60%) + test (20%) splits
    print(f"\n{'='*70}")
    print("Loading source datasets (train + test splits)")
    print(f"{'='*70}")
    train_dfs = []
    for fp, name in zip(args.train_de_doc_file, args.train_dataset_name):
        print(f"\n  [{name}]")
        train_dfs.append(load_de_doc_file(fp, splits=['train', 'test']))

    # Load held-out cell line: calibration and test splits separately
    print(f"\n{'='*70}")
    print(f"Loading target dataset: {args.test_dataset_name} (calibration + test splits)")
    print(f"{'='*70}")
    calib_df = load_de_doc_file(args.test_de_doc_file, splits=['calibration'])
    test_df  = load_de_doc_file(args.test_de_doc_file, splits=['test'])

    # TESTING PURPOSE ONLY
    if args.subsample is not None:
        train_dfs = [df[df['pert'].isin(sorted(df['pert'].unique())[:args.subsample])]
                     for df in train_dfs]
        calib_df = calib_df[calib_df['pert'].isin(sorted(calib_df['pert'].unique())[:args.subsample])]
        test_df  = test_df[test_df['pert'].isin(sorted(test_df['pert'].unique())[:args.subsample])]
        print(f"\nSubsampled to {args.subsample} perts per dataset.")

    # Drop perturbations without embedding / co-path neighbors for the chosen mode.
    # With --coverage_filter also drops perts missing from any of the 4 types.
    for split_name, df in [("train", pd.concat(train_dfs, ignore_index=True)),
                            ("calib", calib_df), ("test", test_df)]:
        dropped = set(df['pert'].unique()) & effective_drop_set
        if dropped:
            print(f"\nDropping {len(dropped)} {split_name} perturbations with no {mode_label}: "
                  f"{sorted(dropped)}")
    train_dfs = [df[~df['pert'].isin(effective_drop_set)] for df in train_dfs]
    calib_df  = calib_df[~calib_df['pert'].isin(effective_drop_set)]
    test_df   = test_df[~test_df['pert'].isin(effective_drop_set)]
    print(f"\nAfter filtering: train perts={sum(df['pert'].nunique() for df in train_dfs)}, "
          f"calib perts={calib_df['pert'].nunique()}, "
          f"test perts={test_df['pert'].nunique()}")

    train_df   = merge_train_datasets(train_dfs)
    eval_combined_df = pd.concat([calib_df, test_df], ignore_index=True)

    train_perts = set(train_df['pert'].unique())
    train_genes = set(train_df['gene'].unique())
    eval_genes  = set(eval_combined_df['gene'].unique())
    eval_perts  = set(eval_combined_df['pert'].unique())

    print(f"\n{'='*70}")
    print("Cross-Dataset Overlap Statistics")
    print(f"{'='*70}")
    print(f"  Train perts: {len(train_perts)}   Eval perts: {len(eval_perts)}   "
          f"Shared: {len(train_perts & eval_perts)}")
    print(f"  Train genes: {len(train_genes)}   Eval genes: {len(eval_genes)}   "
          f"Shared: {len(train_genes & eval_genes)}")
    unseen_genes = sorted(eval_genes - train_genes)
    print(f"  Eval-only genes (no training label, will default to class 0): "
          f"{len(unseen_genes)}")

    # Save unseen genes JSON
    unseen_info = {'unseen_genes': unseen_genes, 'n_unseen_genes': len(unseen_genes)}
    unseen_path = os.path.join(args.output_dir, f'{transfer_name}_transfer_unseen.json')
    with open(unseen_path, 'w') as f:
        json.dump(unseen_info, f, indent=2)
    print(f"  Unseen genes saved to {unseen_path}")

    if use_copath:
        print(f"\nLoading co-pathway neighbors from {args.copath_file}...")
        copath_neighbors = load_copath_neighbors(args.copath_file)
        embedding_dict = None
    else:
        copath_neighbors = None
        print(f"\nLoading {mode_label} embeddings from {args.embedding_file}...")
        primary_dict = load_embedding_from_pkl(args.embedding_file)
        if args.embedding_file2 is not None:
            print(f"Loading secondary embeddings from {args.embedding_file2}...")
            secondary_dict = load_embedding_from_pkl(args.embedding_file2)
            embedding_dict = {**primary_dict, **secondary_dict}
            print(f"  {len(primary_dict)} (primary) + {len(secondary_dict)} (secondary) "
                  f"= {len(embedding_dict)} unique entries")
        else:
            embedding_dict = primary_dict
            print(f"  Loaded {len(embedding_dict)} entries")

    plot_data_summary(train_df, eval_combined_df, transfer_name, output_dir=args.output_dir)

    train_pert_names = sorted(train_df['pert'].unique().tolist())

    def predict_split(df, split_label):
        print(f"\n{'='*70}")
        print(f"Predicting on {split_label} set (K={args.k}, expandable search)")
        print(f"  {len(df)} pairs, {df['pert'].nunique()} perts")
        print(f"{'='*70}")
        return predict_knn_batch(
            test_df=df,
            train_df=train_df,
            train_pert_names=train_pert_names,
            embedding_dict=embedding_dict,
            k=args.k,
            weights=args.weights,
            return_proba=True,
            verbose=True,
            copath_neighbors=copath_neighbors,
        )

    y_pred_calib, y_proba_calib, _ = predict_split(calib_df, "calibration")
    y_pred_test,  y_proba_test,  _ = predict_split(test_df,  "test")

    y_true_calib     = calib_df['label'].values
    y_true_test      = test_df['label'].values
    y_pred_combined  = np.concatenate([y_pred_calib, y_pred_test])
    y_proba_combined = np.concatenate([y_proba_calib, y_proba_test], axis=0)
    y_true_combined  = np.concatenate([y_true_calib, y_true_test])

    metrics_calib    = compute_3class_metrics(y_true_calib,    y_pred_calib,    y_proba_calib)
    metrics_test     = compute_3class_metrics(y_true_test,     y_pred_test,     y_proba_test)
    metrics_combined = compute_3class_metrics(y_true_combined, y_pred_combined, y_proba_combined)

    def print_metrics(metrics, header):
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
            if f'auroc_ovr_class_{cls}' in metrics:
                print(f"    AUROC OvR: {metrics[f'auroc_ovr_class_{cls}']:.4f}")

    print_metrics(metrics_calib,    f"RESULTS — Calibration Set (transfer: {train_name} → {transfer_name})")
    print_metrics(metrics_test,     f"RESULTS — Test Set (transfer: {train_name} → {transfer_name})")
    print_metrics(metrics_combined, f"RESULTS — Combined Calib + Test Set (transfer: {train_name} → {transfer_name})")

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

    # Save 3 metrics CSVs
    for split_label, metrics in [
        ("calib",    metrics_calib),
        ("test",     metrics_test),
        ("combined", metrics_combined),
    ]:
        metrics_df = pd.DataFrame([metrics])
        metrics_df['k']              = args.k
        metrics_df['train_dataset']  = train_name
        metrics_df['test_dataset']   = args.test_dataset_name
        metrics_df['n_train_perts']  = len(train_perts)
        metrics_df['n_eval_perts']   = len(eval_perts)
        metrics_df['n_shared_perts'] = len(train_perts & eval_perts)
        metrics_df['n_shared_genes'] = len(train_genes & eval_genes)
        metrics_path = os.path.join(
            args.output_dir, f'{transfer_name}_knn_transfer_metrics_{split_label}.csv'
        )
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Metrics ({split_label}) saved to {metrics_path}")

    # Save 3 predictions CSVs (with per-class probabilities)
    for split_label, df_split, y_pred, y_true, y_proba in [
        ("calib",    calib_df,         y_pred_calib,    y_true_calib,    y_proba_calib),
        ("test",     test_df,          y_pred_test,     y_true_test,     y_proba_test),
        ("combined", eval_combined_df, y_pred_combined, y_true_combined, y_proba_combined),
    ]:
        pred_df = df_split.copy()
        pred_df['prediction']   = y_pred
        pred_df['correct']      = (y_pred == y_true)
        pred_df['prob_class_0'] = y_proba[:, 0]
        pred_df['prob_class_1'] = y_proba[:, 1]
        pred_df['prob_class_2'] = y_proba[:, 2]
        pred_path = os.path.join(
            args.output_dir, f'{transfer_name}_knn_transfer_predictions_{split_label}.csv'
        )
        pred_df.to_csv(pred_path, index=False)
        print(f"Predictions ({split_label}) saved to {pred_path}")

    print(f"\n{'='*70}")
    print("ERROR ANALYSIS (Combined Calib + Test)")
    print(f"{'='*70}")
    combined_keys = list(zip(eval_combined_df['pert'], eval_combined_df['gene']))
    run_error_analysis_from_results(
        dataset=transfer_name,
        test_keys=combined_keys,
        predictions=y_pred_combined,
        true_labels=y_true_combined,
        adata=None,
        task_name="3class_KNN_transfer",
        class_names=['Non-DE', 'Up-regulated', 'Down-regulated'],
        output_dir=error_dir,
    )

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
