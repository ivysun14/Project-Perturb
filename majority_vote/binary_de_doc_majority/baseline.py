"""
baseline.py
===========

Gene-level majority vote baseline for PerturbQA tasks I.

This script evaluates a simple baseline that predicts the majority class
for each gene based on training data, following PerturbQA's evaluation protocol.
"""


import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from typing import Dict, Tuple, List
from collections import defaultdict
from scipy.stats import ranksums
from statsmodels.stats.multitest import multipletests

# Import perturbqa for data loading
from perturbqa import load_de, load_dir
from perturbqa import auc_per_gene as perturbqa_auc_per_gene

# Use local evaluation module with extended metrics
import sys
sys.path.insert(0, '..')
from perturbseq_lang.evaluation import (
    auc_per_gene,
    compute_perturbqa_metrics,
    run_perturbqa_evaluation
)


def resolve_repeats(adata, gene_row="gene_name"):
    """
    Take column-wise averages for repeatedly measured genes.

    Args:
        adata: an anndata of cell by gene matrix
        gene_row: adata.var row name representing genes measured
    
    Returns: 
        a new AnnData object with uniquely measured gene columns
    """
    
    genes = adata.var[gene_row]
    groups = {}
    for i, g in enumerate(genes):
        groups.setdefault(g, []).append(i)
    
    # take average of columns with repeated gene name
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    X_avg = np.stack([X[:, idxs].mean(axis=1) for idxs in groups.values()], axis=1)
    
    # build new var and new anndata
    var_new = adata.var.iloc[[idxs[0] for idxs in groups.values()]].copy()
    adata = ad.AnnData(
        X=X_avg,
        obs=adata.obs.copy(),
        var=var_new
    )

    return adata


def compute_de_per_gene(adata, 
                        control_key='non-targeting', 
                        target_col='target_gene',
                        gene_row='gene_name', 
                        alpha=0.05, 
                        save_path=None):
    """
    Compute differential expression labels for each (perturbation, gene) pair.
    Uses Wilcoxon signed-rank test with Benjamini-Hochberg correction (the same with PerturbQA).
    
    Args:
        adata: AnnData object
        control_key: Key indicating control cells
        target_col: Column with perturbation labels
        gene_row: Row with measured gene names
        alpha: Significance threshold (default: 0.05)
        save_path: Path to save DE results CSV (optional)
    
    Returns:
        dict: {(pert, gene): 0 or 1} indicating if gene is DE upon perturbation
    """
    
    de_labels = {}
    de_results = []  # Store for CSV output
    
    # Get control cells
    control_mask = adata.obs[target_col] == control_key
    control_expr = adata.X[control_mask]
    
    # Get unique perturbations (excluding control)
    perturbations = [p for p in adata.obs[target_col].unique() if p != control_key]
    
    print(f"\nComputing DE for {len(perturbations)} perturbations across {adata.n_vars} genes...")
    
    # For each perturbation
    for pert_idx, pert in enumerate(perturbations):
        
        if (pert_idx + 1) % 10 == 0:
            print(f"  Processing perturbation {pert_idx + 1}/{len(perturbations)}")
        
        pert_mask = adata.obs[target_col] == pert
        pert_expr = adata.X[pert_mask]
        
        # Collect p-values for all genes for this perturbation
        pvalues = []
        gene_names = []
        
        # For each gene subjected to this perturbation
        for gene_idx, gene_name in enumerate(adata.var[gene_row]):
            control_vals = control_expr[:, gene_idx]
            pert_vals = pert_expr[:, gene_idx]
            
            # Perform Wilcoxon rank-sum test (Mann-Whitney U test)
            # Note: Using ranksums (equivalent to Mann-Whitney U) which is appropriate
            # for comparing two independent samples
            try:
                statistic, pval = ranksums(pert_vals, control_vals)
                pvalues.append(pval)
                gene_names.append(gene_name)
            except:
                # If test fails (e.g., all values identical), assign p=1
                pvalues.append(1.0)
                gene_names.append(gene_name)
        
        # Apply Benjamini-Hochberg correction
        if len(pvalues) > 0:
            reject, pvals_corrected, _, _ = multipletests(pvalues, alpha=alpha, method='fdr_bh')
            
            # Assign DE labels and collect results
            for gene_name, is_significant, pval_corrected in zip(gene_names, reject, pvals_corrected):
                label = 1 if is_significant else 0
                de_labels[(pert, gene_name)] = label
                
                # Store for CSV
                de_results.append({
                    'pert': pert,
                    'gene': gene_name,
                    'label': label,
                    'p_value': pval_corrected
                })
    
    print(f"DE computation complete. {sum(de_labels.values())} DE pairs out of {len(de_labels)} total.")
    
    # Save to CSV if path provided
    if save_path:
        df = pd.DataFrame(de_results)
        df.to_csv(save_path, index=False)
        print(f"DE results saved to {save_path}")
    
    return de_labels


def compute_direction_per_gene(adata, 
                               control_key='non-targeting', 
                               target_col='target_gene',
                               gene_row='gene_name', 
                               save_path=None):
    """
    Compute direction of change (up/down) for each (perturbation, gene) pair.
    Compare control mean vs. perturbed mean.

    Args:
        adata: AnnData object
        control_key: Key indicating control cells
        target_col: Column with perturbation labels
        gene_row: Row with measured gene names
        save_path: Path to save direction results CSV (optional)
    
    Returns:
        dict: {(pert, gene): 1 (up) or 0 (down)}
    """
    
    dir_labels = {}
    dir_results = []  # Store for CSV output
    
    # Get control cells
    control_mask = adata.obs[target_col] == control_key
    control_expr = adata.X[control_mask]
    
    # Get unique perturbations (excluding control)
    perturbations = [p for p in adata.obs[target_col].unique() if p != control_key]
    
    print(f"\nComputing direction for {len(perturbations)} perturbations across {adata.n_vars} genes...")
    
    for pert_idx, pert in enumerate(perturbations):
        
        if (pert_idx + 1) % 10 == 0:
            print(f"  Processing perturbation {pert_idx + 1}/{len(perturbations)}")
        
        pert_mask = adata.obs[target_col] == pert
        pert_expr = adata.X[pert_mask]
        
        # For each gene, determine direction
        for gene_idx, gene_name in enumerate(adata.var[gene_row]):
            control_vals = control_expr[:, gene_idx]
            pert_vals = pert_expr[:, gene_idx]
            
            control_mean = np.mean(control_vals)
            pert_mean = np.mean(pert_vals)
            
            # Calculate mean change (log fold change since data is log-transformed)
            mean_change = pert_mean - control_mean
            
            # 1 = up-regulated, 0 = down-regulated
            direction = 1 if pert_mean > control_mean else 0
            
            dir_labels[(pert, gene_name)] = direction
            
            # Store for CSV
            dir_results.append({
                'pert': pert,
                'gene': gene_name,
                'label': direction,
                'mean_change': mean_change
            })
    
    print(f"Direction computation complete. {sum(dir_labels.values())} up-regulated pairs out of {len(dir_labels)} total.")
    
    # Save to CSV if path provided
    if save_path:
        df = pd.DataFrame(dir_results)
        df.to_csv(save_path, index=False)
        print(f"Direction results saved to {save_path}")
    
    return dir_labels


def compute_split_statistics(de_labels: Dict, 
                             doc_labels: Dict, 
                             train_keys: List[Tuple], 
                             test_keys: List[Tuple], 
                             split_name: str):
    """
    Compute statistics for train or test split (Table 5 in PerturbQA).
    
    Args:
        de_labels: Dict mapping (pert, gene) -> DE label (0 or 1)
        doc_labels: Dict mapping (pert, gene) -> direction label (0 or 1)
        train_keys: List of (pert, gene) tuples for train split
        test_keys: List of (pert, gene) tuples for test split
        split_name: "Train" or "Test"
    
    Returns:
        Dict with statistics
    """

    keys = train_keys if split_name == "Train" else test_keys
    
    # Count DE and non-DE
    de_count = sum([de_labels.get(key, 0) for key in keys])
    non_de_count = len(keys) - de_count
    
    # Count up and down among DE genes
    up_count = 0
    down_count = 0
    for key in keys:
        if de_labels.get(key, 0) == 1:  # Only count if DE
            if doc_labels.get(key, 0) == 1:  # Up
                up_count += 1
            else:  # Down
                down_count += 1
    
    return {
        'split': split_name,
        'total': len(keys),
        'non_de': non_de_count,
        'de_total': de_count,
        'up': up_count,
        'down': down_count
    }


def print_statistics_table(dataset_name: str, train_stats: Dict, test_stats: Dict):
    """
    Print statistics table in the format of PerturbQA Table 5.
    
    Args:
        dataset_name: Name of the dataset
        train_stats: Statistics for training split
        test_stats: Statistics for test split
    """

    print("\n" + "="*80)
    print(f"Data Split Statistics for {dataset_name.upper()}")
    print("="*80)
    print(f"{'Dataset':<12} {'Split':<8} {'Total':>10} {'non-DE':>10} {'DE Total':>10} {'Up':>10} {'Down':>10}")
    print("-"*80)
    
    # Train row
    print(f"{dataset_name.upper():<12} {train_stats['split']:<8} "
          f"{train_stats['total']:>10,} {train_stats['non_de']:>10,} "
          f"{train_stats['de_total']:>10,} {train_stats['up']:>10,} {train_stats['down']:>10,}")
    
    # Test row
    print(f"{'':12} {test_stats['split']:<8} "
          f"{test_stats['total']:>10,} {test_stats['non_de']:>10,} "
          f"{test_stats['de_total']:>10,} {test_stats['up']:>10,} {test_stats['down']:>10,}")
    print("="*80)


def train_gene_majority_vote(train_keys: List[Tuple], labels: Dict,
                             return_proportion: bool = True) -> Dict:
    """
    Train gene-level majority vote predictor.

    Following PerturbQA protocol: "Retrieval takes the mean label over (p',g')"
    This gives a probability-like score between 0 and 1 for proper AUROC computation.

    Args:
        train_keys: List of (perturbation, gene) tuples in training set
        labels: Dict mapping (pert, gene) -> label
        return_proportion: If True, return proportion (mean) of positive labels.
                          If False, return binary majority class (0 or 1).

    Returns:
        Dict mapping gene -> proportion (float 0-1) or majority class (0 or 1)
    """

    gene_stats = defaultdict(lambda: {'pos': 0, 'neg': 0, 'total': 0})

    for pert, gene in train_keys:
        key = (pert, gene)
        if key in labels:
            label = labels[key]
            gene_stats[gene]['total'] += 1
            if label == 1:
                gene_stats[gene]['pos'] += 1
            else:
                gene_stats[gene]['neg'] += 1

    gene_scores = {}
    for gene, stats in gene_stats.items():
        if return_proportion:
            # Return proportion/percentage of positive labels (for AUROC)
            gene_scores[gene] = stats['pos'] / stats['total'] if stats['total'] > 0 else 0.5
        else:
            # Return binary majority class
            gene_scores[gene] = 1 if stats['pos'] >= stats['neg'] else 0

    return gene_scores


def predict_from_gene_majority(test_keys: List[Tuple],
                               gene_scores: Dict,
                               default_pred: float = 0.5) -> np.ndarray:
    """
    Generate predictions based on gene-level scores (proportion or majority).

    Args:
        test_keys: List of (perturbation, gene) tuples in test set
        gene_scores: Dict mapping gene -> score (proportion or binary)
        default_pred: Default prediction for unseen genes

    Returns:
        Array of predictions (floats between 0 and 1)
    """
    predictions = []
    for pert, gene in test_keys:
        if gene in gene_scores:
            predictions.append(float(gene_scores[gene]))
        else:
            predictions.append(default_pred)

    return np.array(predictions)


def evaluate_majority_vote_baseline(
    adata,
    dataset_name: str,
    control_key: str = 'non-targeting',
    target_col: str = 'target_gene',
    gene_row: str = 'gene_name',
    alpha: float = 0.05,
    n_runs: int = 3
) -> Dict:
    """
    Evaluate gene-level majority vote baseline on PerturbQA tasks.
    
    Args:
        adata: AnnData object with cells as rows, genes as columns
        dataset_name: Name of dataset with pre-split train/test set in PerturbQA
        control_key: Value in obs[target_col] indicating control cells
        target_col: Column in obs containing perturbation labels
        gene_row: Row with measured gene names
        alpha: Significance threshold for DE gene
        n_runs: Number of runs for subsampling (to compute std)
        
    Returns:
        Dict with results for DE and DOC tasks
    """

    # Load PerturbQA train/test split for dir and de
    data_de = load_de(dataset_name)
    data_dir = load_dir(dataset_name)
    
    X_de_train = data_de["train"]
    X_de_test = data_de["test"]
    print(f"\nDE: Loaded {len(X_de_train)} training pairs, {len(X_de_test)} test pairs")

    X_dir_train = data_dir["train"]
    X_dir_test = data_dir["test"]
    print(f"Dir: Loaded {len(X_dir_train)} training pairs, {len(X_dir_test)} test pairs")

    train_de_keys = [(x["pert"], x["gene"]) for x in X_de_train]
    test_de_keys = [(x["pert"], x["gene"]) for x in X_de_test]
    train_dir_keys = [(x["pert"], x["gene"]) for x in X_dir_train]
    test_dir_keys = [(x["pert"], x["gene"]) for x in X_dir_test]
    
    # Compute ground truth labels
    de_labels = compute_de_per_gene(adata, control_key, target_col, gene_row, alpha,
                                    save_path=f"files/{dataset_name}_de_results.csv")
    doc_labels = compute_direction_per_gene(adata, control_key, target_col, gene_row,
                                            save_path=f"files/{dataset_name}_direction_results.csv")

    # Recreate Table 5 in PerturbQA
    print("\nComputing split statistics...")
    train_stats = compute_split_statistics(de_labels, doc_labels, train_de_keys, test_de_keys, "Train")
    test_stats = compute_split_statistics(de_labels, doc_labels, train_de_keys, test_de_keys, "Test")
    print_statistics_table(dataset_name, train_stats, test_stats)

    results = {
        'de_aurocs': [],
        'doc_aurocs': [],
        'de_aurocs_perturbqa': [],  # For comparison with perturbqa implementation
        'doc_aurocs_perturbqa': [],
        'de_metrics': [],  # Full metrics from local implementation
        'doc_metrics': []
    }

    # Run multiple times with subsampling
    for run in range(n_runs):

        print(f"\nRun {run + 1}/{n_runs}")

        # Subsample training data (80% of train_de_keys)
        np.random.seed(run)

        n_de_train = int(len(train_de_keys) * 0.8)
        train_de_subsample = np.random.choice(len(train_de_keys), n_de_train, replace=False)
        train_de_keys_sub = [train_de_keys[i] for i in train_de_subsample]

        n_dir_train = int(len(train_dir_keys) * 0.8)
        train_dir_subsample = np.random.choice(len(train_dir_keys), n_dir_train, replace=False)
        train_dir_keys_sub = [train_dir_keys[i] for i in train_dir_subsample]

        # Train gene-level majority vote for DE (using proportions for AUROC)
        print("  Training DE predictor...")
        gene_proportion_de = train_gene_majority_vote(train_de_keys_sub, de_labels, return_proportion=True)

        # Train gene-level majority vote for DOC (using proportions for AUROC)
        print("  Training DOC predictor...")
        gene_proportion_doc = train_gene_majority_vote(train_dir_keys_sub, doc_labels, return_proportion=True)

        # Predict on test set using PROPORTIONS (proper for AUROC)
        print("  Predicting on test set (using proportions)...")
        pred_de = predict_from_gene_majority(test_de_keys, gene_proportion_de)
        pred_doc = predict_from_gene_majority(test_dir_keys, gene_proportion_doc)

        # Get true labels for test set
        true_de = np.array([de_labels.get(key, 0) for key in test_de_keys])
        true_doc = np.array([doc_labels.get(key, 0) for key in test_dir_keys])

        # Calculate AUROC using local implementation (with extended metrics)
        print("  Calculating metrics (local implementation)...")
        de_metrics = compute_perturbqa_metrics(test_de_keys, pred_de, true_de)
        doc_metrics = compute_perturbqa_metrics(test_dir_keys, pred_doc, true_doc)

        auroc_de = de_metrics['auroc_per_gene']
        auroc_doc = doc_metrics['auroc_per_gene']

        # Also calculate using perturbqa implementation for comparison
        auroc_de_pqa = perturbqa_auc_per_gene(test_de_keys, pred_de, true_de)
        auroc_doc_pqa = perturbqa_auc_per_gene(test_dir_keys, pred_doc, true_doc)

        print(f"  DE AUROC (local):     {auroc_de:.4f}")
        print(f"  DE AUROC (perturbqa): {auroc_de_pqa:.4f}")
        print(f"  DOC AUROC (local):     {auroc_doc:.4f}")
        print(f"  DOC AUROC (perturbqa): {auroc_doc_pqa:.4f}")

        # Show additional metrics
        print(f"  DE Accuracy:   {de_metrics['accuracy']:.4f}  |  DOC Accuracy:   {doc_metrics['accuracy']:.4f}")
        print(f"  DE F1:         {de_metrics['f1']:.4f}  |  DOC F1:         {doc_metrics['f1']:.4f}")
        print(f"  DE Precision:  {de_metrics['precision']:.4f}  |  DOC Precision:  {doc_metrics['precision']:.4f}")
        print(f"  DE Recall:     {de_metrics['recall']:.4f}  |  DOC Recall:     {doc_metrics['recall']:.4f}")

        results['de_aurocs'].append(auroc_de)
        results['doc_aurocs'].append(auroc_doc)
        results['de_aurocs_perturbqa'].append(auroc_de_pqa)
        results['doc_aurocs_perturbqa'].append(auroc_doc_pqa)
        results['de_metrics'].append(de_metrics)
        results['doc_metrics'].append(doc_metrics)

    # Calculate mean and std
    de_mean = np.mean(results['de_aurocs'])
    de_std = np.std(results['de_aurocs'])
    doc_mean = np.mean(results['doc_aurocs'])
    doc_std = np.std(results['doc_aurocs'])

    de_mean_pqa = np.mean(results['de_aurocs_perturbqa'])
    de_std_pqa = np.std(results['de_aurocs_perturbqa'])
    doc_mean_pqa = np.mean(results['doc_aurocs_perturbqa'])
    doc_std_pqa = np.std(results['doc_aurocs_perturbqa'])

    # Aggregate other metrics
    de_acc_mean = np.mean([m['accuracy'] for m in results['de_metrics']])
    de_f1_mean = np.mean([m['f1'] for m in results['de_metrics']])
    doc_acc_mean = np.mean([m['accuracy'] for m in results['doc_metrics']])
    doc_f1_mean = np.mean([m['f1'] for m in results['doc_metrics']])

    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"{'Metric':<30} {'DE':>18} {'DOC':>18}")
    print("-"*70)
    print(f"{'AUROC (local, per-gene)':<30} {de_mean:.4f}±{de_std:.4f}   {doc_mean:.4f}±{doc_std:.4f}")
    print(f"{'AUROC (perturbqa, per-gene)':<30} {de_mean_pqa:.4f}±{de_std_pqa:.4f}   {doc_mean_pqa:.4f}±{doc_std_pqa:.4f}")
    print(f"{'Accuracy':<30} {de_acc_mean:.4f}            {doc_acc_mean:.4f}")
    print(f"{'F1 Score':<30} {de_f1_mean:.4f}            {doc_f1_mean:.4f}")
    print("="*70)

    # Check if implementations match
    if abs(de_mean - de_mean_pqa) > 0.001 or abs(doc_mean - doc_mean_pqa) > 0.001:
        print("\n⚠️  WARNING: Local and PerturbQA AUROC implementations differ!")
        print(f"   DE difference:  {abs(de_mean - de_mean_pqa):.6f}")
        print(f"   DOC difference: {abs(doc_mean - doc_mean_pqa):.6f}")
    else:
        print("\n✓ Local and PerturbQA AUROC implementations match.")

    return {
        'de_mean': de_mean,
        'de_std': de_std,
        'doc_mean': doc_mean,
        'doc_std': doc_std,
        'de_aurocs': results['de_aurocs'],
        'doc_aurocs': results['doc_aurocs'],
        'de_aurocs_perturbqa': results['de_aurocs_perturbqa'],
        'doc_aurocs_perturbqa': results['doc_aurocs_perturbqa'],
        'de_metrics': results['de_metrics'],
        'doc_metrics': results['doc_metrics'],
        'de_accuracy': de_acc_mean,
        'doc_accuracy': doc_acc_mean,
        'de_f1': de_f1_mean,
        'doc_f1': doc_f1_mean
    }


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Evaluate majority vote baseline on PerturbQA')
    parser.add_argument('--adata', type=str, required=True, help='Path to AnnData h5ad file')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of AnnData h5ad file to use in PerturbQA')
    parser.add_argument('--control', type=str, default='non-targeting', help='Control perturbation label')
    parser.add_argument('--target_col', type=str, default='target_gene', help='Column with perturbation labels')
    parser.add_argument('--gene_row', type=str, default='gene_name', help='Row with measured genes')
    parser.add_argument('--alpha', type=float, default=0.05, help='Significance threshold for DE test')
    parser.add_argument('--n_runs', type=int, default=3, help='Number of runs for subsampling')
    parser.add_argument("--test_size", type=int, default=None, help="Test size (only use for debugging)")
    
    args = parser.parse_args()
    
    # Load AnnData
    print(f"Loading AnnData from {args.adata}...")
    adata = ad.read_h5ad(args.adata)
    
    # Subset to a small number of cells for debugging
    if args.test_size:
        adata = adata[:args.test_size].copy()
    
    print(f"Loaded: {adata.shape[0]} cells x {adata.shape[1]} genes")
    
    # Minor preprocessing
    adata.obs.rename(columns={"gene": "target_gene"}, inplace=True)
    adata = resolve_repeats(adata, gene_row="gene_name")

    # Normalize to match PerturbQA protocol
    sc.pp.normalize_total(adata, target_sum=1e4, key_added="size_factors")
    sc.pp.log1p(adata)
    print("Log TP10k normalization completed (matching PerturbQA).")
    
    # Run evaluation
    results = evaluate_majority_vote_baseline(
        adata,
        dataset_name=args.dataset_name,
        control_key=args.control,
        target_col=args.target_col,
        gene_row=args.gene_row,
        alpha=args.alpha,
        n_runs=args.n_runs
    )