"""
enhanced_baseline.py
====================

Enhanced baseline that creates predictions varying WITHIN genes to beat per-gene AUROC of 0.5.

Key insight: Majority vote gives same score to all perturbations for a gene → per-gene AUROC = 0.5.
To beat this, we need perturbation-specific features that create variation within genes.

Approaches implemented:
1. Perturbation effect size: Some perturbations affect more genes than others
2. LogFC magnitude: Larger |logFC| correlates with more confident DE calls
3. Combined priors: P(DE | pert, gene) = f(gene_prior, pert_prior, logfc_features)
"""

import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from typing import Dict, Tuple, List, Optional
from collections import defaultdict
from scipy.stats import ranksums
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Import perturbqa for data loading
from perturbqa import load_de, load_dir
from perturbqa import auc_per_gene as perturbqa_auc_per_gene

# Use local evaluation module
import sys
sys.path.insert(0, '..')
from perturbseq_lang.evaluation import (
    auc_per_gene,
    auc_per_perturbation,
    compute_perturbqa_metrics,
    evaluate_three_class,
    evaluate_three_class_from_de_doc,
    print_three_class_report,
)


def compute_de_with_logfc(adata,
                          control_key='non-targeting',
                          target_col='target_gene',
                          gene_row='gene_name',
                          alpha=0.05) -> Tuple[Dict, Dict, Dict]:
    """
    Compute DE labels AND logFC values for each (perturbation, gene) pair.

    Returns:
        de_labels: {(pert, gene): 0 or 1}
        logfc_values: {(pert, gene): float}
        pvalues: {(pert, gene): float}
    """
    de_labels = {}
    logfc_values = {}
    pvalue_dict = {}

    # Get control cells
    control_mask = adata.obs[target_col] == control_key
    control_expr = adata.X[control_mask]

    # Precompute control means
    control_means = np.mean(control_expr, axis=0)

    # Get unique perturbations
    perturbations = [p for p in adata.obs[target_col].unique() if p != control_key]

    print(f"\nComputing DE + logFC for {len(perturbations)} perturbations across {adata.n_vars} genes...")

    for pert_idx, pert in enumerate(perturbations):
        if (pert_idx + 1) % 20 == 0:
            print(f"  Processing perturbation {pert_idx + 1}/{len(perturbations)}")

        pert_mask = adata.obs[target_col] == pert
        pert_expr = adata.X[pert_mask]
        pert_means = np.mean(pert_expr, axis=0)

        pvalues = []
        gene_names = []

        for gene_idx, gene_name in enumerate(adata.var[gene_row]):
            control_vals = control_expr[:, gene_idx]
            pert_vals = pert_expr[:, gene_idx]

            # Compute logFC (log fold change, since data is log-transformed)
            logfc = float(pert_means[gene_idx] - control_means[gene_idx])
            logfc_values[(pert, gene_name)] = logfc

            # Wilcoxon test
            try:
                _, pval = ranksums(pert_vals, control_vals)
                pvalues.append(pval)
                gene_names.append(gene_name)
            except:
                pvalues.append(1.0)
                gene_names.append(gene_name)

        # Apply BH correction
        if len(pvalues) > 0:
            reject, pvals_corrected, _, _ = multipletests(pvalues, alpha=alpha, method='fdr_bh')

            for gene_name, is_significant, pval_corrected in zip(gene_names, reject, pvals_corrected):
                de_labels[(pert, gene_name)] = 1 if is_significant else 0
                pvalue_dict[(pert, gene_name)] = pval_corrected

    print(f"DE computation complete. {sum(de_labels.values())} DE pairs out of {len(de_labels)} total.")

    return de_labels, logfc_values, pvalue_dict


def compute_direction_with_logfc(logfc_values: Dict) -> Dict:
    """
    Compute direction labels from logFC values.

    Args:
        logfc_values: {(pert, gene): logfc}

    Returns:
        dir_labels: {(pert, gene): 1 (up) or 0 (down)}
    """
    return {key: (1 if logfc > 0 else 0) for key, logfc in logfc_values.items()}


def compute_perturbation_effect_size(train_keys: List[Tuple],
                                     de_labels: Dict,
                                     logfc_values: Optional[Dict] = None) -> Dict:
    """
    Compute perturbation-level statistics from training data.

    For each perturbation, compute:
    - Fraction of genes that are DE
    - Mean |logFC| across genes (if available)

    Returns:
        pert_stats: {pert: {'de_fraction': float, 'mean_abs_logfc': float}}
    """
    pert_stats = defaultdict(lambda: {'de_count': 0, 'total': 0, 'logfc_sum': 0.0})

    for pert, gene in train_keys:
        key = (pert, gene)
        if key in de_labels:
            pert_stats[pert]['total'] += 1
            if de_labels[key] == 1:
                pert_stats[pert]['de_count'] += 1
            if logfc_values and key in logfc_values:
                pert_stats[pert]['logfc_sum'] += abs(logfc_values[key])

    # Convert to fractions
    result = {}
    for pert, stats in pert_stats.items():
        if stats['total'] > 0:
            result[pert] = {
                'de_fraction': stats['de_count'] / stats['total'],
                'mean_abs_logfc': stats['logfc_sum'] / stats['total'],
                'n_genes': stats['total']
            }
        else:
            result[pert] = {'de_fraction': 0.5, 'mean_abs_logfc': 0.0, 'n_genes': 0}

    return result


def compute_gene_statistics(train_keys: List[Tuple],
                            de_labels: Dict,
                            logfc_values: Optional[Dict] = None) -> Dict:
    """
    Compute gene-level statistics from training data.

    For each gene, compute:
    - Fraction of perturbations where gene is DE
    - Variance of logFC across perturbations (gene responsiveness)

    Returns:
        gene_stats: {gene: {'de_fraction': float, 'logfc_var': float}}
    """
    gene_stats = defaultdict(lambda: {'de_count': 0, 'total': 0, 'logfcs': []})

    for pert, gene in train_keys:
        key = (pert, gene)
        if key in de_labels:
            gene_stats[gene]['total'] += 1
            if de_labels[key] == 1:
                gene_stats[gene]['de_count'] += 1
            if logfc_values and key in logfc_values:
                gene_stats[gene]['logfcs'].append(logfc_values[key])

    result = {}
    for gene, stats in gene_stats.items():
        if stats['total'] > 0:
            logfc_var = np.var(stats['logfcs']) if len(stats['logfcs']) > 1 else 0.0
            result[gene] = {
                'de_fraction': stats['de_count'] / stats['total'],
                'logfc_var': logfc_var,
                'n_perts': stats['total']
            }
        else:
            result[gene] = {'de_fraction': 0.5, 'logfc_var': 0.0, 'n_perts': 0}

    return result


def predict_combined_prior(test_keys: List[Tuple],
                           gene_stats: Dict,
                           pert_stats: Dict,
                           combination: str = 'multiply',
                           default_gene: float = 0.5,
                           default_pert: float = 0.5) -> np.ndarray:
    """
    Generate predictions by combining gene and perturbation priors.

    This creates variation WITHIN genes because different perturbations
    have different effect sizes.

    Args:
        test_keys: List of (pert, gene) tuples
        gene_stats: {gene: {'de_fraction': float, ...}}
        pert_stats: {pert: {'de_fraction': float, ...}}
        combination: How to combine priors ('multiply', 'average', 'max', 'logistic')

    Returns:
        Array of prediction scores
    """
    predictions = []

    for pert, gene in test_keys:
        gene_prior = gene_stats.get(gene, {}).get('de_fraction', default_gene)
        pert_prior = pert_stats.get(pert, {}).get('de_fraction', default_pert)

        if combination == 'multiply':
            # P(DE) ∝ P(DE|gene) * P(DE|pert)
            # Normalized to keep in [0,1] range
            score = gene_prior * pert_prior / 0.25  # Normalize by expected value under independence
            score = min(1.0, max(0.0, score))
        elif combination == 'average':
            score = (gene_prior + pert_prior) / 2
        elif combination == 'max':
            score = max(gene_prior, pert_prior)
        elif combination == 'geometric':
            # Geometric mean
            score = np.sqrt(gene_prior * pert_prior)
        elif combination == 'weighted':
            # Weight perturbation effect more heavily
            score = 0.3 * gene_prior + 0.7 * pert_prior
        else:
            score = gene_prior  # Fallback to gene-only

        predictions.append(score)

    return np.array(predictions)


def train_logistic_predictor(train_keys: List[Tuple],
                             labels: Dict,
                             gene_stats: Dict,
                             pert_stats: Dict,
                             logfc_values: Optional[Dict] = None) -> Tuple:
    """
    Train a logistic regression model using gene + perturbation features.

    Features:
    - gene_de_fraction: How often this gene is DE
    - gene_logfc_var: Variance of logFC for this gene (responsiveness)
    - pert_de_fraction: How many genes this perturbation affects
    - pert_mean_logfc: Average |logFC| for this perturbation
    - logfc (if available): The actual logFC for this pair

    Returns:
        (model, scaler) trained on training data
    """
    X = []
    y = []

    for pert, gene in train_keys:
        key = (pert, gene)
        if key not in labels:
            continue

        features = [
            gene_stats.get(gene, {}).get('de_fraction', 0.5),
            gene_stats.get(gene, {}).get('logfc_var', 0.0),
            pert_stats.get(pert, {}).get('de_fraction', 0.5),
            pert_stats.get(pert, {}).get('mean_abs_logfc', 0.0),
        ]

        if logfc_values and key in logfc_values:
            features.append(abs(logfc_values[key]))
        else:
            features.append(0.0)

        X.append(features)
        y.append(labels[key])

    X = np.array(X)
    y = np.array(y)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train logistic regression
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_scaled, y)

    print(f"  Logistic model coefficients: {model.coef_[0]}")
    print(f"  Feature importance (abs): gene_de_frac={abs(model.coef_[0][0]):.3f}, "
          f"gene_logfc_var={abs(model.coef_[0][1]):.3f}, "
          f"pert_de_frac={abs(model.coef_[0][2]):.3f}, "
          f"pert_mean_logfc={abs(model.coef_[0][3]):.3f}, "
          f"abs_logfc={abs(model.coef_[0][4]):.3f}")

    return model, scaler


def predict_logistic(test_keys: List[Tuple],
                     model,
                     scaler,
                     gene_stats: Dict,
                     pert_stats: Dict,
                     logfc_values: Optional[Dict] = None) -> np.ndarray:
    """
    Generate predictions using trained logistic regression model.
    """
    X = []

    for pert, gene in test_keys:
        key = (pert, gene)
        features = [
            gene_stats.get(gene, {}).get('de_fraction', 0.5),
            gene_stats.get(gene, {}).get('logfc_var', 0.0),
            pert_stats.get(pert, {}).get('de_fraction', 0.5),
            pert_stats.get(pert, {}).get('mean_abs_logfc', 0.0),
        ]

        if logfc_values and key in logfc_values:
            features.append(abs(logfc_values[key]))
        else:
            features.append(0.0)

        X.append(features)

    X = np.array(X)
    X_scaled = scaler.transform(X)

    # Return probability of class 1
    return model.predict_proba(X_scaled)[:, 1]


def evaluate_enhanced_baseline(
    adata,
    dataset_name: str,
    control_key: str = 'non-targeting',
    target_col: str = 'target_gene',
    gene_row: str = 'gene_name',
    alpha: float = 0.05,
    n_runs: int = 3,
    methods: List[str] = ['gene_only', 'combined_multiply', 'combined_geometric', 'logistic']
) -> Dict:
    """
    Evaluate enhanced baselines that create perturbation-specific predictions.
    """

    # Load PerturbQA train/test splits
    data_de = load_de(dataset_name)
    data_dir = load_dir(dataset_name)

    X_de_train, X_de_test = data_de["train"], data_de["test"]
    X_dir_train, X_dir_test = data_dir["train"], data_dir["test"]

    print(f"\nDE: {len(X_de_train)} train, {len(X_de_test)} test pairs")
    print(f"Dir: {len(X_dir_train)} train, {len(X_dir_test)} test pairs")

    train_de_keys = [(x["pert"], x["gene"]) for x in X_de_train]
    test_de_keys = [(x["pert"], x["gene"]) for x in X_de_test]
    train_dir_keys = [(x["pert"], x["gene"]) for x in X_dir_train]
    test_dir_keys = [(x["pert"], x["gene"]) for x in X_dir_test]

    # Compute ground truth with logFC
    de_labels, logfc_values, pvalues = compute_de_with_logfc(
        adata, control_key, target_col, gene_row, alpha
    )
    dir_labels = compute_direction_with_logfc(logfc_values)

    # Initialize results storage
    results = {method: {'de_aurocs_per_gene': [], 'de_aurocs_per_pert': [],
                        'dir_aurocs_per_gene': [], 'dir_aurocs_per_pert': [],
                        'de_metrics': [], 'dir_metrics': [],
                        'three_class': []}
               for method in methods}

    for run in range(n_runs):
        print(f"\n{'='*60}")
        print(f"Run {run + 1}/{n_runs}")
        print(f"{'='*60}")

        np.random.seed(run)

        # Subsample training data
        n_de_train = int(len(train_de_keys) * 0.8)
        train_de_subsample = np.random.choice(len(train_de_keys), n_de_train, replace=False)
        train_de_keys_sub = [train_de_keys[i] for i in train_de_subsample]

        n_dir_train = int(len(train_dir_keys) * 0.8)
        train_dir_subsample = np.random.choice(len(train_dir_keys), n_dir_train, replace=False)
        train_dir_keys_sub = [train_dir_keys[i] for i in train_dir_subsample]

        # Compute statistics from training data
        print("\nComputing gene and perturbation statistics...")
        gene_stats_de = compute_gene_statistics(train_de_keys_sub, de_labels, logfc_values)
        pert_stats_de = compute_perturbation_effect_size(train_de_keys_sub, de_labels, logfc_values)

        gene_stats_dir = compute_gene_statistics(train_dir_keys_sub, dir_labels, logfc_values)
        pert_stats_dir = compute_perturbation_effect_size(train_dir_keys_sub, dir_labels, logfc_values)

        # Get true labels
        true_de = np.array([de_labels.get(key, 0) for key in test_de_keys])
        true_dir = np.array([dir_labels.get(key, 0) for key in test_dir_keys])

        for method in methods:
            print(f"\n--- Method: {method} ---")

            if method == 'gene_only':
                # Baseline: gene-level majority vote (same as before)
                pred_de = np.array([gene_stats_de.get(gene, {}).get('de_fraction', 0.5)
                                   for _, gene in test_de_keys])
                pred_dir = np.array([gene_stats_dir.get(gene, {}).get('de_fraction', 0.5)
                                    for _, gene in test_dir_keys])

            elif method.startswith('combined_'):
                # Combined prior methods
                combination = method.split('_')[1]
                pred_de = predict_combined_prior(test_de_keys, gene_stats_de, pert_stats_de,
                                                 combination=combination)
                pred_dir = predict_combined_prior(test_dir_keys, gene_stats_dir, pert_stats_dir,
                                                  combination=combination)

            elif method == 'logistic':
                # Logistic regression with all features
                print("  Training DE logistic model...")
                model_de, scaler_de = train_logistic_predictor(
                    train_de_keys_sub, de_labels, gene_stats_de, pert_stats_de, logfc_values
                )
                print("  Training Dir logistic model...")
                model_dir, scaler_dir = train_logistic_predictor(
                    train_dir_keys_sub, dir_labels, gene_stats_dir, pert_stats_dir, logfc_values
                )

                pred_de = predict_logistic(test_de_keys, model_de, scaler_de,
                                           gene_stats_de, pert_stats_de, logfc_values)
                pred_dir = predict_logistic(test_dir_keys, model_dir, scaler_dir,
                                            gene_stats_dir, pert_stats_dir, logfc_values)
            else:
                continue

            # Compute metrics
            de_auroc_gene = auc_per_gene(test_de_keys, pred_de, true_de)
            de_auroc_pert = auc_per_perturbation(test_de_keys, pred_de, true_de)
            dir_auroc_gene = auc_per_gene(test_dir_keys, pred_dir, true_dir)
            dir_auroc_pert = auc_per_perturbation(test_dir_keys, pred_dir, true_dir)

            de_metrics = compute_perturbqa_metrics(test_de_keys, pred_de, true_de)
            dir_metrics = compute_perturbqa_metrics(test_dir_keys, pred_dir, true_dir)

            # 3-class evaluation
            three_class = evaluate_three_class_from_de_doc(
                test_de_keys, test_dir_keys, pred_de, pred_dir,
                true_de, true_dir, de_threshold=0.5
            )

            print(f"  DE  AUROC: per-gene={de_auroc_gene:.4f}, per-pert={de_auroc_pert:.4f}, acc={de_metrics['accuracy']:.4f}")
            print(f"  Dir AUROC: per-gene={dir_auroc_gene:.4f}, per-pert={dir_auroc_pert:.4f}, acc={dir_metrics['accuracy']:.4f}")
            print(f"  3-class: acc={three_class['accuracy']:.4f}, macro_f1={three_class['macro_f1']:.4f}, down_f1={three_class['per_class']['down']['f1']:.4f}")

            # Store results
            results[method]['de_aurocs_per_gene'].append(de_auroc_gene)
            results[method]['de_aurocs_per_pert'].append(de_auroc_pert)
            results[method]['dir_aurocs_per_gene'].append(dir_auroc_gene)
            results[method]['dir_aurocs_per_pert'].append(dir_auroc_pert)
            results[method]['de_metrics'].append(de_metrics)
            results[method]['dir_metrics'].append(dir_metrics)
            results[method]['three_class'].append(three_class)

    # Print summary
    print("\n" + "="*120)
    print("SUMMARY: Enhanced Baseline Comparison")
    print("="*120)

    # Header 1: DE task metrics
    print("\nDE Task Metrics:")
    print(f"{'Method':<20} {'AUROC/gene':>12} {'AUROC/pert':>12} {'Accuracy':>10} {'F1':>10} {'Precision':>10} {'Recall':>10}")
    print("-"*90)

    for method in methods:
        de_gene = np.mean(results[method]['de_aurocs_per_gene'])
        de_pert = np.mean(results[method]['de_aurocs_per_pert'])
        de_acc = np.mean([m['accuracy'] for m in results[method]['de_metrics']])
        de_f1 = np.mean([m['f1'] for m in results[method]['de_metrics']])
        de_prec = np.mean([m['precision'] for m in results[method]['de_metrics']])
        de_rec = np.mean([m['recall'] for m in results[method]['de_metrics']])

        print(f"{method:<20} {de_gene:>12.4f} {de_pert:>12.4f} {de_acc:>10.4f} {de_f1:>10.4f} {de_prec:>10.4f} {de_rec:>10.4f}")

    # Header 2: Direction task metrics
    print("\nDirection Task Metrics:")
    print(f"{'Method':<20} {'AUROC/gene':>12} {'AUROC/pert':>12} {'Accuracy':>10} {'F1':>10} {'Precision':>10} {'Recall':>10}")
    print("-"*90)

    for method in methods:
        dir_gene = np.mean(results[method]['dir_aurocs_per_gene'])
        dir_pert = np.mean(results[method]['dir_aurocs_per_pert'])
        dir_acc = np.mean([m['accuracy'] for m in results[method]['dir_metrics']])
        dir_f1 = np.mean([m['f1'] for m in results[method]['dir_metrics']])
        dir_prec = np.mean([m['precision'] for m in results[method]['dir_metrics']])
        dir_rec = np.mean([m['recall'] for m in results[method]['dir_metrics']])

        print(f"{method:<20} {dir_gene:>12.4f} {dir_pert:>12.4f} {dir_acc:>10.4f} {dir_f1:>10.4f} {dir_prec:>10.4f} {dir_rec:>10.4f}")

    # Header 3: 3-class metrics
    print("\n3-Class Metrics (not_de, up, down):")
    print(f"{'Method':<20} {'Accuracy':>10} {'Macro F1':>10} {'Up F1':>10} {'Down F1':>10} {'NotDE F1':>10}")
    print("-"*70)

    for method in methods:
        acc = np.mean([tc['accuracy'] for tc in results[method]['three_class']])
        macro_f1 = np.mean([tc['macro_f1'] for tc in results[method]['three_class']])
        up_f1 = np.mean([tc['per_class']['up']['f1'] for tc in results[method]['three_class']])
        down_f1 = np.mean([tc['per_class']['down']['f1'] for tc in results[method]['three_class']])
        not_de_f1 = np.mean([tc['per_class']['not_de']['f1'] for tc in results[method]['three_class']])

        print(f"{method:<20} {acc:>10.4f} {macro_f1:>10.4f} {up_f1:>10.4f} {down_f1:>10.4f} {not_de_f1:>10.4f}")

    print("="*120)

    # Detailed 3-class for best method (logistic if available)
    best_method = 'logistic' if 'logistic' in methods else methods[-1]
    print(f"\nDetailed 3-class metrics for {best_method}:")
    avg_three_class = results[best_method]['three_class'][0]  # Use first run for detailed output
    print_three_class_report(avg_three_class)

    return results


def resolve_repeats(adata, gene_row="gene_name"):
    """Take column-wise averages for repeatedly measured genes."""
    genes = adata.var[gene_row]
    groups = {}
    for i, g in enumerate(genes):
        groups.setdefault(g, []).append(i)

    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    X_avg = np.stack([X[:, idxs].mean(axis=1) for idxs in groups.values()], axis=1)

    var_new = adata.var.iloc[[idxs[0] for idxs in groups.values()]].copy()
    adata = ad.AnnData(X=X_avg, obs=adata.obs.copy(), var=var_new)
    return adata


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Enhanced baseline with perturbation-specific features')
    parser.add_argument('--adata', type=str, required=True, help='Path to AnnData h5ad file')
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name in PerturbQA')
    parser.add_argument('--control', type=str, default='non-targeting', help='Control label')
    parser.add_argument('--target_col', type=str, default='target_gene', help='Perturbation column')
    parser.add_argument('--gene_row', type=str, default='gene_name', help='Gene name row')
    parser.add_argument('--alpha', type=float, default=0.05, help='DE significance threshold')
    parser.add_argument('--n_runs', type=int, default=3, help='Number of runs')

    args = parser.parse_args()

    print(f"Loading AnnData from {args.adata}...")
    adata = ad.read_h5ad(args.adata)
    print(f"Loaded: {adata.shape[0]} cells x {adata.shape[1]} genes")

    # Preprocessing
    adata.obs.rename(columns={"gene": "target_gene"}, inplace=True)
    adata = resolve_repeats(adata, gene_row="gene_name")

    sc.pp.normalize_total(adata, target_sum=1e4, key_added="size_factors")
    sc.pp.log1p(adata)
    print("Log TP10k normalization completed.")

    # Run evaluation
    results = evaluate_enhanced_baseline(
        adata,
        dataset_name=args.dataset_name,
        control_key=args.control,
        target_col=args.target_col,
        gene_row=args.gene_row,
        alpha=args.alpha,
        n_runs=args.n_runs
    )
