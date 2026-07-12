"""
test_enhanced_methods.py
========================

Test enhanced prediction methods using just PerturbQA labels (no h5ad needed).
This tests whether combining gene and perturbation priors beats per-gene AUROC of 0.5.

Key insight:
- Gene-only prior: same score for all perturbations → per-gene AUROC = 0.5
- Combined prior (gene + pert): different scores per perturbation → may beat 0.5
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple

# Import perturbqa
from perturbqa import load_de, load_dir
from perturbqa import auc_per_gene as perturbqa_auc_per_gene

# Local evaluation
import sys
sys.path.insert(0, '..')
from perturbseq_lang.evaluation import (
    auc_per_gene,
    auc_per_perturbation,
    compute_perturbqa_metrics,
    evaluate_three_class_from_de_doc,
)


def compute_gene_prior(train_data: List[Dict]) -> Dict[str, float]:
    """Compute P(label=1 | gene) from training data."""
    gene_stats = defaultdict(lambda: {'pos': 0, 'total': 0})
    for item in train_data:
        gene = item['gene']
        gene_stats[gene]['total'] += 1
        if item['label'] == 1:
            gene_stats[gene]['pos'] += 1

    return {gene: stats['pos'] / stats['total'] if stats['total'] > 0 else 0.5
            for gene, stats in gene_stats.items()}


def compute_pert_prior(train_data: List[Dict]) -> Dict[str, float]:
    """Compute P(label=1 | perturbation) from training data."""
    pert_stats = defaultdict(lambda: {'pos': 0, 'total': 0})
    for item in train_data:
        pert = item['pert']
        pert_stats[pert]['total'] += 1
        if item['label'] == 1:
            pert_stats[pert]['pos'] += 1

    return {pert: stats['pos'] / stats['total'] if stats['total'] > 0 else 0.5
            for pert, stats in pert_stats.items()}


def predict_gene_only(test_data: List[Dict], gene_prior: Dict[str, float]) -> np.ndarray:
    """Predict using only gene prior."""
    return np.array([gene_prior.get(item['gene'], 0.5) for item in test_data])


def predict_pert_only(test_data: List[Dict], pert_prior: Dict[str, float]) -> np.ndarray:
    """Predict using only perturbation prior."""
    return np.array([pert_prior.get(item['pert'], 0.5) for item in test_data])


def predict_combined(test_data: List[Dict],
                     gene_prior: Dict[str, float],
                     pert_prior: Dict[str, float],
                     method: str = 'multiply') -> np.ndarray:
    """Predict using combined gene and perturbation priors."""
    predictions = []
    for item in test_data:
        g = gene_prior.get(item['gene'], 0.5)
        p = pert_prior.get(item['pert'], 0.5)

        if method == 'multiply':
            # Normalize by expected value under independence
            score = min(1.0, g * p / 0.25)
        elif method == 'geometric':
            score = np.sqrt(g * p)
        elif method == 'average':
            score = (g + p) / 2
        elif method == 'max':
            score = max(g, p)
        elif method == 'weighted_pert':
            # Weight perturbation more heavily
            score = 0.3 * g + 0.7 * p
        elif method == 'weighted_gene':
            # Weight gene more heavily
            score = 0.7 * g + 0.3 * p
        else:
            score = g

        predictions.append(score)

    return np.array(predictions)


def evaluate_method(test_data: List[Dict], predictions: np.ndarray, method_name: str):
    """Evaluate a prediction method."""
    keys = [(item['pert'], item['gene']) for item in test_data]
    true = np.array([item['label'] for item in test_data])

    auroc_gene = auc_per_gene(keys, predictions, true)
    auroc_pert = auc_per_perturbation(keys, predictions, true)
    metrics = compute_perturbqa_metrics(keys, predictions, true)

    return {
        'method': method_name,
        'auroc_per_gene': auroc_gene,
        'auroc_per_pert': auroc_pert,
        'accuracy': metrics['accuracy'],
        'f1': metrics['f1'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
    }


def run_experiment(dataset_name: str, task: str = 'de'):
    """Run experiment for a dataset and task (de or dir)."""
    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_name.upper()}, Task: {task.upper()}")
    print(f"{'='*80}")

    # Load data
    if task == 'de':
        data = load_de(dataset_name)
    else:
        data = load_dir(dataset_name)

    train_data, test_data = data['train'], data['test']
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    # Compute priors from training data
    gene_prior = compute_gene_prior(train_data)
    pert_prior = compute_pert_prior(train_data)

    print(f"Unique genes in train: {len(gene_prior)}")
    print(f"Unique perturbations in train: {len(pert_prior)}")

    # Methods to test
    methods = ['gene_only', 'pert_only', 'multiply', 'geometric', 'average', 'max', 'weighted_pert', 'weighted_gene']
    results = []

    for method in methods:
        if method == 'gene_only':
            pred = predict_gene_only(test_data, gene_prior)
        elif method == 'pert_only':
            pred = predict_pert_only(test_data, pert_prior)
        else:
            pred = predict_combined(test_data, gene_prior, pert_prior, method)

        result = evaluate_method(test_data, pred, method)
        results.append(result)

    # Print results table
    print(f"\n{'Method':<18} {'AUROC/gene':>12} {'AUROC/pert':>12} {'Accuracy':>10} {'F1':>10} {'Precision':>10} {'Recall':>10}")
    print("-"*90)

    for r in results:
        print(f"{r['method']:<18} {r['auroc_per_gene']:>12.4f} {r['auroc_per_pert']:>12.4f} "
              f"{r['accuracy']:>10.4f} {r['f1']:>10.4f} {r['precision']:>10.4f} {r['recall']:>10.4f}")

    # Highlight key findings
    gene_only = [r for r in results if r['method'] == 'gene_only'][0]
    best_combined = max([r for r in results if r['method'] not in ['gene_only', 'pert_only']],
                        key=lambda x: x['auroc_per_gene'])

    print(f"\nKey Finding:")
    print(f"  Gene-only per-gene AUROC: {gene_only['auroc_per_gene']:.4f}")
    print(f"  Best combined per-gene AUROC: {best_combined['auroc_per_gene']:.4f} ({best_combined['method']})")

    if best_combined['auroc_per_gene'] > gene_only['auroc_per_gene'] + 0.01:
        print(f"  ✓ Combined method beats gene-only by {best_combined['auroc_per_gene'] - gene_only['auroc_per_gene']:.4f}")
    else:
        print(f"  ✗ Combined method does NOT significantly beat gene-only")

    return results


def run_3class_experiment(dataset_name: str):
    """Run 3-class evaluation."""
    print(f"\n{'='*80}")
    print(f"3-Class Evaluation: {dataset_name.upper()}")
    print(f"{'='*80}")

    data_de = load_de(dataset_name)
    data_dir = load_dir(dataset_name)

    train_de, test_de = data_de['train'], data_de['test']
    train_dir, test_dir = data_dir['train'], data_dir['test']

    # Build test key sets
    test_de_keys = [(item['pert'], item['gene']) for item in test_de]
    test_dir_keys = [(item['pert'], item['gene']) for item in test_dir]

    # Find common keys for 3-class eval
    de_key_set = set(test_de_keys)
    dir_key_set = set(test_dir_keys)
    common_keys = de_key_set & dir_key_set
    print(f"Common keys for 3-class: {len(common_keys)}")

    # Compute priors
    gene_prior_de = compute_gene_prior(train_de)
    pert_prior_de = compute_pert_prior(train_de)
    gene_prior_dir = compute_gene_prior(train_dir)
    pert_prior_dir = compute_pert_prior(train_dir)

    # For each method, compute 3-class predictions
    methods = ['gene_only', 'geometric', 'weighted_pert']

    for method in methods:
        print(f"\nMethod: {method}")

        # DE predictions
        if method == 'gene_only':
            pred_de = np.array([gene_prior_de.get(item['gene'], 0.5) for item in test_de])
            pred_dir = np.array([gene_prior_dir.get(item['gene'], 0.5) for item in test_dir])
        else:
            pred_de = predict_combined(test_de, gene_prior_de, pert_prior_de, method)
            pred_dir = predict_combined(test_dir, gene_prior_dir, pert_prior_dir, method)

        # True labels
        true_de = np.array([item['label'] for item in test_de])
        true_dir = np.array([item['label'] for item in test_dir])

        # 3-class evaluation
        three_class = evaluate_three_class_from_de_doc(
            test_de_keys, test_dir_keys, pred_de, pred_dir,
            true_de, true_dir, de_threshold=0.5
        )

        print(f"  Accuracy: {three_class['accuracy']:.4f}")
        print(f"  Macro F1: {three_class['macro_f1']:.4f}")
        print(f"  Up F1:    {three_class['per_class']['up']['f1']:.4f}")
        print(f"  Down F1:  {three_class['per_class']['down']['f1']:.4f}")
        print(f"  NotDE F1: {three_class['per_class']['not_de']['f1']:.4f}")


if __name__ == "__main__":
    # Test on all available datasets
    datasets = ['k562', 'rpe1', 'hepg2', 'jurkat']

    all_results = {}
    for dataset in datasets:
        try:
            de_results = run_experiment(dataset, 'de')
            dir_results = run_experiment(dataset, 'dir')
            all_results[dataset] = {'de': de_results, 'dir': dir_results}
        except Exception as e:
            print(f"Error with {dataset}: {e}")

    # 3-class for first dataset
    print("\n" + "="*80)
    print("3-CLASS EVALUATION")
    print("="*80)
    for dataset in datasets[:2]:  # Just first two to save time
        try:
            run_3class_experiment(dataset)
        except Exception as e:
            print(f"Error with {dataset}: {e}")

    # Summary
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    print("\nCan combining gene + perturbation priors beat per-gene AUROC of 0.5?")

    for dataset, res in all_results.items():
        for task, results in res.items():
            gene_only = [r for r in results if r['method'] == 'gene_only'][0]
            best_combined = max([r for r in results if r['method'] not in ['gene_only', 'pert_only']],
                               key=lambda x: x['auroc_per_gene'])
            improvement = best_combined['auroc_per_gene'] - gene_only['auroc_per_gene']
            status = "✓" if improvement > 0.01 else "✗"
            print(f"  {dataset}/{task}: gene_only={gene_only['auroc_per_gene']:.4f}, "
                  f"best={best_combined['auroc_per_gene']:.4f} ({best_combined['method']}) {status}")
