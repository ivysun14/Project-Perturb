"""
baseline_accuracy.py
====================

Gene-level majority vote baseline for PerturbQA tasks II.

Read in pre-computed DE results and compute additional metrics (accuracy, balanced
accuracy) + conduct error analysis.
"""


import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from typing import Dict, Tuple, List
from collections import defaultdict
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from perturbqa import load_de, load_dir, auc_per_gene
from error_utility import run_error_analysis_from_results


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


def train_gene_majority_vote(train_keys: List[Tuple], labels: Dict) -> Dict:
    """
    Train gene-level majority vote predictor.
    
    Args:
        train_keys: List of (perturbation, gene) tuples in training set
        labels: Dict mapping (pert, gene) -> label
        
    Returns:
        Dict mapping gene -> majority class
    """
    
    gene_stats = defaultdict(lambda: {'pos': 0, 'neg': 0})
    
    for pert, gene in train_keys:
        key = (pert, gene)
        if key in labels:
            label = labels[key]
            if label == 1:
                gene_stats[gene]['pos'] += 1
            else:
                gene_stats[gene]['neg'] += 1
    
    # Determine majority class per gene
    gene_majority = {}
    for gene, stats in gene_stats.items():
        gene_majority[gene] = 1 if stats['pos'] >= stats['neg'] else 0
    
    return gene_majority


def predict_from_gene_majority(test_keys: List[Tuple], 
                               gene_majority: Dict, 
                               default_pred: float = 0.5) -> np.ndarray:
    """
    Generate predictions based on gene-level majority vote.
    
    Args:
        test_keys: List of (perturbation, gene) tuples in test set
        gene_majority: Dict mapping gene -> majority class
        default_pred: Default prediction for unseen genes
        
    Returns:
        Array of predictions
    """
    predictions = []
    for pert, gene in test_keys:
        if gene in gene_majority:
            predictions.append(float(gene_majority[gene]))
        else:
            predictions.append(default_pred)
    
    return np.array(predictions)

# adata from VCC
adata = ad.read_h5ad("../../VCC2025/vcc_raw_data/rpe1_raw_singlecell_01.h5ad")
print(f"Loaded: {adata.shape[0]} cells x {adata.shape[1]} genes")
adata.obs.rename(columns={"gene": "target_gene"}, inplace=True)
adata = resolve_repeats(adata, gene_row="gene_name")
sc.pp.normalize_total(adata, target_sum=1e4, key_added="size_factors")
sc.pp.log1p(adata)
print("Log TP10k normalization completed (matching PerturbQA).")

# Load pre-computed DE and DOC files
dataset_name = "rpe1"
de = pd.read_csv('files/rpe1_de_results.csv')
doc = pd.read_csv("files/rpe1_direction_results.csv")

print("Constructing DE labels...")
de_labels = {}
for _, row in de.iterrows():
    if row['p_value'] < 0.01:
        label = 1
    else:
        label = 0
    de_labels[(row['pert'], row['gene'])] = label
print("DE label construction completed.")

print("Constructing DOC labels...")
doc_labels = {}
for _, row in doc.iterrows():
    doc_labels[(row['pert'], row['gene'])] = row['label']
print("Direction label construction completed.")

# Load PerturbQA datasets
data_de = load_de(dataset_name)
X_de_train = data_de["train"]
X_de_test = data_de["test"]
train_de_keys = [(x["pert"], x["gene"]) for x in X_de_train]
test_de_keys = [(x["pert"], x["gene"]) for x in X_de_test]

data_dir = load_dir(dataset_name)
X_dir_train = data_dir["train"]
X_dir_test = data_dir["test"]
train_dir_keys = [(x["pert"], x["gene"]) for x in X_dir_train]
test_dir_keys = [(x["pert"], x["gene"]) for x in X_dir_test]

results = {'de_aurocs': [], 
           'de_accuracy': [],
           'de_balanced_accuracy': [],
           'doc_aurocs': [], 
           'doc_accuracy': [],
           'doc_balanced_accuracy': []}

# Metrics and error analysis
for run in range(3):
        
    print(f"\nRun {run + 1}/3")
    
    # Subsample training data (80% of train_de_keys and train_dir_keys)
    np.random.seed(run)
    
    n_de_train = int(len(train_de_keys) * 0.8)
    train_de_subsample = np.random.choice(len(train_de_keys), n_de_train, replace=False)
    train_de_keys_sub = [train_de_keys[i] for i in train_de_subsample]
    
    n_dir_train = int(len(train_dir_keys) * 0.8)
    train_dir_subsample = np.random.choice(len(train_dir_keys), n_dir_train, replace=False)
    train_dir_keys_sub = [train_dir_keys[i] for i in train_dir_subsample]
    
    # Train gene-level majority vote for DE and DOC
    print("  Training DE/DOC predictor...")
    gene_majority_de = train_gene_majority_vote(train_de_keys_sub, de_labels)
    gene_majority_doc = train_gene_majority_vote(train_dir_keys_sub, doc_labels)
    
    # Predict on test set
    print("  Predicting on test set...")
    pred_de = predict_from_gene_majority(test_de_keys, gene_majority_de)
    pred_doc = predict_from_gene_majority(test_dir_keys, gene_majority_doc)
    
    # Get true labels for test set
    true_de = np.array([de_labels.get(key, 0) for key in test_de_keys])
    true_doc = np.array([doc_labels.get(key, 0) for key in test_dir_keys])
    
    # Calculate AUROC per gene
    print("  Calculating AUROC...")
    auroc_de = auc_per_gene(test_de_keys, pred_de, true_de)
    auroc_doc = auc_per_gene(test_dir_keys, pred_doc, true_doc)
    print("  Calculating ACCURACY...")
    accuracy_de = accuracy_score(true_de, pred_de)
    accuracy_doc = accuracy_score(true_doc, pred_doc)
    print("  Calculating WEIGHTED ACCURACY...")
    accuracy_weighted_de = balanced_accuracy_score(true_de, pred_de)
    accuracy_weighted_doc= balanced_accuracy_score(true_doc, pred_doc)
    
    print(f"  DE AUROC: {auroc_de:.4f}")
    print(f"  DE ACCURACY: {accuracy_de:.4f}")
    print(f"  DE BALANCED ACCURACY: {accuracy_weighted_de:.4f}")
    print(f"  DOC AUROC: {auroc_doc:.4f}")
    print(f"  DOC ACCURACY: {accuracy_doc:.4f}")
    print(f"  DOC BALANCED ACCURACY: {accuracy_weighted_doc:.4f}")
    
    results['de_aurocs'].append(auroc_de)
    results['de_accuracy'].append(accuracy_de)
    results['de_balanced_accuracy'].append(accuracy_weighted_de)
    results['doc_aurocs'].append(auroc_doc)
    results['doc_accuracy'].append(accuracy_doc)
    results['doc_balanced_accuracy'].append(accuracy_weighted_doc)

    # Run error analysis in the final run
    if run == 3:
        # DE task error analysis
        print("\n" + "="*60)
        print("Differential Expression Error Analysis")
        print("="*60)
        de_analyzer, de_error_results = run_error_analysis_from_results(
            test_de_keys,
            pred_de,
            true_de,
            adata=adata,
            task_name="DE"
        )

        # Direction task error analysis
        print("\n" + "="*60)
        print("Direction of Change Error Analysis")
        print("="*60)
        doc_analyzer, doc_error_results = run_error_analysis_from_results(
            test_dir_keys,
            pred_doc,
            true_doc,
            adata=adata,
            task_name="DOC"
        )

# Calculate mean and std
de_auroc_mean = np.mean(results['de_aurocs'])
de_auroc_std = np.std(results['de_aurocs'])
de_accuracy_mean = np.mean(results['de_accuracy'])
de_accuracy_std = np.std(results['de_accuracy'])
de_balanced_accuracy_mean = np.mean(results['de_balanced_accuracy'])
de_balanced_accuracy_std = np.std(results['de_balanced_accuracy'])
doc_auroc_mean = np.mean(results['doc_aurocs'])
doc_auroc_std = np.std(results['doc_aurocs'])
doc_accuracy_mean = np.mean(results['doc_accuracy'])
doc_accuracy_std = np.std(results['doc_accuracy'])
doc_balanced_accuracy_mean = np.mean(results['doc_balanced_accuracy'])
doc_balanced_accuracy_std = np.std(results['doc_balanced_accuracy'])

print("\n" + "="*50)
print("FINAL RESULTS")
print("="*50)
print(f"MAJORITY VOTE (DE_AUROC):  {de_auroc_mean:.2f}±{de_auroc_std:.2f}")
print(f"MAJORITY VOTE (DE_ACCURACY): {de_accuracy_mean:.2f}±{de_accuracy_std:.2f}")
print(f"MAJORITY VOTE (DE_BALANCED_ACCURACY): {de_balanced_accuracy_mean:.2f}±{de_balanced_accuracy_std:.2f}")
print(f"MAJORITY VOTE (DOC_AUROC):  {doc_auroc_mean:.2f}±{doc_auroc_std:.2f}")
print(f"MAJORITY VOTE (DOC_ACCURACY): {doc_accuracy_mean:.2f}±{doc_accuracy_std:.2f}")
print(f"MAJORITY VOTE (DOC_BALANCED_ACCURACY): {doc_balanced_accuracy_mean:.2f}±{doc_balanced_accuracy_std:.2f}")