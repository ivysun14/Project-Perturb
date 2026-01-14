"""
baseline_transferability.py
===========================

Gene-level majority vote baseline for PerturbQA tasks III.

Train a gene-level majority vote predictor using a training cell line,
and assess the predictor's performance on other testing cell lines using
selected metrics (accuracy, balanced accuracy) + conduct error analysis.
"""


import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from typing import Dict, Tuple, List
from collections import defaultdict
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from perturbqa import load_de, load_dir, auc_per_gene
from baseline import resolve_repeats, train_gene_majority_vote, predict_from_gene_majority
from error_utility import run_error_analysis_from_results


# Define train and test cell line
train_cellline = "hepg2"
test_cellline = "rpe1"

# adata from VCC (Test cell line)
adata = ad.read_h5ad("../../VCC2025/vcc_raw_data/rpe1_raw_singlecell_01.h5ad")
print(f"Loaded: {adata.shape[0]} cells x {adata.shape[1]} genes")
adata.obs.rename(columns={"gene": "target_gene"}, inplace=True)
adata = resolve_repeats(adata, gene_row="gene_name")
sc.pp.normalize_total(adata, target_sum=1e4, key_added="size_factors")
sc.pp.log1p(adata)
print("Log TP10k normalization completed (matching PerturbQA).")

# Load pre-computed DE and DOC files (Train cell line)
de_train = pd.read_csv('files/hepg2_de_results.csv')
doc_train = pd.read_csv("files/hepg2_direction_results.csv")

# Load pre-computed DE and DOC files (Test cell line)
de_test = pd.read_csv('files/rpe1_de_results.csv')
doc_test = pd.read_csv("files/rpe1_direction_results.csv")

# Construst DE and DOC labels (Train cell line)
print("Constructing DE labels for training cell line...")
de_train_labels = {}
for _, row in de_train.iterrows():
    if row['p_value'] < 0.01:
        label = 1
    else:
        label = 0
    de_train_labels[(row['pert'], row['gene'])] = label
print("DE label for training cell line construction completed.")
print("Constructing DOC labels for training cell line...")
doc_train_labels = {}
for _, row in doc_train.iterrows():
    doc_train_labels[(row['pert'], row['gene'])] = row['label']
print("Direction label for training cell line construction completed.")

# Construst DE and DOC labels (Test cell line)
print("Constructing DE labels for testing cell line...")
de_test_labels = {}
for _, row in de_test.iterrows():
    if row['p_value'] < 0.01:
        label = 1
    else:
        label = 0
    de_test_labels[(row['pert'], row['gene'])] = label
print("DE label for testing cell line construction completed.")
print("Constructing DOC labels for testing cell line...")
doc_test_labels = {}
for _, row in doc_test.iterrows():
    doc_test_labels[(row['pert'], row['gene'])] = row['label']
print("Direction label for testing cell line construction completed.")

# Load PerturbQA datasets (Train and test cell line)
data_train_de = load_de(train_cellline)
data_test_de = load_de(test_cellline)
data_train_dir = load_dir(train_cellline)
data_test_dir = load_dir(test_cellline)

X_de_train = data_train_de["train"]
X_de_test = data_test_de["test"]
train_de_keys = [(x["pert"], x["gene"]) for x in X_de_train]
test_de_keys = [(x["pert"], x["gene"]) for x in X_de_test]

X_dir_train = data_train_dir["train"]
X_dir_test = data_test_dir["test"]
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
    print("  Training DE/DOC predictor using training cell line...")
    gene_majority_de = train_gene_majority_vote(train_de_keys_sub, de_train_labels)
    gene_majority_doc = train_gene_majority_vote(train_dir_keys_sub, doc_train_labels)
    
    # Predict on test set
    print("  Predicting on test set from testing cell line...")
    pred_de = predict_from_gene_majority(test_de_keys, gene_majority_de)
    pred_doc = predict_from_gene_majority(test_dir_keys, gene_majority_doc)
    
    # Get true labels for test set
    true_de = np.array([de_test_labels.get(key, 0) for key in test_de_keys])
    true_doc = np.array([doc_test_labels.get(key, 0) for key in test_dir_keys])

    # Calculate AUROC per gene
    print("  Calculating AUROC...")
    auroc_de = auc_per_gene(test_de_keys, pred_de, true_de)
    auroc_doc = auc_per_gene(test_dir_keys, pred_doc, true_doc)
    print("  Calculating ACCURACY...")
    accuracy_de = accuracy_score(true_de, pred_de.astype(int))
    accuracy_doc = accuracy_score(true_doc, pred_doc.astype(int))
    print("  Calculating WEIGHTED ACCURACY...")
    accuracy_weighted_de = balanced_accuracy_score(true_de, pred_de.astype(int))
    accuracy_weighted_doc= balanced_accuracy_score(true_doc, pred_doc.astype(int))
    
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
    if run == 2:
        # DE task error analysis
        print("\n" + "="*60)
        print("Differential Expression Error Analysis")
        print("="*60)
        de_analyzer, de_error_results = run_error_analysis_from_results(
            dataset=f"{train_cellline}_to_{test_cellline}",
            test_keys=test_de_keys,
            predictions=pred_de.astype(int),
            true_labels=true_de.astype(int),
            adata=adata,
            task_name="DE"
        )

        # Direction task error analysis
        print("\n" + "="*60)
        print("Direction of Change Error Analysis")
        print("="*60)
        doc_analyzer, doc_error_results = run_error_analysis_from_results(
            dataset=f"{train_cellline}_to_{test_cellline}",
            test_keys=test_dir_keys,
            predictions=pred_doc.astype(int),
            true_labels=true_doc.astype(int),
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
print(f"FINAL RESULTS - Train on {train_cellline} and test on {test_cellline}")
print("="*50)
print(f"MAJORITY VOTE (DE_AUROC):  {de_auroc_mean:.2f}±{de_auroc_std:.2f}")
print(f"MAJORITY VOTE (DE_ACCURACY): {de_accuracy_mean:.2f}±{de_accuracy_std:.2f}")
print(f"MAJORITY VOTE (DE_BALANCED_ACCURACY): {de_balanced_accuracy_mean:.2f}±{de_balanced_accuracy_std:.2f}")
print(f"MAJORITY VOTE (DOC_AUROC):  {doc_auroc_mean:.2f}±{doc_auroc_std:.2f}")
print(f"MAJORITY VOTE (DOC_ACCURACY): {doc_accuracy_mean:.2f}±{doc_accuracy_std:.2f}")
print(f"MAJORITY VOTE (DOC_BALANCED_ACCURACY): {doc_balanced_accuracy_mean:.2f}±{doc_balanced_accuracy_std:.2f}")