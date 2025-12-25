import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from typing import Dict, Tuple, List
from collections import defaultdict
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from perturbqa import load_de, load_dir, auc_per_gene

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

dataset_name = "rpe1"
de = pd.read_csv('files/rpe1_de_results.csv')

print("Constructing DE labels...")
de_labels = {}
for _, row in de.iterrows():
    if row['p_value'] < 0.01:
        label = 1
    else:
        label = 0
    de_labels[(row['pert'], row['gene'])] = label
print("DE label construction completed.")

data_de = load_de(dataset_name)
X_de_train = data_de["train"]
X_de_test = data_de["test"]
train_de_keys = [(x["pert"], x["gene"]) for x in X_de_train]
test_de_keys = [(x["pert"], x["gene"]) for x in X_de_test]

results = {'de_aurocs': [], 
           'de_accuracy': [],
           'de_balanced_accuracy': []}

for run in range(3):
        
    print(f"\nRun {run + 1}/3")
    
    # Subsample training data (80% of train_de_keys)
    np.random.seed(run)
    
    n_de_train = int(len(train_de_keys) * 0.8)
    train_de_subsample = np.random.choice(len(train_de_keys), n_de_train, replace=False)
    train_de_keys_sub = [train_de_keys[i] for i in train_de_subsample]
    
    # Train gene-level majority vote for DE
    print("  Training DE predictor...")
    gene_majority_de = train_gene_majority_vote(train_de_keys_sub, de_labels)
    
    # Predict on test set
    print("  Predicting on test set...")
    pred_de = predict_from_gene_majority(test_de_keys, gene_majority_de)
    
    # Get true labels for test set
    true_de = np.array([de_labels.get(key, 0) for key in test_de_keys])
    
    # Calculate AUROC per gene
    print("  Calculating AUROC...")
    auroc_de = auc_per_gene(test_de_keys, pred_de, true_de)
    print("  Calculating ACCURACY...")
    accuracy_de = accuracy_score(true_de, pred_de)
    print("  Calculating WEIGHTED ACCURACY...")
    accuracy_weighted_de = balanced_accuracy_score(true_de, pred_de)
    
    print(f"  DE AUROC: {auroc_de:.4f}")
    print(f"  DE ACCURACY: {accuracy_de:.4f}")
    print(f"  DE BALANCED ACCURACY: {accuracy_weighted_de:.4f}")
    
    results['de_aurocs'].append(auroc_de)
    results['de_accuracy'].append(accuracy_de)
    results['de_balanced_accuracy'].append(accuracy_weighted_de)

# Calculate mean and std
de_auroc_mean = np.mean(results['de_aurocs'])
de_auroc_std = np.std(results['de_aurocs'])
de_accuracy_mean = np.mean(results['de_accuracy'])
de_accuracy_std = np.std(results['de_accuracy'])
de_balanced_accuracy_mean = np.mean(results['de_balanced_accuracy'])
de_balanced_accuracy_std = np.std(results['de_balanced_accuracy'])

print("\n" + "="*50)
print("FINAL RESULTS")
print("="*50)
print(f"MAJORITY VOTE (DE_AUROC):  {de_auroc_mean:.2f}±{de_auroc_std:.2f}")
print(f"MAJORITY VOTE (DE_ACCURACY): {de_accuracy_mean:.2f}±{de_accuracy_std:.2f}")
print(f"MAJORITY VOTE (DE_BALANCED_ACCURACY): {de_balanced_accuracy_mean:.2f}±{de_balanced_accuracy_std:.2f}")