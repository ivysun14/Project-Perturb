"""
Evaluation utilities for perturbation prediction models.

Provides functions to score model predictions against ground truth
and compare different prediction strategies.

Includes PerturbQA-style evaluation metrics for:
- Differential Expression (DE) prediction
- Direction of Change (DOC) prediction
"""

import json
import re
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from collections import Counter, defaultdict
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
    average_precision_score
)


def extract_json_from_output(output: str) -> Dict:
    """
    Extract JSON from model output string.

    Handles outputs wrapped in markdown code blocks.

    Parameters
    ----------
    output : str
        Model output string, possibly containing ```json blocks

    Returns
    -------
    Dict or List
        Parsed JSON object

    Raises
    ------
    ValueError
        If no valid JSON block is found
    """
    # Try to extract from markdown code block
    json_match = re.search(r"```json\n(.*?)\n```", output, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(1))

    # Try to extract from generic code block
    json_match = re.search(r"```\n(.*?)\n```", output, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try parsing the whole string
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        pass

    # Try finding JSON array or object in the string
    json_match = re.search(r'(\[.*\]|\{.*\})', output, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    raise ValueError("No valid JSON block found in the output string.")


def evaluate_direction_predictions(
    predictions: List[Dict],
    ground_truth: List[Dict],
) -> Dict:
    """
    Evaluate predictions for direction-based format (increase/decrease/no change).

    Parameters
    ----------
    predictions : List[Dict]
        List of predictions with format:
        [{'gene_name': str, 'direction': str, 'amount': float}, ...]
    ground_truth : List[Dict]
        Ground truth in same format

    Returns
    -------
    Dict
        Evaluation metrics including total score and breakdown
    """
    total_score = 0
    correct_direction = 0
    wrong_direction = 0
    opposite_direction = 0

    gt_dict = {item['gene_name']: item for item in ground_truth}

    for pred in predictions:
        gene = pred.get('gene_name')
        if gene not in gt_dict:
            continue

        gt = gt_dict[gene]
        pred_dir = pred.get('direction', '').lower()
        gt_dir = gt.get('direction', '').lower()
        pred_amt = pred.get('amount', 0)
        gt_amt = gt.get('amount', 0)

        if pred_dir == gt_dir:
            correct_direction += 1
            if gt_dir == "no significant movement":
                total_score += 0.5
            else:
                # Score based on amount accuracy
                if abs(pred_amt - gt_amt) <= 20:
                    total_score += 1.0
                elif abs(pred_amt - gt_amt) <= 40:
                    total_score += 0.75
                else:
                    total_score += 0.5
        elif {pred_dir, gt_dir} == {"increase", "decrease"}:
            opposite_direction += 1
            total_score -= 0.5
        else:
            wrong_direction += 1

    return {
        "total_score": total_score,
        "correct_direction": correct_direction,
        "wrong_direction": wrong_direction,
        "opposite_direction": opposite_direction,
        "n_predictions": len(predictions),
        "n_ground_truth": len(ground_truth),
    }


def evaluate_category_predictions(
    predictions: List[Dict],
    ground_truth: List[Dict],
    gene_key: str = "gene",
    category_key: str = "zscore",
) -> Dict:
    """
    Evaluate predictions for category-based format (z-score categories).

    Parameters
    ----------
    predictions : List[Dict]
        Predictions with gene and category keys
    ground_truth : List[Dict]
        Ground truth in same format
    gene_key : str
        Key for gene name in dicts
    category_key : str
        Key for category in dicts

    Returns
    -------
    Dict
        Evaluation metrics including accuracy and confusion matrix
    """
    gt_dict = {item[gene_key]: item[category_key] for item in ground_truth}

    correct = 0
    total = 0
    confusion = Counter()

    for pred in predictions:
        gene = pred.get(gene_key)
        if gene not in gt_dict:
            continue

        pred_cat = pred.get(category_key, "")
        gt_cat = gt_dict[gene]
        total += 1

        if pred_cat == gt_cat:
            correct += 1

        confusion[(gt_cat, pred_cat)] += 1

    accuracy = correct / total if total > 0 else 0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "confusion_matrix": dict(confusion),
    }


def evaluate_set_predictions(
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]],
) -> Dict:
    """
    Evaluate predictions in set format (INCREASE/DECREASE/NO_CHANGE lists).

    Parameters
    ----------
    predictions : Dict[str, List[str]]
        Predictions with keys INCREASE, DECREASE, NO_CHANGE
    ground_truth : Dict[str, List[str]]
        Ground truth in same format

    Returns
    -------
    Dict
        Metrics for each category and overall
    """
    results = {}

    for category in ["INCREASE", "DECREASE", "NO_CHANGE"]:
        pred_set = set(predictions.get(category, []))
        gt_set = set(ground_truth.get(category, []))

        tp = len(pred_set & gt_set)
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results[category] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
        }

    # Overall metrics
    all_pred = set()
    all_gt = set()
    for cat in ["INCREASE", "DECREASE", "NO_CHANGE"]:
        for gene in predictions.get(cat, []):
            all_pred.add((gene, cat))
        for gene in ground_truth.get(cat, []):
            all_gt.add((gene, cat))

    correct = len(all_pred & all_gt)
    total = len(all_gt)
    accuracy = correct / total if total > 0 else 0

    results["overall"] = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }

    return results


def create_baseline_predictions(
    ground_truth: List[Dict],
    strategy: str = "all_no_change",
) -> List[Dict]:
    """
    Create baseline predictions for comparison.

    Parameters
    ----------
    ground_truth : List[Dict]
        Ground truth data
    strategy : str
        Baseline strategy: "all_no_change", "random", "majority_class"

    Returns
    -------
    List[Dict]
        Baseline predictions
    """
    import random

    if strategy == "all_no_change":
        return [
            {
                "gene_name": gt.get("gene_name", gt.get("gene")),
                "direction": "no significant movement",
                "amount": 0
            }
            for gt in ground_truth
        ]

    elif strategy == "random":
        directions = ["increase", "decrease", "no significant movement"]
        return [
            {
                "gene_name": gt.get("gene_name", gt.get("gene")),
                "direction": random.choice(directions),
                "amount": random.randint(0, 100) if random.choice([True, False]) else 0
            }
            for gt in ground_truth
        ]

    elif strategy == "majority_class":
        # Find majority class
        direction_counts = Counter()
        for gt in ground_truth:
            direction_counts[gt.get("direction", "")] += 1
        majority = direction_counts.most_common(1)[0][0]

        return [
            {
                "gene_name": gt.get("gene_name", gt.get("gene")),
                "direction": majority,
                "amount": 0
            }
            for gt in ground_truth
        ]

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def count_gene_changes(jsonl_file: str) -> Dict[str, int]:
    """
    Count gene changes by category in a JSONL file.

    Parameters
    ----------
    jsonl_file : str
        Path to JSONL file

    Returns
    -------
    Dict[str, int]
        Counts for each category
    """
    counts = {"NO_CHANGE": 0, "INCREASE": 0, "DECREASE": 0}

    with open(jsonl_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            answer = data.get("answer", {})
            for key in counts:
                counts[key] += len(answer.get(key, []))

    return counts


def find_contradictory_pairs(jsonl_file: str) -> Tuple[Dict, Counter]:
    """
    Find genes with contradictory effects across different perturbations.

    Parameters
    ----------
    jsonl_file : str
        Path to JSONL file

    Returns
    -------
    Tuple[Dict, Counter]
        Dict of genes with contradictory effects and counts by contradiction type
    """
    from collections import defaultdict

    gene_effects = defaultdict(set)

    with open(jsonl_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            answer = data.get("answer", {})

            for effect, genes in answer.items():
                for gene in genes:
                    gene_effects[gene].add(effect)

    contradictory = {
        gene: effects
        for gene, effects in gene_effects.items()
        if len(effects) > 1
    }

    contradiction_counts = Counter()
    for effects in contradictory.values():
        sorted_effects = tuple(sorted(effects))
        contradiction_counts[sorted_effects] += 1

    return contradictory, contradiction_counts


def print_evaluation_summary(results: Dict) -> None:
    """Print a formatted evaluation summary."""
    print("=" * 50)
    print("Evaluation Summary")
    print("=" * 50)

    if "total_score" in results:
        print(f"Total Score: {results['total_score']:.2f}")
        print(f"Correct Direction: {results.get('correct_direction', 'N/A')}")
        print(f"Wrong Direction: {results.get('wrong_direction', 'N/A')}")
        print(f"Opposite Direction: {results.get('opposite_direction', 'N/A')}")

    if "accuracy" in results:
        print(f"Accuracy: {results['accuracy']:.2%}")
        print(f"Correct: {results['correct']} / {results['total']}")

    if "overall" in results:
        print(f"\nOverall Accuracy: {results['overall']['accuracy']:.2%}")
        for cat in ["INCREASE", "DECREASE", "NO_CHANGE"]:
            if cat in results:
                print(f"\n{cat}:")
                print(f"  Precision: {results[cat]['precision']:.2%}")
                print(f"  Recall: {results[cat]['recall']:.2%}")
                print(f"  F1: {results[cat]['f1']:.2%}")

    print("=" * 50)


# =============================================================================
# PerturbQA-style Evaluation Metrics
# =============================================================================
# Following the evaluation protocol from:
# "Contextualizing biological perturbation experiments through language"
# (Wu et al., ICLR 2025) https://arxiv.org/abs/2502.21290


def get_group_to_idx(vals: List) -> Dict:
    """
    Create a mapping from unique values to their indices.

    Parameters
    ----------
    vals : List
        List of values to group

    Returns
    -------
    Dict
        Mapping from unique values to list of indices
    """
    group_to_idx = defaultdict(list)
    for idx, val in enumerate(vals):
        group_to_idx[val].append(idx)
    return dict(group_to_idx)


def auc_per_gene(
    keys: List[Tuple[str, str]],
    pred: Union[List, np.ndarray],
    true: Union[List, np.ndarray],
    reduce: bool = True
) -> Union[float, List[float]]:
    """
    Compute AUROC per gene, following PerturbQA evaluation protocol.

    AUROC is computed for predictions associated with each gene (across
    all perturbations for that gene), then averaged across genes.

    Parameters
    ----------
    keys : List[Tuple[str, str]]
        List of (perturbation, gene) tuples identifying each prediction
    pred : array-like
        Predictions (can be probabilities or scores, not just 0/1)
    true : array-like
        Ground truth binary labels (0 or 1)
    reduce : bool
        If True, return mean AUROC across genes.
        If False, return list of per-gene AUROC scores.

    Returns
    -------
    float or List[float]
        Mean AUROC (if reduce=True) or list of per-gene AUROCs

    Notes
    -----
    - Genes with uniform labels (all 0s or all 1s) are skipped since
      AUROC is undefined for them
    - This follows the PerturbQA protocol: "AUROC is computed over the
      predictions associated with each gene, and averaged over perturbations"
    """
    pred = np.asarray(pred)
    true = np.asarray(true)

    # Extract gene names (second element of each key tuple)
    genes = [key[1] for key in keys]

    # Group indices by gene
    gene_to_idx = get_group_to_idx(genes)

    aurocs = []
    skipped_genes = []

    for gene, indices in gene_to_idx.items():
        gene_true = true[indices]
        gene_pred = pred[indices]

        # Skip genes with uniform labels (AUROC is undefined)
        unique_labels = np.unique(gene_true)
        if len(unique_labels) < 2:
            skipped_genes.append(gene)
            continue

        try:
            auroc = roc_auc_score(gene_true, gene_pred)
            aurocs.append(auroc)
        except ValueError as e:
            # This can happen if all predictions are the same
            skipped_genes.append(gene)
            continue

    if len(aurocs) == 0:
        raise ValueError(
            f"No valid AUROC could be computed. "
            f"All {len(skipped_genes)} genes had uniform labels."
        )

    if reduce:
        return float(np.mean(aurocs))
    return aurocs


def auc_per_perturbation(
    keys: List[Tuple[str, str]],
    pred: Union[List, np.ndarray],
    true: Union[List, np.ndarray],
    reduce: bool = True
) -> Union[float, List[float]]:
    """
    Compute AUROC per perturbation, then averaged across perturbations.

    For each perturbation, compute AUROC across all genes affected by that
    perturbation, then average across perturbations.

    Parameters
    ----------
    keys : List[Tuple[str, str]]
        List of (perturbation, gene) tuples identifying each prediction
    pred : array-like
        Predictions (can be probabilities or scores)
    true : array-like
        Ground truth binary labels (0 or 1)
    reduce : bool
        If True, return mean AUROC across perturbations.
        If False, return list of per-perturbation AUROC scores.

    Returns
    -------
    float or List[float]
        Mean AUROC (if reduce=True) or list of per-perturbation AUROCs

    Notes
    -----
    - Perturbations with uniform labels are skipped
    - This is more appropriate for gene-level baselines that assign
      different scores to different genes
    """
    pred = np.asarray(pred)
    true = np.asarray(true)

    # Extract perturbation names (first element of each key tuple)
    perts = [key[0] for key in keys]

    # Group indices by perturbation
    pert_to_idx = get_group_to_idx(perts)

    aurocs = []
    skipped_perts = []

    for pert, indices in pert_to_idx.items():
        pert_true = true[indices]
        pert_pred = pred[indices]

        # Skip perturbations with uniform labels
        unique_labels = np.unique(pert_true)
        if len(unique_labels) < 2:
            skipped_perts.append(pert)
            continue

        try:
            auroc = roc_auc_score(pert_true, pert_pred)
            aurocs.append(auroc)
        except ValueError:
            skipped_perts.append(pert)
            continue

    if len(aurocs) == 0:
        raise ValueError(
            f"No valid AUROC could be computed. "
            f"All {len(skipped_perts)} perturbations had uniform labels."
        )

    if reduce:
        return float(np.mean(aurocs))
    return aurocs


def auprc_per_gene(
    keys: List[Tuple[str, str]],
    pred: Union[List, np.ndarray],
    true: Union[List, np.ndarray],
    reduce: bool = True
) -> Union[float, List[float]]:
    """
    Compute Area Under Precision-Recall Curve per gene.

    Similar to auc_per_gene but uses AUPRC instead of AUROC.
    AUPRC is more informative when classes are imbalanced.

    Parameters
    ----------
    keys : List[Tuple[str, str]]
        List of (perturbation, gene) tuples
    pred : array-like
        Prediction scores
    true : array-like
        Ground truth binary labels
    reduce : bool
        If True, return mean AUPRC

    Returns
    -------
    float or List[float]
        Mean AUPRC (if reduce=True) or list of per-gene AUPRCs
    """
    pred = np.asarray(pred)
    true = np.asarray(true)

    genes = [key[1] for key in keys]
    gene_to_idx = get_group_to_idx(genes)

    auprcs = []

    for gene, indices in gene_to_idx.items():
        gene_true = true[indices]
        gene_pred = pred[indices]

        unique_labels = np.unique(gene_true)
        if len(unique_labels) < 2:
            continue

        try:
            auprc = average_precision_score(gene_true, gene_pred)
            auprcs.append(auprc)
        except ValueError:
            continue

    if len(auprcs) == 0:
        raise ValueError("No valid AUPRC could be computed.")

    if reduce:
        return float(np.mean(auprcs))
    return auprcs


def compute_binary_metrics(
    pred: Union[List, np.ndarray],
    true: Union[List, np.ndarray],
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute comprehensive binary classification metrics.

    Parameters
    ----------
    pred : array-like
        Predictions (probabilities/scores for AUROC, or binary for other metrics)
    true : array-like
        Ground truth binary labels
    threshold : float
        Threshold for converting probabilities to binary predictions

    Returns
    -------
    Dict[str, float]
        Dictionary with metrics: accuracy, precision, recall, f1, auroc, auprc
    """
    pred = np.asarray(pred)
    true = np.asarray(true)

    # Binary predictions for precision/recall/F1
    pred_binary = (pred >= threshold).astype(int)

    results = {
        'accuracy': accuracy_score(true, pred_binary),
    }

    # Precision, recall, F1 for positive class (label=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true, pred_binary, average='binary', zero_division=0
    )
    results['precision'] = precision
    results['recall'] = recall
    results['f1'] = f1

    # AUROC and AUPRC (need non-binary predictions for these to be meaningful)
    unique_labels = np.unique(true)
    if len(unique_labels) >= 2:
        try:
            results['auroc'] = roc_auc_score(true, pred)
            results['auprc'] = average_precision_score(true, pred)
        except ValueError:
            results['auroc'] = np.nan
            results['auprc'] = np.nan
    else:
        results['auroc'] = np.nan
        results['auprc'] = np.nan

    return results


def compute_perturbqa_metrics(
    keys: List[Tuple[str, str]],
    pred: Union[List, np.ndarray],
    true: Union[List, np.ndarray],
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute full suite of PerturbQA-style metrics.

    Parameters
    ----------
    keys : List[Tuple[str, str]]
        List of (perturbation, gene) tuples
    pred : array-like
        Prediction scores
    true : array-like
        Ground truth binary labels
    threshold : float
        Threshold for binary predictions

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - auroc_per_gene: AUROC computed per gene, then averaged (PerturbQA style)
        - auprc_per_gene: AUPRC computed per gene, then averaged
        - auroc_overall: Overall AUROC (not grouped by gene)
        - accuracy: Overall accuracy
        - precision: Precision for positive class
        - recall: Recall for positive class
        - f1: F1 score for positive class
        - n_genes_evaluated: Number of genes with valid AUROC
        - n_genes_skipped: Number of genes skipped (uniform labels)
    """
    pred = np.asarray(pred)
    true = np.asarray(true)

    results = {}

    # Per-gene metrics (PerturbQA style)
    genes = [key[1] for key in keys]
    gene_to_idx = get_group_to_idx(genes)

    aurocs_per_gene = []
    auprcs_per_gene = []
    skipped = 0

    for gene, indices in gene_to_idx.items():
        gene_true = true[indices]
        gene_pred = pred[indices]

        unique_labels = np.unique(gene_true)
        if len(unique_labels) < 2:
            skipped += 1
            continue

        try:
            aurocs_per_gene.append(roc_auc_score(gene_true, gene_pred))
            auprcs_per_gene.append(average_precision_score(gene_true, gene_pred))
        except ValueError:
            skipped += 1
            continue

    results['auroc_per_gene'] = float(np.mean(aurocs_per_gene)) if aurocs_per_gene else np.nan
    results['auprc_per_gene'] = float(np.mean(auprcs_per_gene)) if auprcs_per_gene else np.nan
    results['n_genes_evaluated'] = len(aurocs_per_gene)
    results['n_genes_skipped'] = skipped

    # Overall metrics (not grouped)
    unique_labels = np.unique(true)
    if len(unique_labels) >= 2:
        results['auroc_overall'] = roc_auc_score(true, pred)
        results['auprc_overall'] = average_precision_score(true, pred)
    else:
        results['auroc_overall'] = np.nan
        results['auprc_overall'] = np.nan

    # Binary classification metrics
    pred_binary = (pred >= threshold).astype(int)
    results['accuracy'] = accuracy_score(true, pred_binary)

    precision, recall, f1, _ = precision_recall_fscore_support(
        true, pred_binary, average='binary', zero_division=0
    )
    results['precision'] = precision
    results['recall'] = recall
    results['f1'] = f1

    # Class distribution
    results['n_positive'] = int(np.sum(true))
    results['n_negative'] = int(len(true) - np.sum(true))
    results['positive_rate'] = float(np.mean(true))

    return results


def evaluate_de_task(
    keys: List[Tuple[str, str]],
    pred: Union[List, np.ndarray],
    true: Union[List, np.ndarray],
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate Differential Expression (DE) prediction task.

    This is a binary classification task: predict whether a gene is
    differentially expressed (DE=1) or not (DE=0) upon perturbation.

    Parameters
    ----------
    keys : List[Tuple[str, str]]
        List of (perturbation, gene) tuples
    pred : array-like
        Prediction scores for DE
    true : array-like
        Ground truth DE labels (0 or 1)
    threshold : float
        Threshold for binary predictions

    Returns
    -------
    Dict[str, float]
        Evaluation metrics for DE task
    """
    results = compute_perturbqa_metrics(keys, pred, true, threshold)
    results['task'] = 'DE'
    return results


def evaluate_doc_task(
    keys: List[Tuple[str, str]],
    pred: Union[List, np.ndarray],
    true: Union[List, np.ndarray],
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate Direction of Change (DOC) prediction task.

    This is a binary classification task: predict whether a gene is
    up-regulated (DOC=1) or down-regulated (DOC=0) upon perturbation.

    Note: This task is typically evaluated only on DE genes.

    Parameters
    ----------
    keys : List[Tuple[str, str]]
        List of (perturbation, gene) tuples
    pred : array-like
        Prediction scores for direction (higher = up-regulated)
    true : array-like
        Ground truth direction labels (1=up, 0=down)
    threshold : float
        Threshold for binary predictions

    Returns
    -------
    Dict[str, float]
        Evaluation metrics for DOC task
    """
    results = compute_perturbqa_metrics(keys, pred, true, threshold)
    results['task'] = 'DOC'
    return results


def run_perturbqa_evaluation(
    de_keys: List[Tuple[str, str]],
    de_pred: Union[List, np.ndarray],
    de_true: Union[List, np.ndarray],
    doc_keys: List[Tuple[str, str]],
    doc_pred: Union[List, np.ndarray],
    doc_true: Union[List, np.ndarray],
    threshold: float = 0.5,
    print_results: bool = True
) -> Dict[str, Dict]:
    """
    Run complete PerturbQA evaluation for both DE and DOC tasks.

    Parameters
    ----------
    de_keys : List[Tuple[str, str]]
        Keys for DE task
    de_pred : array-like
        DE predictions
    de_true : array-like
        DE ground truth
    doc_keys : List[Tuple[str, str]]
        Keys for DOC task
    doc_pred : array-like
        DOC predictions
    doc_true : array-like
        DOC ground truth
    threshold : float
        Threshold for binary predictions
    print_results : bool
        Whether to print formatted results

    Returns
    -------
    Dict[str, Dict]
        Results for both tasks
    """
    de_results = evaluate_de_task(de_keys, de_pred, de_true, threshold)
    doc_results = evaluate_doc_task(doc_keys, doc_pred, doc_true, threshold)

    if print_results:
        print("\n" + "=" * 60)
        print("PerturbQA Evaluation Results")
        print("=" * 60)

        print("\n--- Differential Expression (DE) Task ---")
        print(f"  AUROC (per-gene avg):  {de_results['auroc_per_gene']:.4f}")
        print(f"  AUROC (overall):       {de_results['auroc_overall']:.4f}")
        print(f"  AUPRC (per-gene avg):  {de_results['auprc_per_gene']:.4f}")
        print(f"  Accuracy:              {de_results['accuracy']:.4f}")
        print(f"  Precision:             {de_results['precision']:.4f}")
        print(f"  Recall:                {de_results['recall']:.4f}")
        print(f"  F1:                    {de_results['f1']:.4f}")
        print(f"  Genes evaluated:       {de_results['n_genes_evaluated']}")
        print(f"  Genes skipped:         {de_results['n_genes_skipped']}")
        print(f"  Positive rate:         {de_results['positive_rate']:.2%}")

        print("\n--- Direction of Change (DOC) Task ---")
        print(f"  AUROC (per-gene avg):  {doc_results['auroc_per_gene']:.4f}")
        print(f"  AUROC (overall):       {doc_results['auroc_overall']:.4f}")
        print(f"  AUPRC (per-gene avg):  {doc_results['auprc_per_gene']:.4f}")
        print(f"  Accuracy:              {doc_results['accuracy']:.4f}")
        print(f"  Precision:             {doc_results['precision']:.4f}")
        print(f"  Recall:                {doc_results['recall']:.4f}")
        print(f"  F1:                    {doc_results['f1']:.4f}")
        print(f"  Genes evaluated:       {doc_results['n_genes_evaluated']}")
        print(f"  Genes skipped:         {doc_results['n_genes_skipped']}")
        print(f"  Up-regulation rate:    {doc_results['positive_rate']:.2%}")

        print("\n" + "=" * 60)

    return {'de': de_results, 'doc': doc_results}


def print_perturbqa_summary(results: Dict[str, Dict]) -> None:
    """
    Print a summary table of PerturbQA results.

    Parameters
    ----------
    results : Dict[str, Dict]
        Results from run_perturbqa_evaluation
    """
    de = results['de']
    doc = results['doc']

    print("\n" + "=" * 50)
    print("PerturbQA Results Summary")
    print("=" * 50)
    print(f"{'Metric':<25} {'DE':>10} {'DOC':>10}")
    print("-" * 50)
    print(f"{'AUROC (per-gene)':<25} {de['auroc_per_gene']:>10.4f} {doc['auroc_per_gene']:>10.4f}")
    print(f"{'AUROC (overall)':<25} {de['auroc_overall']:>10.4f} {doc['auroc_overall']:>10.4f}")
    print(f"{'AUPRC (per-gene)':<25} {de['auprc_per_gene']:>10.4f} {doc['auprc_per_gene']:>10.4f}")
    print(f"{'Accuracy':<25} {de['accuracy']:>10.4f} {doc['accuracy']:>10.4f}")
    print(f"{'F1':<25} {de['f1']:>10.4f} {doc['f1']:>10.4f}")
    print("=" * 50)


# =============================================================================
# SynthPert-style 3-Class Evaluation Metrics
# =============================================================================
# Following SynthPert (https://github.com/qpdcdp/SynthPert)
# 3-class prediction: upregulated, downregulated, not differentially expressed


def evaluate_three_class(
    pred: Union[List[str], np.ndarray],
    true: Union[List[str], np.ndarray],
    labels: Optional[List[str]] = None
) -> Dict:
    """
    Evaluate 3-class perturbation predictions (up/down/not DE).

    Following SynthPert evaluation protocol for direct 3-class prediction.

    Parameters
    ----------
    pred : array-like
        Predicted class labels (strings or integers)
    true : array-like
        Ground truth class labels
    labels : List[str], optional
        List of class labels. Default: ['down', 'not_de', 'up']

    Returns
    -------
    Dict
        Dictionary containing:
        - accuracy: Overall accuracy
        - per_class: Dict with precision/recall/f1 for each class
        - macro_avg: Macro-averaged precision/recall/f1
        - weighted_avg: Weighted-averaged precision/recall/f1
        - confusion_matrix: Confusion matrix as nested dict
    """
    from sklearn.metrics import classification_report, confusion_matrix as sk_confusion_matrix

    if labels is None:
        labels = ['down', 'not_de', 'up']

    pred = np.asarray(pred)
    true = np.asarray(true)

    # Get classification report as dict
    report = classification_report(true, pred, labels=labels, output_dict=True, zero_division=0)

    # Build results
    results = {
        'accuracy': report.get('accuracy', accuracy_score(true, pred)),
        'per_class': {},
        'macro_avg': {
            'precision': report['macro avg']['precision'],
            'recall': report['macro avg']['recall'],
            'f1': report['macro avg']['f1-score']
        },
        'weighted_avg': {
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1': report['weighted avg']['f1-score']
        }
    }

    # Per-class metrics
    for label in labels:
        if label in report:
            results['per_class'][label] = {
                'precision': report[label]['precision'],
                'recall': report[label]['recall'],
                'f1': report[label]['f1-score'],
                'support': int(report[label]['support'])
            }

    # Confusion matrix
    cm = sk_confusion_matrix(true, pred, labels=labels)
    results['confusion_matrix'] = {
        labels[i]: {labels[j]: int(cm[i, j]) for j in range(len(labels))}
        for i in range(len(labels))
    }

    return results


def evaluate_three_class_from_de_doc(
    de_pred: Union[List, np.ndarray],
    doc_pred: Union[List, np.ndarray],
    de_true: Union[List, np.ndarray],
    doc_true: Union[List, np.ndarray],
    de_threshold: float = 0.5,
    doc_threshold: float = 0.5
) -> Dict:
    """
    Convert DE + DOC predictions to 3-class and evaluate.

    Combines binary DE and DOC predictions into 3 classes:
    - 'not_de': DE prediction < threshold
    - 'up': DE >= threshold AND DOC >= threshold
    - 'down': DE >= threshold AND DOC < threshold

    Parameters
    ----------
    de_pred : array-like
        DE prediction scores
    doc_pred : array-like
        DOC prediction scores
    de_true : array-like
        Ground truth DE labels
    doc_true : array-like
        Ground truth DOC labels
    de_threshold : float
        Threshold for DE prediction
    doc_threshold : float
        Threshold for DOC prediction

    Returns
    -------
    Dict
        3-class evaluation metrics
    """
    de_pred = np.asarray(de_pred)
    doc_pred = np.asarray(doc_pred)
    de_true = np.asarray(de_true)
    doc_true = np.asarray(doc_true)

    # Convert to 3-class predictions
    pred_3class = []
    for de_p, doc_p in zip(de_pred, doc_pred):
        if de_p < de_threshold:
            pred_3class.append('not_de')
        elif doc_p >= doc_threshold:
            pred_3class.append('up')
        else:
            pred_3class.append('down')

    # Convert ground truth to 3-class
    true_3class = []
    for de_t, doc_t in zip(de_true, doc_true):
        if de_t == 0:
            true_3class.append('not_de')
        elif doc_t == 1:
            true_3class.append('up')
        else:
            true_3class.append('down')

    return evaluate_three_class(pred_3class, true_3class)


def print_three_class_report(results: Dict) -> None:
    """
    Print formatted 3-class evaluation report.

    Parameters
    ----------
    results : Dict
        Results from evaluate_three_class
    """
    print("\n" + "=" * 60)
    print("3-Class Evaluation Report (SynthPert-style)")
    print("=" * 60)

    print(f"\nOverall Accuracy: {results['accuracy']:.4f}")

    print("\n--- Per-Class Metrics ---")
    print(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 52)
    for label, metrics in results['per_class'].items():
        print(f"{label:<12} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
              f"{metrics['f1']:>10.4f} {metrics['support']:>10}")

    print("-" * 52)
    print(f"{'macro avg':<12} {results['macro_avg']['precision']:>10.4f} "
          f"{results['macro_avg']['recall']:>10.4f} {results['macro_avg']['f1']:>10.4f}")
    print(f"{'weighted avg':<12} {results['weighted_avg']['precision']:>10.4f} "
          f"{results['weighted_avg']['recall']:>10.4f} {results['weighted_avg']['f1']:>10.4f}")

    print("\n--- Confusion Matrix ---")
    labels = list(results['confusion_matrix'].keys())
    print(f"{'':>12}", end='')
    for label in labels:
        print(f"{label:>10}", end='')
    print()

    for true_label in labels:
        print(f"{true_label:<12}", end='')
        for pred_label in labels:
            print(f"{results['confusion_matrix'][true_label][pred_label]:>10}", end='')
        print()

    print("=" * 60)


# =============================================================================
# Cross-Cell-Type Evaluation
# =============================================================================


def build_gene_proportions(
    data: List[Dict],
    gene_key: str = 'gene',
    label_key: str = 'label'
) -> Dict[str, float]:
    """
    Build gene-level proportion scores from training data.

    For each gene, computes the proportion of positive labels across
    all perturbations in the training data.

    Parameters
    ----------
    data : List[Dict]
        Training data with gene and label fields
    gene_key : str
        Key for gene name in data dicts
    label_key : str
        Key for label in data dicts

    Returns
    -------
    Dict[str, float]
        Mapping from gene to proportion (0.0 to 1.0)
    """
    gene_stats = defaultdict(lambda: {'pos': 0, 'total': 0})
    for x in data:
        gene = x[gene_key]
        gene_stats[gene]['total'] += 1
        if x[label_key] == 1:
            gene_stats[gene]['pos'] += 1

    return {
        gene: stats['pos'] / stats['total'] if stats['total'] > 0 else 0.5
        for gene, stats in gene_stats.items()
    }


def cross_cell_type_evaluation(
    train_datasets: Dict[str, Dict],
    test_dataset: Dict,
    dataset_name: str = 'held_out',
    threshold: float = 0.5,
    print_results: bool = True
) -> Dict:
    """
    Evaluate majority vote baseline in cross-cell-type setting.

    Train on data from multiple cell types, test on held-out cell type.

    Parameters
    ----------
    train_datasets : Dict[str, Dict]
        Dictionary mapping dataset names to their data.
        Each value should have 'de' and 'dir' keys with train/test splits.
        Example: {'k562': {'de': {'train': [...], 'test': [...]}, 'dir': {...}}}
    test_dataset : Dict
        Held-out dataset with same structure
    dataset_name : str
        Name of held-out dataset for reporting
    threshold : float
        Threshold for binary predictions
    print_results : bool
        Whether to print results

    Returns
    -------
    Dict
        Evaluation results including accuracy, per-class F1, etc.
    """
    from sklearn.metrics import accuracy_score, classification_report

    # Combine all training data from source cell types
    train_de_all = []
    train_dir_all = []
    for name, data in train_datasets.items():
        if 'de' in data:
            train_de_all.extend(data['de'].get('train', []))
            train_de_all.extend(data['de'].get('test', []))  # Use all data
        if 'dir' in data:
            train_dir_all.extend(data['dir'].get('train', []))
            train_dir_all.extend(data['dir'].get('test', []))

    # Build gene proportions from training cell types
    gene_de_prop = build_gene_proportions(train_de_all)
    gene_dir_prop = build_gene_proportions(train_dir_all)

    # Get test data from held-out cell type (all data)
    test_de = []
    test_dir = []
    if 'de' in test_dataset:
        test_de.extend(test_dataset['de'].get('train', []))
        test_de.extend(test_dataset['de'].get('test', []))
    if 'dir' in test_dataset:
        test_dir.extend(test_dataset['dir'].get('train', []))
        test_dir.extend(test_dataset['dir'].get('test', []))

    # Build dir lookup
    dir_lookup = {
        (x['pert'], x['gene']): (gene_dir_prop.get(x['gene'], 0.5), x['label'])
        for x in test_dir
    }

    # 3-class evaluation
    pred_3class, true_3class = [], []

    for x in test_de:
        key = (x['pert'], x['gene'])
        de_pred = gene_de_prop.get(x['gene'], 0.5)
        de_true = x['label']

        # True label
        if de_true == 0:
            true_3class.append('not_de')
        elif key in dir_lookup:
            _, dir_true = dir_lookup[key]
            true_3class.append('up' if dir_true == 1 else 'down')
        else:
            continue

        # Prediction
        if de_pred < threshold:
            pred_3class.append('not_de')
        elif key in dir_lookup:
            dir_pred, _ = dir_lookup[key]
            pred_3class.append('up' if dir_pred >= threshold else 'down')
        else:
            pred_3class.append('not_de')

    # Compute metrics
    acc = accuracy_score(true_3class, pred_3class)
    report = classification_report(
        true_3class, pred_3class,
        labels=['not_de', 'up', 'down'],
        output_dict=True,
        zero_division=0
    )

    # Gene overlap stats
    test_genes = set(x['gene'] for x in test_de)
    train_genes = set(gene_de_prop.keys())
    gene_overlap = len(test_genes & train_genes)

    results = {
        'dataset': dataset_name,
        'train_sources': list(train_datasets.keys()),
        'accuracy': acc,
        'not_de_f1': report['not_de']['f1-score'],
        'up_f1': report['up']['f1-score'],
        'down_f1': report['down']['f1-score'],
        'macro_f1': report['macro avg']['f1-score'],
        'test_samples': len(test_de),
        'test_genes': len(test_genes),
        'gene_overlap': gene_overlap,
        'gene_overlap_pct': gene_overlap / len(test_genes) if test_genes else 0
    }

    if print_results:
        print(f"\n=== Cross-Cell-Type: {dataset_name.upper()} held out ===")
        print(f"Train on: {', '.join(train_datasets.keys())}")
        print(f"Test samples: {results['test_samples']:,}")
        print(f"Gene overlap: {gene_overlap}/{len(test_genes)} ({results['gene_overlap_pct']:.1%})")
        print(f"\nAccuracy: {acc:.1%}")
        print(f"Not DE F1: {results['not_de_f1']:.4f}")
        print(f"Up F1:     {results['up_f1']:.4f}")
        print(f"Down F1:   {results['down_f1']:.4f}")
        print(f"Macro F1:  {results['macro_f1']:.4f}")

    return results


def run_cross_cell_type_benchmark(
    datasets: Dict[str, Dict],
    print_summary: bool = True
) -> List[Dict]:
    """
    Run cross-cell-type evaluation for all possible held-out configurations.

    Parameters
    ----------
    datasets : Dict[str, Dict]
        Dictionary mapping dataset names to their data
    print_summary : bool
        Whether to print summary table

    Returns
    -------
    List[Dict]
        Results for each held-out configuration
    """
    results = []

    for held_out in datasets.keys():
        train_datasets = {k: v for k, v in datasets.items() if k != held_out}
        test_dataset = datasets[held_out]

        result = cross_cell_type_evaluation(
            train_datasets=train_datasets,
            test_dataset=test_dataset,
            dataset_name=held_out,
            print_results=False
        )
        results.append(result)

    if print_summary:
        print("\n" + "=" * 80)
        print("Cross-Cell-Type Benchmark Results")
        print("=" * 80)
        print(f"{'Held Out':<10} {'Train On':<30} {'Acc':>8} {'NotDE':>8} {'Up':>8} {'Down':>8} {'Macro':>8}")
        print("-" * 80)
        for r in results:
            train_str = ', '.join(r['train_sources'])[:28]
            print(f"{r['dataset'].upper():<10} {train_str:<30} {r['accuracy']:>7.1%} "
                  f"{r['not_de_f1']:>8.4f} {r['up_f1']:>8.4f} {r['down_f1']:>8.4f} {r['macro_f1']:>8.4f}")

        avg_acc = np.mean([r['accuracy'] for r in results])
        avg_macro = np.mean([r['macro_f1'] for r in results])
        print("-" * 80)
        print(f"{'AVERAGE':<10} {'':<30} {avg_acc:>7.1%} {'':<8} {'':<8} {'':<8} {avg_macro:>8.4f}")
        print("=" * 80)

    return results
