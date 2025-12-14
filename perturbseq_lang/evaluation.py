"""
Evaluation utilities for perturbation prediction models.

Provides functions to score model predictions against ground truth
and compare different prediction strategies.
"""

import json
import re
from typing import List, Dict, Optional, Tuple
from collections import Counter


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
