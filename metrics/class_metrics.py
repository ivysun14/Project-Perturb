"""
class_metrics.py
================
Class-specific metrics for the 3-class DE prediction task (0 = non-DE, 1 = up, 2 = down), computed
per (perturbation, gene) pair from predicted vs reference labels.
"""

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def compute_3class_metrics(y_true: np.ndarray,
                           y_pred: np.ndarray,
                           y_proba: np.ndarray = None) -> Dict[str, float]:
    """
    Compute classification metrics for 3-class problem.

    Computes overall and per-class metrics for:
    - Class 0: Non-DE
    - Class 1: Up-regulated
    - Class 2: Down-regulated

    Args:
        y_true:  True labels (0, 1, or 2)
        y_pred:  Predicted labels (0, 1, or 2)
        y_proba: Optional predicted probabilities of shape (n_samples, 3).
                 If provided, One-vs-Rest AUROC metrics will be computed.
    Returns:
        Dict with keys:
            - accuracy: Overall accuracy
            - balanced_accuracy: Balanced accuracy (accounts for class imbalance)
            - precision_macro: Macro-averaged precision
            - recall_macro: Macro-averaged recall
            - f1_macro: Macro-averaged F1 score
            - precision_class_{i}: Precision for class i (i=0,1,2)
            - recall_class_{i}: Recall for class i (i=0,1,2) (aka. accuracy)
            - f1_class_{i}: F1 score for class i (i=0,1,2)
            - auroc_ovr_class_{i}: One-vs-Rest AUROC for class i (if y_proba provided)
            - auroc_ovr_macro: Macro-average One-vs-Rest AUROC (if y_proba provided)
    """

    # Overall metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0)
    }

    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    for class_idx in range(3):
        metrics[f'precision_class_{class_idx}'] = precision_per_class[class_idx]
        metrics[f'recall_class_{class_idx}'] = recall_per_class[class_idx]
        metrics[f'f1_class_{class_idx}'] = f1_per_class[class_idx]

    # Compute One-vs-Rest AUROC if probabilities are provided
    if y_proba is not None:
        auroc_per_class = []
        for class_idx in range(3):
            y_true_binary = (y_true == class_idx).astype(int)
            y_score = y_proba[:, class_idx]
            # Only compute AUROC if both classes are present
            if len(np.unique(y_true_binary)) > 1:
                auroc = roc_auc_score(y_true_binary, y_score)
                metrics[f'auroc_ovr_class_{class_idx}'] = auroc
                auroc_per_class.append(auroc)
            else:
                # If only one class present, AUROC is undefined
                metrics[f'auroc_ovr_class_{class_idx}'] = np.nan
        if auroc_per_class:
            metrics['auroc_ovr_macro'] = np.mean(auroc_per_class)
        else:
            metrics['auroc_ovr_macro'] = np.nan

    return metrics
