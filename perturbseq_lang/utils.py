"""
Utility functions for perturbation data processing.
"""

import numpy as np
from typing import List, Dict, Optional


def clean_perturbation_condition(condition: str) -> str:
    """
    Clean and normalize perturbation condition strings.

    Removes control indicators and sorts multi-gene perturbations
    for consistent naming.

    Parameters
    ----------
    condition : str
        Raw condition string (e.g., "GENE1+ctrl", "ctrl+GENE2")

    Returns
    -------
    str
        Cleaned condition string
    """
    condition = condition.replace('+ctrl', '').replace('ctrl+', '')
    return '+'.join(sorted(condition.split('+')))


def categorize_z_score(z: float) -> str:
    """
    Categorize a z-score into discrete categories.

    Categories:
        - STRONG_INCREASE: z >= 3
        - MODERATE_INCREASE: 1.5 <= z < 3
        - SLIGHT_INCREASE: 0 < z < 1.5
        - SLIGHT_DECREASE: -1.5 < z < 0
        - MODERATE_DECREASE: -3 < z <= -1.5
        - STRONG_DECREASE: z <= -3

    Parameters
    ----------
    z : float
        Z-score value

    Returns
    -------
    str
        Category name
    """
    if z >= 3:
        return "STRONG_INCREASE"
    elif 1.5 <= z < 3:
        return "MODERATE_INCREASE"
    elif 0 < z < 1.5:
        return "SLIGHT_INCREASE"
    elif -1.5 < z < 0:
        return "SLIGHT_DECREASE"
    elif -3 < z <= -1.5:
        return "MODERATE_DECREASE"
    elif z <= -3:
        return "STRONG_DECREASE"
    else:
        return "NO_CHANGE"


def categorize_logfc(
    logfc: float,
    pval: float,
    alpha: float = 0.05,
    threshold_strong: float = 1.0,
) -> str:
    """
    Categorize based on log fold change and significance.

    Parameters
    ----------
    logfc : float
        Log2 fold change
    pval : float
        Adjusted p-value
    alpha : float
        Significance threshold
    threshold_strong : float
        Log2FC threshold for strong effects

    Returns
    -------
    str
        Category name
    """
    if pval >= alpha:
        return "No_Significant_Change"
    if logfc > threshold_strong:
        return "Strong_Increase"
    elif logfc < -threshold_strong:
        return "Strong_Decrease"
    elif logfc > 0:
        return "Moderate_Increase"
    else:
        return "Moderate_Decrease"


def logfc_to_percentage(logfc: float) -> float:
    """
    Convert log2 fold change to percentage change.

    Parameters
    ----------
    logfc : float
        Log2 fold change

    Returns
    -------
    float
        Percentage change
    """
    return (2 ** logfc - 1) * 100


def split_train_test(
    items: List,
    train_ratio: float = 0.5,
    random_seed: Optional[int] = None,
) -> tuple:
    """
    Split a list into training and testing sets.

    Parameters
    ----------
    items : List
        Items to split
    train_ratio : float
        Fraction for training set
    random_seed : Optional[int]
        Random seed

    Returns
    -------
    tuple
        (training_items, testing_items)
    """
    np.random.seed(random_seed)
    items = list(items)
    np.random.shuffle(items)
    n_train = int(len(items) * train_ratio)
    return items[:n_train], items[n_train:]


def batch_items(items: List, batch_size: int = 20) -> List[List]:
    """
    Split items into batches.

    Parameters
    ----------
    items : List
        Items to batch
    batch_size : int
        Size of each batch

    Returns
    -------
    List[List]
        List of batches
    """
    return [
        items[i:i + batch_size]
        for i in range(0, len(items), batch_size)
    ]


def format_gene_list(genes: List[str], max_display: int = 10) -> str:
    """
    Format a list of genes for display.

    Parameters
    ----------
    genes : List[str]
        Gene names
    max_display : int
        Maximum genes to show before truncating

    Returns
    -------
    str
        Formatted string
    """
    if len(genes) <= max_display:
        return ", ".join(genes)
    else:
        return ", ".join(genes[:max_display]) + f"... (+{len(genes) - max_display} more)"


def validate_adata_structure(adata) -> Dict[str, bool]:
    """
    Validate that an AnnData object has expected structure.

    Parameters
    ----------
    adata : AnnData
        Dataset to validate

    Returns
    -------
    Dict[str, bool]
        Validation results for each check
    """
    results = {
        "has_condition": "condition" in adata.obs.columns,
        "has_gene_name": "gene_name" in adata.var.columns,
        "has_control": False,
        "has_perturbations": False,
    }

    if results["has_condition"]:
        conditions = adata.obs["condition"].unique()
        # Check for control (various possible labels)
        ctrl_labels = ["ctrl", "control", "CTRL", "Control"]
        results["has_control"] = any(c in conditions for c in ctrl_labels)
        results["has_perturbations"] = len(conditions) > 1

    return results
