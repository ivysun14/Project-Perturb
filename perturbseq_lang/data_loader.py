"""
Data loading utilities for Perturb-seq datasets.

Supports loading preprocessed h5ad files from GEARS and custom datasets.
"""

import scanpy as sc
from anndata import AnnData
from typing import List, Set, Optional, Dict
import numpy as np


def load_gears_dataset(path: str) -> AnnData:
    """
    Load a preprocessed h5ad file from the GEARS repository.

    These datasets can be downloaded from:
    https://github.com/snap-stanford/GEARS/blob/master/gears/pertdata.py

    Available datasets include:
        - adamson: CRISPRi perturbations in K562 cells
        - norman: CRISPRa perturbations in K562 cells
        - replogle_k562_essential: Essential gene perturbations in K562
        - replogle_rpe1_essential: Essential gene perturbations in RPE1
        - dixit: CRISPRi perturbations
        - And others listed in the GEARS repository

    Parameters
    ----------
    path : str
        Path to the perturb_processed.h5ad file.

    Returns
    -------
    AnnData
        Loaded AnnData object with perturbation data.
        Expected structure:
            - adata.obs['condition']: Perturbation condition labels
            - adata.var['gene_name']: Gene names

    Example
    -------
    >>> adamson = load_gears_dataset("/path/to/adamson/perturb_processed.h5ad")
    """
    adata = sc.read_h5ad(path)

    # Validate expected structure
    if 'condition' not in adata.obs.columns:
        raise ValueError(
            "Dataset missing 'condition' column in adata.obs. "
            "Expected GEARS preprocessed format."
        )
    if 'gene_name' not in adata.var.columns:
        raise ValueError(
            "Dataset missing 'gene_name' column in adata.var. "
            "Expected GEARS preprocessed format."
        )

    return adata


def load_custom_dataset(
    path: str,
    condition_col: str = "condition",
    gene_name_col: str = "gene_name",
    control_label: str = "ctrl",
) -> AnnData:
    """
    Load a custom h5ad dataset and standardize column names.

    Use this function when working with datasets that are not from GEARS
    or have different column naming conventions.

    Parameters
    ----------
    path : str
        Path to the h5ad file.
    condition_col : str
        Column name in adata.obs containing perturbation conditions.
    gene_name_col : str
        Column name in adata.var containing gene names.
    control_label : str
        Label used for control cells in the condition column.

    Returns
    -------
    AnnData
        Standardized AnnData object with:
            - adata.obs['condition']: Perturbation conditions
            - adata.var['gene_name']: Gene names

    Example
    -------
    >>> custom_data = load_custom_dataset(
    ...     "/path/to/custom.h5ad",
    ...     condition_col="perturbation",
    ...     gene_name_col="symbol"
    ... )
    """
    adata = sc.read_h5ad(path)

    # Standardize column names
    if condition_col != "condition":
        if condition_col not in adata.obs.columns:
            raise ValueError(f"Column '{condition_col}' not found in adata.obs")
        adata.obs["condition"] = adata.obs[condition_col]

    if gene_name_col != "gene_name":
        if gene_name_col not in adata.var.columns:
            raise ValueError(f"Column '{gene_name_col}' not found in adata.var")
        adata.var["gene_name"] = adata.var[gene_name_col]

    # Validate control group exists
    if control_label not in adata.obs["condition"].unique():
        raise ValueError(
            f"Control label '{control_label}' not found in conditions. "
            f"Available conditions: {adata.obs['condition'].unique()[:10]}..."
        )

    return adata


def load_from_raw_counts(
    counts_path: str,
    obs_path: str,
    var_path: str,
    condition_col: str = "condition",
    gene_name_col: str = "gene_name",
) -> AnnData:
    """
    Placeholder for loading from raw count matrices.

    Use this when you have separate files for counts, cell metadata,
    and gene metadata.

    Parameters
    ----------
    counts_path : str
        Path to count matrix (csv, mtx, etc.)
    obs_path : str
        Path to cell metadata (csv)
    var_path : str
        Path to gene metadata (csv)
    condition_col : str
        Column in obs containing perturbation conditions
    gene_name_col : str
        Column in var containing gene names

    Returns
    -------
    AnnData
        Constructed AnnData object

    TODO: Implement based on specific data formats needed
    """
    raise NotImplementedError(
        "This is a placeholder. Implement based on your specific data format. "
        "See scanpy.read_10x_mtx() or similar functions for guidance."
    )


def preprocess_raw_data(
    adata: AnnData,
    min_genes: int = 200,
    min_cells: int = 3,
    normalize: bool = True,
    log_transform: bool = True,
) -> AnnData:
    """
    Placeholder for preprocessing raw count data.

    Apply standard single-cell preprocessing steps if working with
    unprocessed data.

    Parameters
    ----------
    adata : AnnData
        Raw count data
    min_genes : int
        Minimum genes per cell
    min_cells : int
        Minimum cells per gene
    normalize : bool
        Whether to normalize counts
    log_transform : bool
        Whether to log-transform

    Returns
    -------
    AnnData
        Preprocessed data

    TODO: Customize preprocessing based on your needs
    """
    raise NotImplementedError(
        "This is a placeholder. Standard preprocessing might include:\n"
        "  - sc.pp.filter_cells(adata, min_genes=min_genes)\n"
        "  - sc.pp.filter_genes(adata, min_cells=min_cells)\n"
        "  - sc.pp.normalize_total(adata, target_sum=1e4)\n"
        "  - sc.pp.log1p(adata)\n"
        "Customize based on your specific needs."
    )


def find_common_genes(adata_list: List[AnnData]) -> Set[str]:
    """
    Find genes common across multiple datasets.

    Useful when combining data from multiple experiments for training
    to ensure consistent feature sets.

    Parameters
    ----------
    adata_list : List[AnnData]
        List of AnnData objects to find common genes across.

    Returns
    -------
    Set[str]
        Set of gene names present in all datasets.

    Example
    -------
    >>> datasets = [adamson, norman, replogle_k562]
    >>> common = find_common_genes(datasets)
    >>> print(f"Found {len(common)} common genes")
    """
    if not adata_list:
        return set()

    gene_sets = [set(adata.var['gene_name']) for adata in adata_list]
    common_genes = set.intersection(*gene_sets)

    return common_genes


def subsample_perturbations(
    adata: AnnData,
    max_perturbations: int = 100,
    random_seed: int = 42,
) -> AnnData:
    """
    Subsample an AnnData object to include only a random subset of perturbations.

    Useful for creating smaller test datasets or balancing perturbation counts.

    Parameters
    ----------
    adata : AnnData
        Input data with perturbations in adata.obs['condition']
    max_perturbations : int
        Maximum number of perturbations to keep (plus control)
    random_seed : int
        Random seed for reproducibility

    Returns
    -------
    AnnData
        Subsampled data containing selected perturbations and all control cells
    """
    np.random.seed(random_seed)

    # Clean perturbation names
    def clean_condition(condition: str) -> str:
        condition = condition.replace('+ctrl', '').replace('ctrl+', '')
        return '+'.join(sorted(condition.split('+')))

    adata.obs['clean_condition'] = adata.obs['condition'].apply(clean_condition)

    # Get control and perturbation masks
    control_mask = adata.obs['clean_condition'] == 'ctrl'
    if not np.any(control_mask):
        raise ValueError("No control cells found with label 'ctrl'")

    # Get unique perturbations (excluding control)
    unique_perts = [p for p in adata.obs['clean_condition'].unique() if p != 'ctrl']

    # Sample perturbations
    n_sample = min(len(unique_perts), max_perturbations)
    sampled_perts = np.random.choice(unique_perts, size=n_sample, replace=False)

    # Create mask for selected cells
    selected_mask = adata.obs['clean_condition'].isin(sampled_perts) | control_mask

    return adata[selected_mask].copy()


def get_dataset_summary(adata: AnnData) -> Dict:
    """
    Get a summary of dataset characteristics.

    Parameters
    ----------
    adata : AnnData
        Input dataset

    Returns
    -------
    Dict
        Summary statistics including cell count, gene count,
        perturbation count, etc.
    """
    # Clean conditions for counting
    def clean_condition(condition: str) -> str:
        condition = condition.replace('+ctrl', '').replace('ctrl+', '')
        return '+'.join(sorted(condition.split('+')))

    clean_conditions = adata.obs['condition'].apply(clean_condition)
    unique_perts = [p for p in clean_conditions.unique() if p != 'ctrl']

    n_control = (clean_conditions == 'ctrl').sum()

    return {
        'n_cells': adata.n_obs,
        'n_genes': adata.n_vars,
        'n_perturbations': len(unique_perts),
        'n_control_cells': n_control,
        'perturbations': unique_perts[:10],  # First 10 as sample
    }
