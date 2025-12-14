"""
Core processing functions for Perturb-seq data.

Converts perturbation data into formats suitable for LLM fine-tuning
and evaluation.
"""

import json
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
from anndata import AnnData
from typing import List, Optional, Dict, Tuple
from collections import Counter

from .utils import clean_perturbation_condition, categorize_z_score


def process_perturbation_dataset(
    adata: AnnData,
    output_genes: List[str],
    output_jsonl: Optional[str] = None,
    cell_lines: str = "unspecified",
    perturbation_type: str = "unspecified",
    training_ratio: float = 0.5,
    random_seed: Optional[int] = None,
    min_cells_per_group: int = 200,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Process perturbation data into INCREASE/DECREASE/NO_CHANGE format.

    This function performs differential expression analysis between
    each perturbation and control, then categorizes genes based on
    statistical significance and direction of change.

    Parameters
    ----------
    adata : AnnData
        Input dataset with perturbation data
    output_genes : List[str]
        List of genes to include in the output
    output_jsonl : Optional[str]
        Base path for output files (will create _training.jsonl and _testing.jsonl)
    cell_lines : str
        Cell line description for prompts (e.g., "K562")
    perturbation_type : str
        Perturbation type for prompts (e.g., "CRISPRi")
    training_ratio : float
        Fraction of perturbations to use for training
    random_seed : Optional[int]
        Random seed for reproducibility
    min_cells_per_group : int
        Minimum cells required per perturbation group

    Returns
    -------
    Tuple[List[Dict], List[Dict]]
        Training and testing result lists
    """
    if output_genes is None:
        raise ValueError("You must provide a list of `output_genes` to process.")

    np.random.seed(random_seed)
    output_genes_set = set(output_genes)

    # Clean perturbation conditions
    adata.obs['clean_condition'] = adata.obs['condition'].apply(
        clean_perturbation_condition
    )

    if 'ctrl' not in adata.obs['clean_condition'].unique():
        raise ValueError("Dataset must include control group ('ctrl')")

    # Get control cells
    control_mask = adata.obs['clean_condition'] == 'ctrl'
    adata_control = adata[control_mask]

    # Get perturbation groups
    perturbation_groups = adata.obs['clean_condition'].unique()
    perturbation_groups = perturbation_groups[perturbation_groups != 'ctrl']

    # Subset to output genes
    adata = adata[:, adata.var['gene_name'].isin(output_genes_set)]

    # Split into training/testing
    n_train = int(len(perturbation_groups) * training_ratio)
    np.random.shuffle(perturbation_groups)
    training_groups = set(perturbation_groups[:n_train])

    training_results = []
    testing_results = []

    for group in perturbation_groups:
        perturbation_mask = adata.obs['clean_condition'] == group
        adata_group = adata[perturbation_mask]

        # Skip small groups
        if adata_group.shape[0] <= min_cells_per_group:
            continue

        # Differential expression analysis
        adata_subset = adata_control.concatenate(adata_group)
        sc.tl.rank_genes_groups(
            adata_subset,
            groupby='clean_condition',
            method='wilcoxon',
            use_raw=False,
            corr_method='benjamini-hochberg'
        )

        # Extract results
        gene_names = adata.var['gene_name']
        pvals = adata_subset.uns['rank_genes_groups']['pvals_adj'][group]
        logfoldchanges = adata_subset.uns['rank_genes_groups']['logfoldchanges'][group]

        # Process in batches of 20 genes
        for i in range(0, len(gene_names), 20):
            genes_batch = gene_names[i:i + 20]
            group_results = {"INCREASE": [], "DECREASE": [], "NO_CHANGE": []}

            for gene, pval, logfc in zip(
                genes_batch,
                pvals[i:i + 20],
                logfoldchanges[i:i + 20]
            ):
                if gene in output_genes_set:
                    if pval < 0.05:
                        pct_change = (2 ** logfc - 1) * 100
                        if pct_change > 0:
                            group_results["INCREASE"].append(gene)
                        else:
                            group_results["DECREASE"].append(gene)
                    else:
                        group_results["NO_CHANGE"].append(gene)

            question = (
                f"Use Biological knowledge on human genomics to answer the following "
                f"question in the context of a Perturb-seq experiment conducted on "
                f"{cell_lines} cell lines sampled at post-{perturbation_type}-transduction.\n"
                f"Perturbing gene {group} will lead to what kind of changes in the "
                f"following genes: {', '.join(genes_batch)}?\n"
                f"Provide your answer in JSON dictionary format using 'INCREASE', "
                f"'DECREASE', and 'NO_CHANGE' as the keys and for the values a list "
                f"of the gene names that belong in each category."
            )

            result_entry = {
                "problem": question,
                "answer": group_results,
                "metadata": {}
            }

            if group in training_groups:
                training_results.append(result_entry)
            else:
                testing_results.append(result_entry)

    # Save to JSONL files
    if output_jsonl:
        _save_jsonl(training_results, f"{output_jsonl}_training.jsonl")
        _save_jsonl(testing_results, f"{output_jsonl}_testing.jsonl")

    return training_results, testing_results


def process_perturbation_dataset_zscore_categories(
    adata: AnnData,
    output_genes: List[str],
    output_jsonl: Optional[str] = None,
    cell_lines: str = "unspecified",
    perturbation_type: str = "unspecified",
    training_ratio: float = 0.5,
    random_seed: Optional[int] = None,
    min_cells_per_group: int = 200,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Process perturbation data with z-score based categories.

    Categories are assigned based on z-score thresholds:
        - STRONG_INCREASE: z >= 3
        - MODERATE_INCREASE: 1.5 <= z < 3
        - SLIGHT_INCREASE: 0 < z < 1.5
        - SLIGHT_DECREASE: -1.5 < z < 0
        - MODERATE_DECREASE: -3 < z <= -1.5
        - STRONG_DECREASE: z <= -3

    Parameters
    ----------
    adata : AnnData
        Input dataset
    output_genes : List[str]
        Genes to include in output
    output_jsonl : Optional[str]
        Base path for output files
    cell_lines : str
        Cell line description
    perturbation_type : str
        Perturbation type description
    training_ratio : float
        Fraction for training set
    random_seed : Optional[int]
        Random seed
    min_cells_per_group : int
        Minimum cells per perturbation

    Returns
    -------
    Tuple[List[Dict], List[Dict]]
        Training and testing results
    """
    if output_genes is None:
        raise ValueError("You must provide a list of `output_genes` to process.")

    np.random.seed(random_seed)
    output_genes_set = set(output_genes)

    adata.obs['clean_condition'] = adata.obs['condition'].apply(
        clean_perturbation_condition
    )

    if 'ctrl' not in adata.obs['clean_condition'].unique():
        raise ValueError("Dataset must include control group ('ctrl')")

    control_mask = adata.obs['clean_condition'] == 'ctrl'
    adata_control = adata[control_mask]

    perturbation_groups = adata.obs['clean_condition'].unique()
    perturbation_groups = perturbation_groups[perturbation_groups != 'ctrl']

    adata = adata[:, adata.var['gene_name'].isin(output_genes_set)]

    n_train = int(len(perturbation_groups) * training_ratio)
    np.random.shuffle(perturbation_groups)
    training_groups = set(perturbation_groups[:n_train])

    training_results = []
    testing_results = []

    for group in perturbation_groups:
        perturbation_mask = adata.obs['clean_condition'] == group
        adata_group = adata[perturbation_mask]

        if adata_group.shape[0] <= min_cells_per_group:
            continue

        adata_subset = adata_control.concatenate(adata_group)
        sc.tl.rank_genes_groups(
            adata_subset,
            groupby='clean_condition',
            method='wilcoxon',
            use_raw=False,
            corr_method='benjamini-hochberg'
        )

        gene_names = adata.var['gene_name']
        scores = adata_subset.uns['rank_genes_groups']['scores'][group]
        z_scores = (scores - np.mean(scores)) / np.std(scores)

        # Shuffle genes
        indices = np.random.permutation(len(gene_names))
        gene_names = gene_names[indices]
        z_scores = z_scores[indices]

        # Process in batches
        for i in range(0, len(gene_names), 20):
            genes_batch = gene_names[i:i + 20]
            z_batch = z_scores[i:i + 20]

            categorized = [
                {"gene": gene, "zscore": categorize_z_score(z)}
                for gene, z in zip(genes_batch, z_batch)
            ]

            example = {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Use Biological knowledge on human genomics to answer the "
                            f"following prediction question in the context of a Perturb-seq "
                            f"experiment conducted on {cell_lines} cell lines sampled at "
                            f"post-{perturbation_type}-transduction.\n"
                            f"Perturbing gene {group} will lead to what kind of effects on "
                            f"the following genes?\n"
                            f"Categories (by z-score):\n"
                            f"1. STRONG_INCREASE: >3\n"
                            f"2. MODERATE_INCREASE: 1.5-3\n"
                            f"3. SLIGHT_INCREASE: 0-1.5\n"
                            f"4. SLIGHT_DECREASE: -1.5-0\n"
                            f"5. MODERATE_DECREASE: -3 to -1.5\n"
                            f"6. STRONG_DECREASE: <-3\n"
                            f"Respond with a JSON array where each element has 'gene' and "
                            f"'zscore' keys.\n"
                            f"Genes to analyze: {', '.join(genes_batch)}."
                        )
                    }
                ],
                "reference_answer": categorized
            }

            if group in training_groups:
                training_results.append(example)
            else:
                testing_results.append(example)

    if output_jsonl:
        _save_jsonl(training_results, f"{output_jsonl}_training.jsonl")
        _save_jsonl(testing_results, f"{output_jsonl}_testing.jsonl")

    return training_results, testing_results


def process_gene_centric_perturbation_dataset(
    adata: AnnData,
    output_genes: List[str],
    output_jsonl: Optional[str] = None,
    cell_lines: str = "unspecified",
    perturbation_type: str = "unspecified",
    training_ratio: float = 0.5,
    random_seed: Optional[int] = None,
    class_balancing: bool = False,
    min_cells_per_group: int = 200,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Process data with a gene-centric view (what perturbations affect each gene).

    Instead of asking "what genes does perturbation X affect?", this asks
    "what perturbations affect gene Y?".

    Parameters
    ----------
    adata : AnnData
        Input dataset
    output_genes : List[str]
        Genes to analyze
    output_jsonl : Optional[str]
        Base output path
    cell_lines : str
        Cell line description
    perturbation_type : str
        Perturbation type
    training_ratio : float
        Training set fraction
    random_seed : Optional[int]
        Random seed
    class_balancing : bool
        Whether to balance classes by category
    min_cells_per_group : int
        Minimum cells per perturbation

    Returns
    -------
    Tuple[List[Dict], List[Dict]]
        Training and testing results
    """
    if output_genes is None:
        raise ValueError("You must provide a list of `output_genes` to process.")

    np.random.seed(random_seed)
    output_genes_set = set(output_genes)

    adata.obs['clean_condition'] = adata.obs['condition'].apply(
        clean_perturbation_condition
    )

    if 'ctrl' not in adata.obs['clean_condition'].unique():
        raise ValueError("Dataset must include control group ('ctrl')")

    control_mask = adata.obs['clean_condition'] == 'ctrl'
    adata_control = adata[control_mask]

    perturbation_groups = list(adata.obs['clean_condition'].unique())
    perturbation_groups = [g for g in perturbation_groups if g != 'ctrl']

    # Split perturbations into train/test
    np.random.shuffle(perturbation_groups)
    n_train = int(len(perturbation_groups) * training_ratio)
    training_perturbations = set(perturbation_groups[:n_train])

    adata = adata[:, adata.var['gene_name'].isin(output_genes_set)]

    training_results = []
    testing_results = []

    def categorize_gene(logfc: float, pval: float, alpha: float = 0.05) -> str:
        """Categorize based on log fold change and significance."""
        if pval >= alpha:
            return "No_Significant_Change"
        if logfc > 1.0:
            return "Strong_Increase"
        elif logfc < -1.0:
            return "Strong_Decrease"
        elif logfc > 0:
            return "Moderate_Increase"
        else:
            return "Moderate_Decrease"

    # Collect results per gene
    gene_results = {gene: {} for gene in output_genes}

    for group in perturbation_groups:
        perturbation_mask = adata.obs['clean_condition'] == group
        adata_group = adata[perturbation_mask]

        if adata_group.shape[0] <= min_cells_per_group:
            continue

        adata_subset = anndata.concat([adata_control, adata_group])

        sc.tl.rank_genes_groups(
            adata_subset,
            groupby='clean_condition',
            method='t-test',
            use_raw=False,
            corr_method='benjamini-hochberg'
        )

        ranked_genes = adata_subset.uns['rank_genes_groups']['names'][group]
        logfcs = adata_subset.uns['rank_genes_groups']['logfoldchanges'][group]
        pvals = adata_subset.uns['rank_genes_groups']['pvals_adj'][group]

        for gene, logfc, pval in zip(ranked_genes, logfcs, pvals):
            if gene in output_genes_set:
                category = categorize_gene(logfc, pval)
                gene_results[gene][group] = {
                    "logFC": logfc,
                    "pval": pval,
                    "category": category,
                }

    # Create examples
    for gene, perturbations in gene_results.items():
        for pert, details in perturbations.items():
            example = {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"In the context of a Perturb-seq experiment on {cell_lines} "
                            f"cell lines post-{perturbation_type}-transduction, what is the "
                            f"predicted effect of perturbation '{pert}' on gene '{gene}'?\n"
                            f"Categories:\n"
                            f"1. Strong_Increase\n"
                            f"2. Moderate_Increase\n"
                            f"3. No_Significant_Change\n"
                            f"4. Moderate_Decrease\n"
                            f"5. Strong_Decrease\n"
                            f"Answer as JSON with 'logFC', 'category' keys."
                        ),
                    }
                ],
                "reference_answer": [
                    {
                        "logFC": str(round(details["logFC"], 2)),
                        "category": details["category"],
                    }
                ],
            }

            if pert in training_perturbations:
                training_results.append(example)
            else:
                testing_results.append(example)

    # Optional class balancing
    if class_balancing:
        training_results = _balance_classes(training_results, random_seed)

    if output_jsonl:
        _save_jsonl(training_results, f"{output_jsonl}_training.jsonl")
        _save_jsonl(testing_results, f"{output_jsonl}_testing.jsonl")

    return training_results, testing_results


def _balance_classes(
    results: List[Dict],
    random_seed: Optional[int] = None
) -> List[Dict]:
    """Balance dataset by sampling equally from each category."""
    np.random.seed(random_seed)

    category_samples = {}
    for result in results:
        ref = result["reference_answer"][0]
        cat = ref["category"]
        if cat not in category_samples:
            category_samples[cat] = []
        category_samples[cat].append(result)

    # Find minimum class size
    min_size = min(len(samples) for samples in category_samples.values() if samples)

    # Sample from each category
    balanced = []
    for samples in category_samples.values():
        if samples:
            indices = np.random.choice(len(samples), min_size, replace=False)
            balanced.extend([samples[i] for i in indices])

    return balanced


def _save_jsonl(data: List[Dict], path: str) -> None:
    """Save list of dicts to JSONL file."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def combine_datasets(
    jsonl_files: List[str],
    output_path: str,
    shuffle: bool = True,
    random_seed: Optional[int] = None,
    max_samples: Optional[int] = None,
) -> int:
    """
    Combine multiple JSONL files into a single file.

    Parameters
    ----------
    jsonl_files : List[str]
        List of JSONL file paths to combine
    output_path : str
        Output file path
    shuffle : bool
        Whether to shuffle the combined data
    random_seed : Optional[int]
        Random seed for shuffling
    max_samples : Optional[int]
        Maximum number of samples to include

    Returns
    -------
    int
        Number of samples written
    """
    np.random.seed(random_seed)

    all_data = []
    for filepath in jsonl_files:
        with open(filepath, 'r') as f:
            for line in f:
                all_data.append(json.loads(line))

    if shuffle:
        np.random.shuffle(all_data)

    if max_samples is not None:
        all_data = all_data[:max_samples]

    _save_jsonl(all_data, output_path)

    return len(all_data)
