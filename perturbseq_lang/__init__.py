"""
PerturbSeq-Lang: Perturb-seq to Language Model Pipeline

A toolkit for processing and analyzing Perturb-seq perturbation data
for LLM fine-tuning and evaluation.
"""

from .data_loader import (
    load_gears_dataset,
    load_custom_dataset,
    find_common_genes,
)
from .processing import (
    process_perturbation_dataset,
    process_perturbation_dataset_zscore_categories,
    process_gene_centric_perturbation_dataset,
)
from .evaluation import (
    evaluate_direction_predictions,
    evaluate_category_predictions,
    evaluate_set_predictions,
    extract_json_from_output,
)
from .utils import (
    clean_perturbation_condition,
    categorize_z_score,
)

__version__ = "0.1.0"
