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
    # PerturbQA-style metrics
    auc_per_gene,
    auc_per_perturbation,
    auprc_per_gene,
    compute_binary_metrics,
    compute_perturbqa_metrics,
    evaluate_de_task,
    evaluate_doc_task,
    run_perturbqa_evaluation,
    print_perturbqa_summary,
    # SynthPert-style 3-class metrics
    evaluate_three_class,
    evaluate_three_class_from_de_doc,
    print_three_class_report,
    # Cross-cell-type evaluation
    build_gene_proportions,
    cross_cell_type_evaluation,
    run_cross_cell_type_benchmark,
)
from .utils import (
    clean_perturbation_condition,
    categorize_z_score,
)

__version__ = "0.1.0"
