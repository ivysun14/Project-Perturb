"""
Basic usage example for perturbseq_lang.

Run from the Project-Perturb directory after installing:
    pip install -e .
    python examples/basic_usage.py
"""

from perturbseq_lang import (
    load_gears_dataset,
    find_common_genes,
    process_perturbation_dataset,
    process_perturbation_dataset_zscore_categories,
)

# =============================================================================
# Step 1: Load datasets
# =============================================================================
# Update these paths to where you downloaded the GEARS h5ad files
DATA_DIR = "/Users/yiquntchen/Downloads"

adamson = load_gears_dataset(f"{DATA_DIR}/adamson/perturb_processed.h5ad")
norman = load_gears_dataset(f"{DATA_DIR}/norman/perturb_processed.h5ad")
replogle_k562 = load_gears_dataset(f"{DATA_DIR}/replogle_k562_essential/perturb_processed.h5ad")
replogle_rpe1 = load_gears_dataset(f"{DATA_DIR}/replogle_rpe1_essential/perturb_processed.h5ad")

print(f"Loaded adamson: {adamson.n_obs} cells, {adamson.n_vars} genes")
print(f"Loaded norman: {norman.n_obs} cells, {norman.n_vars} genes")

# =============================================================================
# Step 2: Find common genes across datasets
# =============================================================================
common_genes = find_common_genes([adamson, norman, replogle_k562, replogle_rpe1])
print(f"Found {len(common_genes)} common genes across all datasets")

# =============================================================================
# Step 3: Process into JSONL format (choose one format)
# =============================================================================

# Option A: Direction-based format (INCREASE/DECREASE/NO_CHANGE)
train, test = process_perturbation_dataset(
    adata=adamson,
    output_genes=list(common_genes),
    output_jsonl="output/adamson_direction",
    cell_lines="K562",
    perturbation_type="CRISPRi",
    training_ratio=0.5,
    random_seed=42,
)
print(f"Direction format: {len(train)} training, {len(test)} testing examples")

# Option B: Z-score category format (6 levels of effect size)
train, test = process_perturbation_dataset_zscore_categories(
    adata=adamson,
    output_genes=list(common_genes),
    output_jsonl="output/adamson_zscore",
    cell_lines="K562",
    perturbation_type="CRISPRi",
    training_ratio=0.5,
    random_seed=42,
)
print(f"Z-score format: {len(train)} training, {len(test)} testing examples")

print("\nDone! Check the 'output/' directory for JSONL files.")
