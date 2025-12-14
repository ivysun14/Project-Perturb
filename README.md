# PerturbSeq-Lang

A toolkit for processing Perturb-seq single-cell perturbation data into formats suitable for LLM fine-tuning and evaluation.

## Overview

This package provides utilities to:

1. Load preprocessed Perturb-seq datasets (h5ad format)
2. Process perturbation data into various output formats for LLM training
3. Evaluate model predictions against ground truth

## Quick Start

```bash
# 1. Install the package
cd Project-Perturb
pip install -e .

# 2. Run the example script (edit paths first)
python examples/basic_usage.py
```

See `examples/basic_usage.py` for a complete working example.

## Installation

```bash
# Clone and install
cd Project-Perturb
pip install -e .

# Or install with optional dependencies
pip install -e ".[openai,viz]"
```

## Data Sources

The h5ad datasets used by this package can be downloaded from the GEARS repository:

https://github.com/snap-stanford/GEARS/blob/master/gears/pertdata.py#L150-L165

Available datasets include:
- `adamson` - CRISPRi perturbations in K562 cells
- `norman` - CRISPRa perturbations in K562 cells
- `replogle_k562_essential` - Essential gene perturbations in K562
- `replogle_rpe1_essential` - Essential gene perturbations in RPE1
- `dixit` - CRISPRi perturbations

To download, you can use the GEARS package:

```python
from gears import PertData
pert_data = PertData('./data')
pert_data.load(data_name='adamson')
# The h5ad file will be at ./data/adamson/perturb_processed.h5ad
```

## Usage

### 1. Loading Data

```python
from perturbseq_lang import load_gears_dataset, find_common_genes

# Load preprocessed datasets
adamson = load_gears_dataset("/path/to/adamson/perturb_processed.h5ad")
norman = load_gears_dataset("/path/to/norman/perturb_processed.h5ad")

# Find genes common across datasets
common_genes = find_common_genes([adamson, norman])
print(f"Found {len(common_genes)} common genes")
```

### 2. Processing Data (Direction-Based Format)

Outputs INCREASE/DECREASE/NO_CHANGE categories based on statistical significance:

```python
from perturbseq_lang import process_perturbation_dataset

train, test = process_perturbation_dataset(
    adata=adamson,
    output_genes=list(common_genes),
    output_jsonl="output/adamson_direction",
    cell_lines="K562",
    perturbation_type="CRISPRi",
    training_ratio=0.5,
    random_seed=42,
)
```

Output format:
```json
{
  "problem": "Perturbing gene X will lead to what kind of changes in genes A, B, C?",
  "answer": {"INCREASE": ["A"], "DECREASE": ["B"], "NO_CHANGE": ["C"]}
}
```

### 3. Processing Data (Z-Score Categories)

Outputs fine-grained categories based on z-score thresholds:

```python
from perturbseq_lang import process_perturbation_dataset_zscore_categories

train, test = process_perturbation_dataset_zscore_categories(
    adata=adamson,
    output_genes=list(common_genes),
    output_jsonl="output/adamson_zscore",
    cell_lines="K562",
    perturbation_type="CRISPRi",
)
```

Categories:
- STRONG_INCREASE (z >= 3)
- MODERATE_INCREASE (1.5 <= z < 3)
- SLIGHT_INCREASE (0 < z < 1.5)
- SLIGHT_DECREASE (-1.5 < z < 0)
- MODERATE_DECREASE (-3 < z <= -1.5)
- STRONG_DECREASE (z <= -3)

### 4. Gene-Centric Processing

Instead of "what genes does perturbation X affect?", asks "what perturbations affect gene Y?":

```python
from perturbseq_lang import process_gene_centric_perturbation_dataset

train, test = process_gene_centric_perturbation_dataset(
    adata=adamson,
    output_genes=list(common_genes),
    output_jsonl="output/adamson_gene_centric",
    cell_lines="K562",
    perturbation_type="CRISPRi",
    class_balancing=True,  # Balance classes for training
)
```

### 5. Evaluating Predictions

```python
from perturbseq_lang import (
    extract_json_from_output,
    evaluate_set_predictions,
    evaluate_category_predictions,
)

# For direction-based predictions
model_output = "```json\n{\"INCREASE\": [\"A\"], ...}\n```"
predictions = extract_json_from_output(model_output)
results = evaluate_set_predictions(predictions, ground_truth)

# For category-based predictions
results = evaluate_category_predictions(predictions, ground_truth)
```

### 6. Custom Datasets

For datasets not from GEARS:

```python
from perturbseq_lang import load_custom_dataset

# Load with custom column names
adata = load_custom_dataset(
    "/path/to/custom.h5ad",
    condition_col="perturbation",  # Your column name for perturbations
    gene_name_col="symbol",        # Your column name for gene names
    control_label="control",       # Your label for control cells
)
```

For raw count data, see `load_from_raw_counts()` and `preprocess_raw_data()` placeholders in `data_loader.py`.

## Package Structure

```
perturbseq_lang/
    __init__.py        # Package exports
    data_loader.py     # Data loading utilities
    processing.py      # Core processing functions
    evaluation.py      # Evaluation metrics
    utils.py           # Helper functions
```

## Output File Formats

All processing functions output JSONL files with `_training.jsonl` and `_testing.jsonl` suffixes.

### Direction Format
```json
{"problem": "...", "answer": {"INCREASE": [], "DECREASE": [], "NO_CHANGE": []}, "metadata": {}}
```

### Chat Format (Z-Score Categories)
```json
{"messages": [{"role": "user", "content": "..."}], "reference_answer": [{"gene": "X", "zscore": "MODERATE_INCREASE"}]}
```

## Requirements

- Python >= 3.8
- numpy >= 1.21.0
- pandas >= 1.3.0
- scanpy >= 1.9.0
- anndata >= 0.8.0
- scikit-learn >= 1.0.0
