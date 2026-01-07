## The Naive Baseline: Majority Vote 
```
python baseline.py \
    --adata "../../VCC2025/vcc_raw_data/rpe1_raw_singlecell_01.h5ad" \
    --dataset_name "rpe1" \
    --control "non-targeting" \
    --target_col "target_gene" \
    --gene_row "gene_name" \
    --alpha 0.01 \
    --n_runs 3
```

1. **Model**: For each gene, learns the majority class from training data (whether that gene is typically DE or not-DE, up or down-regulated across different perturbations)

2. **PerturbQA Evaluation**:

   - Uses pre-defined train/test splits from PerturbQA files
   - Computes AUROC **per gene** (across perturbations for that gene)
   - Averages AUROC across all genes
   - Reports mean ± std over multiple runs with subsampling

3. **Differential Expression Detection**:

   - Use Wilcoxon rank-sum test (Mann-Whitney U test) with Benjamini-Hochberg correction per perturbation
   - 1 = DE, 0 = non-DE

4. **Direction of Change**:

   - Compares mean expression: pert vs control
   - 1 = up-regulated, 0 = down-regulated

`baseline_accuracy.py`: additional metircs calculation + full error analysis. Run after getting DE results (long time to run!).

## Error Analysis

### Overview
Analyzes prediction errors from the majority vote baseline to identify failure patterns:

- By Gene: Which genes are consistently misclassified across perturbations?
- By Perturbation: Which perturbations cause the most prediction errors?
- By Expression: Do errors correlate with gene expression levels, variance, or sparsity?
- Error Types: Are False Positives vs False Negatives driven by different factors?

### Key Features

1. Error Categorization

- True Positives (TP): Correctly predicted positive class
- True Negatives (TN): Correctly predicted negative class
- False Positives (FP): Incorrectly predicted positive
- False Negatives (FN): Incorrectly predicted negative

2. Gene-Level Analysis

Identifies genes that are consistently misclassified across perturbations.

> Key Questions:
Which genes have highest error rates?
Are certain gene families problematic?
Do housekeeping genes perform better?

3. Perturbation-Level Analysis

Identifies perturbations that cause unusual prediction patterns.

> Key Questions:
Which perturbations are hardest to predict?
Do perturbations with strong effects have lower errors?
Are off-target effects causing errors?

4. Expression Pattern Analysis

Correlates prediction errors with gene expression characteristics.

> Key Questions:
Do low-expressed genes have more errors?
Does high variance correlate with errors?
Are zero-inflated genes problematic?

5. Distribution Analysis

Checks for degenerate predictions (e.g., always predicting majority class).

> Warning Signs:
Model only predicts one class
Severe class imbalance in predictions
High error rate on minority class