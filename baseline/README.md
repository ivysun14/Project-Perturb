# Baseline Models for PerturbQA

This directory contains baseline models and analysis tools for the PerturbQA benchmark.

## Quick Start

```python
from perturbqa import load_de, load_dir, auc_per_gene

data_de = load_de("rpe1")
X_train, X_test = data_de["train"], data_de["test"]

keys = [(x["pert"], x["gene"]) for x in X_test]
true = [x["label"] for x in X_test]
# 0: not DE, 1: DE

pred = np.random.randn(len(keys))
auc = auc_per_gene(keys, pred, true)
```

## Scripts

### 1. Gene-Level Majority Vote (`baseline.py`)

```bash
python baseline.py \
    --adata "path/to/data.h5ad" \
    --dataset_name "rpe1" \
    --control "non-targeting" \
    --alpha 0.05 \
    --n_runs 3
```

- Predicts majority class for each gene from training data
- Uses Wilcoxon rank-sum test with BH correction for DE
- Computes AUROC per gene, averaged across genes

### 2. Enhanced Baseline (`enhanced_baseline.py`)

Adds features: gene/pert priors, embedding similarity, pathway similarity.

### 3. Embedding-Weighted Direction (`improve_down_precision.py`)

Addresses Down class precision gap using:
- **Embedding-weighted retrieval**: Weights predictions by GenePT embedding similarity
- **Confidence thresholding**: Only predicts Down when clearly confident
- **Cost-sensitive learning**: Custom class weights for imbalanced data

```python
# Key idea: weight training examples by embedding similarity
weighted_dir = np.average(labels, weights=embedding_similarities)
```

### 4. Cross-Dataset Analysis (`perturbation_profile_analysis.py`)

- **Nearest Neighbor Analysis**: Do same perturbations cluster across datasets?
- **Effect Consistency**: How reproducible are perturbation effects?
- **kNN Transfer**: Predict effects using weighted average of similar perturbations

```python
from perturbation_profile_analysis import run_profile_analysis
results = run_profile_analysis("shared_pairs.csv", "k562", "rpe1")
```

### 5. Concordance Analysis (`concordance_analysis.py`)

Cross-dataset agreement on DE calls and direction.

### 6. LLM Benchmarking (`llm_benchmark.py`)

Evaluates LLM predictions against ground truth.

### 7. Neighbor-Based Baselines (`run_neighbor_baselines.py`)

Systematic evaluation of neighbor-based voting methods:

| Method | Description |
|--------|-------------|
| `majority_gene` | Per-gene majority vote (ignores perturbation) |
| `majority_pert` | Per-perturbation majority vote (ignores gene) |
| `pert_neighbor_*` | Find similar perturbations, vote on target gene |
| `gene_neighbor_*` | Find similar target genes, vote on response |
| `both_neighbor_*` | Combine perturbation and gene neighbors |
| `ensemble` | Combine all neighbor sources |
| `two_stage_*` | First predict DE/not_de, then up/down |

Neighbor similarity sources: **ESM2** (protein), **GenePT** (gene description), **Co-pathway** (KEGG/GO)

### 8. DE/DOC Statistics (`DE_DOC_stats.py`)

Compute differential expression statistics and direction of change for perturbation-gene pairs in a given AnnData object.
```bash
python -u DE_DOC_stats.py \
  --adata "PATH/TO/ADATA.h5ad" \
  --output "PATH/TO/SAVE/OUTPUT.CSV" \
  --alpha 0.01 \
  --log2fc_threshold 0.00
```

### 9. Train/Test Split (`train_test_split.py`)

Perform a 75/25 percent train test split on outputs from `DE_DOC_stats.py`. The split is based on **perturbations** and ensure label proportions are balanced between train and test set. [Currently have to manually change inputs in the actual script.]
```bash
python -u train_test_split.py
```

### 10. Majority Vote on PerturbQA DE Files (`majority_perturbQA_DE_file.py`)

For use with PerturbQA pre-split DE and DOC files. Train a majority vote predictor using the pre-split 75% training set, then evaluate on the 25% testing set.
```bash
python -u majority_perturbQA_DE_file.py \
    --de_file "PerturbQA/perturbqa/datasets/hepg2-de.csv" \
    --doc_file "PerturbQA/perturbqa/datasets/hepg2-dir.csv" \
    --dataset_name hepg2 \
    --n_runs 3 \
    --train_fraction 0.8
```

### 11. Majority Vote on Custom DE Files (`majority_DE_DOC_file.py`)

For use with combined DE+DOC files from `DE_DOC_stats.py` and have gone through train test split using `train_test_split.py`. Train a majority vote predictor on the pre-split training set and evaluate on the test set.
```bash
python -u majority_DE_DOC_file.py \
    --de_doc_file "PATH/TO/DE_DOC_FILE.csv" \
    --dataset_name "THIS_WOULD_BE_THE_PREFIX_FOR_ALL_OUTPUTS"
    --test_size 0.25 \
    --n_runs 3 \
    --train_fraction 0.8 \
    --random_seed 305
```

### Utilities

#### `analysis_utility.py`
Utility functions for model training and DE/DOC label updating.

- `resolve_repeats`: preprocessing function for an AnnData object. If there are repeatedly measured genes in an AnnData object take column-wise averages for such genes
- `update_DE_DOC_labels`: update labels (0/1/2) for perturbation-gene pairs using updated alpha and log2FC thresholds
- `filter_perturbqa_style`: filter out noisy perturbation-gene pairs based on workflow proposed in PerturbQA, accepts files outputted from `DE_DOC_stats.py` as input
- `train_gene_majority_vote` and `predict_from_gene_majority`: gene-based majority vote baseline training and predicting functions
- `compute_3class_metrics`: compute overall and per-class metrics for the 3-class classification problem
- `plot_data_summary`: plotting function that plots train/test splits' class distributions and perturbation-gene pair information

#### `error_utility.py`
Utility class for error analysis.

## Results

All results are on the **FULL DATA** (same train/test splits).

### Dataset Sizes

| Dataset | Train | Test | % not_de |
|---------|-------|------|----------|
| Adamson | 258K | 91K | 78.5% |
| Replogle K562 | 3.27M | 1.09M | 90.8% |
| Replogle RPE1 | 4.63M | 1.55M | 90.1% |
| HepG2 | 13.8M | 4.62M | 96.0% |
| Jurkat | 12.7M | 4.26M | 97.4% |
| T-cell rest | 67.0M | 22.3M | 99.5% |

---

### Majority Vote Baseline

| Dataset | Accuracy | Macro F1 | Up F1 | Down F1 | not_de F1 |
|---------|----------|----------|-------|---------|-----------|
| **Adamson** | 0.884 | **0.773** | 0.639 | 0.749 | 0.933 |
| **Replogle K562** | 0.908 | 0.333 | 0.014 | 0.034 | 0.952 |
| **Replogle RPE1** | 0.905 | 0.501 | 0.195 | 0.357 | 0.950 |
| **HepG2** | 0.960 | 0.327 | 0.000 | 0.000 | 0.980 |
| **Jurkat** | 0.974 | 0.329 | 0.000 | 0.000 | 0.987 |
| **T-cell rest** | 0.995 | 0.332 | 0.000 | 0.000 | 0.997 |

**Note**: Severe class imbalance causes majority vote to predict only `not_de` for HepG2/Jurkat/T-cell (F1=0 for minorities).

---

### Neighbor-Based Methods (Same Data, Better Methods)

#### Best Method per Dataset (3-class: up/down/not_de)

| Dataset | Best Method | Macro F1 | Up F1 | Down F1 |
|---------|-------------|----------|-------|---------|
| **Adamson** | majority_gene | **0.773** | 0.639 | 0.749 |
| **Replogle K562** | pert_neighbor_copath | **0.553** | 0.343 | 0.373 |
| **Replogle RPE1** | pert_neighbor_copath | **0.581** | 0.360 | 0.442 |
| **HepG2** | pert_neighbor_copath | **0.517** | 0.265 | 0.310 |
| **Jurkat** | pert_neighbor_copath | **0.453** | 0.180 | 0.198 |
| **T-cell rest** | pert_neighbor_copath | **0.353** | 0.034 | 0.030 |

#### Average Performance Across All 6 Datasets

| Method | Avg Macro F1 | Avg Up F1 | Avg Down F1 |
|--------|--------------|-----------|-------------|
| **pert_neighbor_copath** | **0.537** | 0.302 | 0.349 |
| ensemble | 0.526 | 0.283 | 0.334 |
| two_stage_esm2 | 0.503 | 0.254 | 0.302 |
| pert_neighbor_esm2 | 0.496 | 0.243 | 0.288 |
| pert_neighbor_genept | 0.481 | 0.219 | 0.259 |
| majority_gene | 0.433 | 0.141 | 0.190 |
| majority_pert | 0.319 | 0.000 | 0.000 |

---

### Key Findings

1. **Co-pathway neighbors are most informative**: `pert_neighbor_copath` wins on 5/6 datasets
2. **Perturbation neighbors >> Gene neighbors**: Gene neighbors perform no better than majority voting
3. **Adamson is qualitatively different**: Only dataset where gene-level patterns are strong (Macro F1: 0.77)
4. **Class imbalance limits performance**: As not_de fraction increases (78.5% Adamson → 99.5% T-cell), minority class detection degrades
5. **Neighbor methods help with imbalance**: Even on highly imbalanced data, neighbor methods achieve non-zero F1 for minorities where majority vote fails

### Detailed Results

- `output/baseline_results/` - Full data majority vote results
- `output/neighbor_baseline_results/summary.md` - Downsampled neighbor method results

## Output

Results saved to `output/`:
- `baseline_results/` - **FULL DATA** majority vote results (JSON + MD)
- `neighbor_baseline_results/` - **DOWNSAMPLED** neighbor method results (JSON + MD)
- `concordance/` - Cross-dataset results
- `llm_benchmark/` - LLM evaluation results
