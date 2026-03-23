# Majority Vote Baseline

Gene-based majority vote baseline for 3-class differential expression (DE) prediction (up / down / non-DE) in perturb-seq experiments.

## Algorithm

**Training:** for each gene seen in the training set, count how many times it was labelled up (1), down (2), or non-DE (0) across all perturbation–gene pairs. The gene is assigned whichever label appeared most often (ties broken at random).

**Prediction:** for a new perturbation–gene pair, look up the gene's majority label. Genes not seen during training default to non-DE (0).

This is a purely gene-level baseline — it ignores which perturbation is applied and relies entirely on each gene's typical response pattern across the training data.

**DE label assignment** (`DE_DOC_stats.py`): when starting from raw count data, DE labels are derived via Wilcoxon rank-sum test (perturbed vs. non-targeting controls), Benjamini–Hochberg multiple-testing correction, and a log₂ fold-change threshold.

## Scripts

### Compute DE/DOC statistics from AnnData

```bash
python DE_DOC_stats.py \
    --adata PATH/TO/ADATA.h5ad \
    --output PATH/TO/OUTPUT.csv \
    --control non-targeting \
    --alpha 0.01 \
    --log2fc_threshold 0.00
```

### Majority vote — custom DE/DOC file

```bash
python majority_DE_DOC_file.py \
    --de_doc_file PATH/TO/DE_DOC_FILE.csv \
    --dataset_name my_dataset \
    --test_size 0.25 \
    --n_runs 3 \
    --train_fraction 0.8 \
    --random_seed 305
```

### Majority vote — PerturbQA pre-split files

```bash
python majority_perturbQA_DE_file.py \
    --de_file PerturbQA/perturbqa/datasets/hepg2-de.csv \
    --doc_file PerturbQA/perturbqa/datasets/hepg2-dir.csv \
    --dataset_name hepg2 \
    --n_runs 3 \
    --train_fraction 0.8
```

### Two-step majority vote

```bash
python binary_de_doc_majority/two_step_majority.py \
    --de_doc_file PATH/TO/DE_DOC_FILE.csv \
    --dataset_name my_dataset
```

### Cross-cell-line transfer — custom DE/DOC files

```bash
python majority_transfer_full.py \
    --de_doc_file_train ../data/DE_DOC_hepg2_splitted.csv ../data/DE_DOC_jurkat_splitted.csv ../data/DE_DOC_rpe1_splitted.csv \
    --de_doc_file_test ../data/DE_DOC_K562_Replogle_splitted.csv \
    --dataset_name_train hepg2 jurkat rpe1 \
    --dataset_name_test K562_Replogle
```

### Cross-cell-line transfer — PerturbQA files

```bash
python majority_transfer_perturbQA.py \
    --de_file_train ../data/perturbqa_datasets/k562-de.csv ../data/perturbqa_datasets/hepg2-de.csv ../data/perturbqa_datasets/jurkat-de.csv \
    --doc_file_train ../data/perturbqa_datasets/k562-dir.csv ../data/perturbqa_datasets/hepg2-dir.csv ../data/perturbqa_datasets/jurkat-dir.csv \
    --de_file_test ../data/perturbqa_datasets/rpe1-de.csv \
    --doc_file_test ../data/perturbqa_datasets/rpe1-dir.csv \
    --dataset_name_train k562 hepg2 jurkat \
    --dataset_name_test rpe1
```

## Utilities

- **`analysis_utility.py`** — `train_gene_majority_vote()`, `predict_from_gene_majority()`, `compute_3class_metrics()`, `update_DE_DOC_labels()`, `plot_data_summary()`
- **`error_utility.py`** — `ErrorAnalyzer` class for per-gene and per-perturbation error analysis and visualization
- **`train_test_split.py`** — perturbation-stratified train/test split
- **`process_CD4_TCell.py`** — process CD4+ T-cell DESeq2 results into per-condition CSV files

## Subdirectories

- `binary_de_doc_majority/` — two-step binary majority vote (original PerturbQA eval protocol)
