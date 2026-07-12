# Baseline

Gene-based majority vote and k-NN baselines for 3-class DE prediction (up / down / not_de).

## Scripts

### Compute DE/DOC statistics from AnnData

```bash
python DE_DOC_stats.py \
    --adata PATH/TO/ADATA.h5ad \
    --output PATH/TO/OUTPUT.csv \
    --control non-targeting \
    --alpha 0.05 \
    --log2fc_threshold 0.25
```

### Majority vote (PerturbQA pre-split files)

```bash
python majority_perturbQA_DE_file.py \
    --de_file PerturbQA/perturbqa/datasets/hepg2-de.csv \
    --doc_file PerturbQA/perturbqa/datasets/hepg2-dir.csv \
    --dataset_name hepg2 \
    --n_runs 3 \
    --train_fraction 0.8
```

### Majority vote (custom DE/DOC file)

```bash
python majority_DE_DOC_file.py \
    --de_doc_file PATH/TO/DE_DOC_FILE.csv \
    --dataset_name my_dataset \
    --test_size 0.25 \
    --n_runs 3 \
    --train_fraction 0.8 \
    --random_seed 305
```

### Two-step majority vote (DE/not_de then up/down)

```bash
python binary_de_doc_majority/two_step_majority.py \
    --de_doc_file PATH/TO/DE_DOC_FILE.csv \
    --dataset_name my_dataset
```

### Cross-cell-line transfer

```bash
python majority_transfer.py \
    --de_file_train PerturbQA/perturbqa/datasets/hepg2-de.csv \
    --doc_file_train PerturbQA/perturbqa/datasets/hepg2-dir.csv \
    --de_file_test PerturbQA/perturbqa/datasets/rpe1-de.csv \
    --doc_file_test PerturbQA/perturbqa/datasets/rpe1-dir.csv \
    --dataset_name_train hepg2 \
    --dataset_name_test rpe1
```

### k-NN baselines

```bash
# On full datasets (3-class, k=1 and k=10)
python run_knn_full_dataset.py

# On PerturbQA downsampled (binary up/down, k=1,5,10)
python run_knn_perturbqa.py

# Batch cross-cell-line transfer
python run_cross_cellline_transfer.py
python run_cross_cellline_transfer_full.py
python run_cross_cellline_transfer_perturbqa.py

# Plot results
python plot_knn_results.py
```

## Utilities

- **`analysis_utility.py`** -- `train_gene_majority_vote()`, `predict_from_gene_majority()`, `compute_3class_metrics()`, `update_DE_DOC_labels()`, `plot_data_summary()`
- **`error_utility.py`** -- `ErrorAnalyzer` class for error analysis and visualization
- **`train_test_split.py`** -- perturbation-stratified train/test split
- **`process_CD4_TCell.py`** -- process CD4+ T cell DESeq2 results

## Subdirectories

- `binary_de_doc_majority/` -- binary DE/DOC majority vote (original PerturbQA eval protocol)
- `writeup/` -- LaTeX writeup
