# Perturbation Majority Vote

Assign the most frequent label (up / down / non-DE) to a gene based on its dominant behavior in the training set. Ignore perturbation identity.

## Experiments

### 1. Full dataset — within-cell-line

Train on the 60% train split, report performance on the 40% test set within the same cell line.

```
python majority_DE_DOC_file.py \
    --de_doc_file  ${DATA_DIR}/DE_DOC_${DATASET}_splitted.csv \
    --dataset_name ${DATASET} \
    --output_dir   DE_full_within_expr/results
```

### 2. Full dataset — cross-cell-line

Train using the `train` and `test` splits of 3 non-target cell lines, report performance on the 40% test set of the target cell line.

```
# example: held-out cell line = K562, trained on the other 3
python majority_transfer_full.py \
    --de_doc_file_train  ${DATA_DIR}/DE_DOC_RPE1_splitted.csv ${DATA_DIR}/DE_DOC_HepG2_splitted.csv ${DATA_DIR}/DE_DOC_Jurkat_splitted.csv \
    --de_doc_file_test   ${DATA_DIR}/DE_DOC_K562_splitted.csv \
    --dataset_name_train RPE1 HepG2 Jurkat \
    --dataset_name_test  K562 \
    --output_dir DE_full_cross_expr/results

# recompute metrics after filtering to the common gene universe and dropping low-coverage perturbations
python metric_drop_perts.py
```

- `{TestDS}_transfer_unseen.json` records genes present in the held-out cell line's test set but not in any of the 3 source cell lines' training data.

### 3. PerturbQA — within-cell-line

Train on the PerturbQA train split, report performance on the PerturbQA test split within the same cell line.

```
python majority_perturbQA_DE_file.py \
    --de_file       ${PQADATA}/k562-de.csv \
    --dir_file      ${PQADATA}/k562-dir.csv \
    --dataset_name  K562 \
    --output_dir    DE_PerturbQA_within_expr/results
```

### 4. PerturbQA — cross-cell-line

Train using all (perturbation, gene) pairs from 3 non-target cell lines restricted to the common gene universe, report performance on the PerturbQA test split of the target cell line.

```
# example: held-out cell line = K562, trained on the other 3
python majority_transfer_perturbQA.py \
    --de_file_train        ${PQADATA}/rpe1-de.csv ${PQADATA}/hepg2-de.csv ${PQADATA}/jurkat-de.csv \
    --dir_file_train       ${PQADATA}/rpe1-dir.csv ${PQADATA}/hepg2-dir.csv ${PQADATA}/jurkat-dir.csv \
    --de_file_test         ${PQADATA}/k562-de.csv \
    --dir_file_test        ${PQADATA}/k562-dir.csv \
    --gene_universe_file   ${GENE_UNIVERSE} \
    --dataset_name_train   RPE1 HepG2 Jurkat \
    --dataset_name_test    K562 \
    --output_dir           DE_PerturbQA_cross_expr/results

# recompute metrics after additionally dropping low-coverage test perturbations
python metric_drop_perts.py
```

## Utilities

- **`analysis_utility.py`** — `train_gene_majority_vote()`, `predict_from_gene_majority()`, `compute_3class_metrics()`, `plot_data_summary()`
- **`error_utility.py`** — `run_error_analysis_from_results()` and `ErrorAnalyzer` class for per-gene and per-perturbation error analysis
- **`metric_drop_perts.py`** — recomputes cross-cell-line metrics after gene universe and pert-drop filtering for both full (`DE_full_cross_expr/`) and PerturbQA (`DE_PerturbQA_cross_expr/`) experiments; outputs `cross_majority_metrics_filtered.csv` in each experiment directory
- **`DE_DOC_stats.py`** — compute DE labels from raw AnnData via Wilcoxon rank-sum test (BH correction, adj. p < 0.01)
- **`train_test_split.py`** — perturbation-stratified train/test split
- **`calibration_split.py`** — adds a `split_calib` column to an existing DE/DOC splitted CSV