# Majority Vote Baseline

Gene-based majority vote baseline for 3-class differential expression (DE) prediction (up / down / non-DE) in perturb-seq experiments.

---

## Algorithm

**Training:** for each gene seen in the training set, count how many times it was labelled up (1), down (2), or non-DE (0) across all perturbation–gene pairs. The gene is assigned whichever label appeared most often; ties are broken uniformly at random.

**Prediction:** for a new perturbation–gene pair, return the gene's majority label. Genes not seen during training default to non-DE (0). Per-class probabilities are the empirical label fractions from training.

This is a purely gene-level baseline — it ignores which perturbation is applied and relies entirely on each gene's typical response pattern across the training data.

> **Design note (cross-cell-line):** `train_gene_majority_vote` iterates every row in `train_df` and increments `gene_stats[gene][label]` by 1, with no deduplication by pert. In the cross setting, `train_df` is a flat concatenation of all source cell lines, so a (pert, gene) pair that appears in multiple source cell lines contributes one count **per cell line** to that gene's label tally. Duplicate (pert, gene) pairs across cell lines are treated as independent observations.

---

## Experiments

All scripts are run from `majority_vote/` (`sbatch <script>.sh`). Logs go to `{experiment_dir}/log/`; error analysis outputs go to `{experiment_dir}/error_analysis/`.

---

### 1. Full dataset — within-cell-line

**Script:** `majority_DE_DOC_file.sh`  
**Python script:** `majority_DE_DOC_file.py`  
**Output directory:** `DE_full_within_expr/`

**Data setup:**
- Input: dense DE_DOC matrix (`DE_DOC_{DS}_splitted.csv`), 60/20/20 train/calib/test split by perturbation.
- Training on the `train` split (60%); calib and test evaluated independently; a `combined` file concatenates both.

**Output files** (`DE_full_within_expr/results/`):
```
{DS}_metrics_{calib,test,combined}.csv
{DS}_predictions_{calib,test,combined}.csv
```

**Caveats:**
- Evaluates on the **combined** calib+test (40%) as the primary reported result.
- No gene universe or pert-drop filter applied; metrics are on the full evaluation set.

---

### 2. Full dataset — cross-cell-line

**Script:** `majority_transfer_full.sh`  
**Python script:** `majority_transfer_full.py`  
**Output directory:** `DE_full_cross_expr/`

**Data setup:**
- Training: the `train` (60%) and `test` (20%) splits of all 3 source cell lines — 80% of each source dataset. The source calibration splits are excluded from training.
- Evaluation: `calib` (20%) and `test` (20%) of the held-out cell line, evaluated independently and combined.

**Output files** (`DE_full_cross_expr/results/`):
```
{TestDS}_transfer_metrics_{calib,test,combined}.csv
{TestDS}_transfer_predictions_{calib,test,combined}.csv
{TestDS}_transfer_unseen.json
```

**Filtered metrics** (`DE_full_cross_expr/cross_majority_metrics_filtered.csv`):  
Run `metric_drop_perts.py` to recompute metrics after restricting to the common gene universe and dropping low-coverage test perturbations. Reads calib+test prediction files directly (combined predictions serve as reference but filtering uses the concatenation of calib and test).

**Caveats:**
- `{TestDS}_transfer_unseen.json` records genes present in the held-out cell line's evaluation set but not in any of the 3 source cell lines' training data.
- No gene universe or pert-drop filter applied to the raw results; use `metric_drop_perts.py` for the filtered evaluation.

---

### 3. PerturbQA — within-cell-line

**Script:** `majority_perturbQA_DE_file.sh`  
**Python script:** `majority_perturbQA_DE_file.py`  
**Output directory:** `DE_PerturbQA_within_expr/`

**Data setup:**
- Input: sparse two-file format — `{ds}-de.csv` (binary DE/non-DE) and `{ds}-dir.csv` (direction for DE pairs). Merged into 3-class labels at load time.
- Train/test split by perturbation (~83%/~17%); no calibration split.
- Training on `train` split; evaluated on `test` split once.

**Output files** (`DE_PerturbQA_within_expr/results/`):
```
{DS}_majority_metrics.csv
{DS}_majority_predictions.csv
```

**Caveats:**
- No calibration split → single metrics/predictions file per dataset (no `_calib`/`_test` suffix).
- No gene universe or pert-drop filter applied.

---

### 4. PerturbQA — cross-cell-line

**Script:** `majority_transfer_perturbQA.sh`  
**Python script:** `majority_transfer_perturbQA.py`  
**Output directory:** `DE_PerturbQA_cross_expr/`

**Data setup:**
- Training: ALL (perturbation, gene) pairs from the 3 source cell lines — both `train` and `test` splits — restricted to the 1337-gene common gene universe (`common_gene_universe_QA.csv`).
- Evaluation: `test` split of the held-out cell line, restricted to the same gene universe.

**Output files** (`DE_PerturbQA_cross_expr/results/`):
```
{TestDS}_majority_transfer_metrics.csv
{TestDS}_majority_transfer_predictions.csv
```

**Filtered metrics** (`DE_PerturbQA_cross_expr/cross_majority_metrics_filtered.csv`):  
Run `metric_drop_perts.py` to recompute metrics after additionally dropping low-coverage test perturbations (`dropped_test_perts_QA.json`).

**Caveats:**
- Gene universe restriction is applied at run time to both training and test data. Because the common universe is defined as genes present in all 4 datasets, every test gene is guaranteed to have at least one training observation — no unseen-gene fallback is ever triggered.
- No pert-drop filter applied to the raw results; use `metric_drop_perts.py` for the filtered evaluation.
- This mirrors the KNN cross-cell-line PerturbQA convention: training uses all source pairs (no split filter), and gene-universe restriction is the only filter.

---

## Data formats

### Full dataset (DE_DOC)

Single CSV per cell line with columns `pert`, `gene`, `label` (0/1/2), and `split_calib` (train / calibration / test). Produced by `DE_DOC_stats.py` from raw AnnData.

### PerturbQA

Two CSVs per cell line:
- `{ds}-de.csv` — all (pert, gene) pairs with binary DE label and `split` column
- `{ds}-dir.csv` — DE pairs only, with direction label (1=up / 0=down)

Stored in `/home/jsun140/scratchyiqunchen/project_perturb_data/perturbqa_datasets/`.

---

## Utilities

- **`analysis_utility.py`** — `train_gene_majority_vote()`, `predict_from_gene_majority()`, `compute_3class_metrics()`, `plot_data_summary()`
- **`error_utility.py`** — `run_error_analysis_from_results()` and `ErrorAnalyzer` class for per-gene and per-perturbation error analysis
- **`metric_drop_perts.py`** — recomputes cross-cell-line metrics after gene universe and pert-drop filtering for both full (`DE_full_cross_expr/`) and PerturbQA (`DE_PerturbQA_cross_expr/`) experiments; outputs `cross_majority_metrics_filtered.csv` in each experiment directory
- **`DE_DOC_stats.py`** — compute DE labels from raw AnnData via Wilcoxon rank-sum test (BH correction, adj. p < 0.01)
- **`train_test_split.py`** — perturbation-stratified train/test split
- **`calibration_split.py`** — adds a `split_calib` column to an existing DE/DOC splitted CSV
