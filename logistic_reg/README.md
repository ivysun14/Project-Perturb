# Logistic Regression Baseline

Per-gene multinomial logistic regression for 3-class differential expression (DE) prediction (up / down / non-DE) in perturb-seq experiments.

---

## Algorithm

Each perturbation is represented as a GenePT embedding vector (3072-d). A separate `LogisticRegression` model is trained for each gene independently, using the perturbation embeddings as features and that gene's DE labels as targets. To predict the label for a test (perturbation, gene) pair, the perturbation's embedding is passed through the gene's model, which outputs a 3-class probability via softmax.

All experiments use a fixed regularisation strength C = 0.01 (L2 penalty, no cross-validation).

**Edge cases:**

| Situation | Handling |
|---|---|
| Gene has < 2 training observations | No model fitted; falls through to the "no fitted model" default below |
| Gene has only one class in training | Constant predictor; always predicts that class (probability 1) |
| Gene model saw only 2 of 3 classes | Probability for the unseen class is set to 0 at prediction time |
| Test perturbation not in embedding dict | Pair dropped from evaluation entirely (excluded from predictions) |
| Test gene has no fitted model (incl. < 2 obs) | Defaults to class 0 with uniform probabilities [1/3, 1/3, 1/3]; pair is retained in output |

> **Design note (cross-cell-line, duplicate pairs):** When the same (pert, gene) pair appears in multiple source cell lines with different labels, `merge_train_sets` resolves the conflict by majority vote over those labels. Ties are broken uniformly at random (seeded by `--random_seed`) and printed as warnings. The merged training set contains exactly one row per unique (pert, gene) pair.

---

## Experiments

All scripts are submitted from `logistic_reg/` (`sbatch <script>.sh`). Logs go to `{experiment_dir}/log/`; error analysis outputs go to `{experiment_dir}/error_analysis/`.

---

### 1. Full dataset — within-cell-line

**Script:** `logreg_within.sh`  
**Python script:** `logistic_reg.py`  
**Output directory:** `logreg_0.01_full_within_expr/`

**Data setup:**
- Input: dense DE_DOC matrix (`DE_DOC_{DS}_splitted.csv`), 60/20/20 train/calib/test split by perturbation.
- Training on the `train` split (60%); calib and test evaluated independently; a `combined` file concatenates both.

**Output files** (`logreg_0.01_full_within_expr/results/`):
```
{DS}_logreg_logreg_metrics_{calib,test,combined}.csv
{DS}_logreg_logreg_predictions_{calib,test,combined}.csv
{DS}_logreg_logreg_no_model_genes.json
```

**Caveats:**
- Evaluates on the **combined** calib+test (40%) as the primary reported result.
- No gene universe or pert-drop filter applied; metrics are on the full evaluation set.
- Output filenames contain `_logreg_` twice due to dataset name suffix (`{DS}_logreg`) combining with the script's fixed `_logreg_` infix.

---

### 2. Full dataset — cross-cell-line

**Script:** `logreg_cross.sh`  
**Python script:** `logistic_reg.py`  
**Output directory:** `logreg_0.01_full_cross_expr/`

**Data setup:**
- Training: the `train` (60%) and `test` (20%) splits of all 3 source cell lines — 80% of each source dataset. The source calibration splits are excluded from training.
- Duplicate (pert, gene) pairs across source cell lines are merged by majority vote into a single training set.
- Evaluation: `calib` (20%) and `test` (20%) of the held-out cell line, evaluated independently and combined.

**Output files** (`logreg_0.01_full_cross_expr/results/`):
```
{TestDS}_logreg_transfer_logreg_metrics_{calib,test,combined}.csv
{TestDS}_logreg_transfer_logreg_predictions_{calib,test,combined}.csv
{TestDS}_logreg_transfer_logreg_no_model_genes.json
{TestDS}_logreg_transfer_transfer_unseen.json
```

**Filtered metrics** (`logreg_0.01_full_cross_expr/cross_logreg_metrics_filtered.csv`):  
Run `metric_drop_perts.py` to recompute metrics after restricting to the common gene universe (`common_gene_universe.csv`) and dropping low-coverage test perturbations (`dropped_test_perts.json`). Reads calib+test prediction files directly.

**Caveats:**
- `{TestDS}_logreg_transfer_transfer_unseen.json` records genes in the held-out evaluation set with no training observations across all 3 source cell lines; these default to class 0.
- No gene universe or pert-drop filter applied to the raw results; use `metric_drop_perts.py` for the filtered evaluation.

---

### 3. PerturbQA — within-cell-line

**Script:** `logreg_perturbQA_within.sh`  
**Python script:** `logistic_reg_perturbQA.py`  
**Output directory:** `logreg_0.01_perturbQA_within_expr/`

**Data setup:**
- Input: sparse two-file format — `{ds}-de.csv` (binary DE/non-DE) and `{ds}-dir.csv` (direction for DE pairs). Merged into 3-class labels at load time.
- Train/test split by perturbation (~83%/~17%); no calibration split.
- Training on `train` split; evaluated on `test` split once.
- The label matrix is **sparse**: each gene's model is fit only on the perturbations that have an observed label for that gene in the training split.

**Output files** (`logreg_0.01_perturbQA_within_expr/results/`):
```
{DS}_logreg_metrics.csv
{DS}_logreg_predictions.csv
```

**Caveats:**
- No calibration split → single metrics/predictions file per dataset (no `_calib`/`_test` suffix).
- No gene universe or pert-drop filter applied.
- Constant predictors are the norm, not the exception: ~70–80% of genes in each dataset become constant predictors (always predicting non-DE). This reflects PerturbQA's sparse coverage — for most genes, all training perturbations either don't observe that gene or label it non-DE, leaving no class variation for the model to learn. The remaining ~20–30% of genes have enough DE examples to fit a real model.

---

### 4. PerturbQA — cross-cell-line

**Script:** `logreg_perturbQA_cross.sh`  
**Python script:** `logistic_reg_perturbQA.py`  
**Output directory:** `logreg_0.01_perturbQA_cross_expr/`

**Data setup:**
- Training: ALL (pert, gene) pairs from 3 source cell lines — both `train` and `test` splits — restricted to the 1337-gene common gene universe (`common_gene_universe_QA.csv`). Duplicate (pert, gene) pairs across cell lines are merged by majority vote.
- Evaluation: `test` split of the held-out cell line, restricted to the same gene universe.

**Output files** (`logreg_0.01_perturbQA_cross_expr/results/`):
```
{TestDS}_logreg_transfer_metrics.csv
{TestDS}_logreg_transfer_predictions.csv
```

**Filtered metrics** (`logreg_0.01_perturbQA_cross_expr/cross_logreg_metrics_filtered.csv`):  
Run `metric_drop_perts.py` to recompute metrics after additionally dropping low-coverage test perturbations (`dropped_test_perts_QA.json`).

**Caveats:**
- Gene universe restriction is applied before training and evaluation. Because the common universe is defined as genes present in all 4 datasets, every test gene is guaranteed to have at least one training observation — no unseen-gene fallback is triggered.
- No pert-drop filter applied to the raw results; use `metric_drop_perts.py` for the filtered evaluation.
- No calibration split → single metrics/predictions file per dataset.
- Constant predictors remain common even with 3 source cell lines of training data: ~50% of the 1337 common-universe genes become constant predictors. Cross-cell-line pooling reduces (but does not eliminate) the class-imbalance issue because DE labels are sparse across perturbations for most genes. The counts below compare the raw data (common genes with exclusively non-DE labels across all 3 source cell lines) against the fitted model output; small differences arise because `merge_train_sets` majority-votes duplicate (pert, gene) pairs across source cell lines, which can shift the effective label set for a gene in two directions:

- **Constant > raw (e.g. HepG2: 707 vs. 672)**: majority voting resolves mixed-label pairs toward non-DE, converting a borderline gene into an all-non-DE constant predictor.
- **Constant < raw (e.g. RPE1: 672 vs. 685; Jurkat: 685 vs. 704)**: if every (pert, gene) pair for a gene appears identically across all 3 source cell lines, `merge_train_sets` deduplicates them into a single unique pair. If that gene's observations collapse to just 1 unique pair post-merge — which happens to carry only non-DE labels — it ends up with fewer than 2 training observations, is skipped entirely, and falls into the "no model" default at prediction time. Such genes are not counted as constant predictors despite appearing all-non-DE in the raw data.

| Held-out | All-nonDE in raw data | Constant predictors (from log) | Fitted models (from log) |
|---|---|---|---|
| K562 | 743 / 1337 (55.6%) | 744 | 593 |
| RPE1 | 685 / 1337 (51.2%) | 672 | 665 |
| HepG2 | 672 / 1337 (50.3%) | 707 | 630 |
| Jurkat | 704 / 1337 (52.7%) | 685 | 652 |
- This mirrors the KNN and majority-vote PerturbQA cross conventions: all source pairs used (no split filter), gene-universe restriction only.

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

- **`logistic_reg.py`** — main script for full DE_DOC experiments (within and cross); handles both modes via `--de_doc_file` vs `--train_de_doc_files`/`--test_de_doc_file`
- **`logistic_reg_perturbQA.py`** — script for PerturbQA experiments (within and cross); imports helper functions from `logistic_reg.py`; handles both modes via `--de_file` vs `--de_file_train`/`--de_file_test`
- **`metric_drop_perts.py`** — recomputes cross-cell-line metrics after gene universe and pert-drop filtering for both full (`logreg_0.01_full_cross_expr/`) and PerturbQA (`logreg_0.01_perturbQA_cross_expr/`) experiments; outputs `cross_logreg_metrics_filtered.csv` in each experiment directory