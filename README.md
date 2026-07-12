# Benchmarking Differential Expression Classification in Single-Cell Perturbation Models

This repository contains code needed to reproduce analysis in the manuscript.

## Overview

There are in total 8 benchmarked methods, each wrapped within its own folder:

| Method | Code |
| --- | --- |
| Perturbation Majority Vote | `majority_vote/` |
| KNN Classifier | `KNN_vote/` |
| Regularized Logistic Regression | `logistic_reg/` |
| Scouter | `Scouter/` |
| TranScouter | `TranScouter/` |
| Variational Autoencoder | `VAE/` |
| scVI | `scVI/` |
| STATE | `STATE/` |

For post-prediction distributional calibration, code is stored in `calibration/`.

Exploratory data analysis code in `EDA/`.

Environments used to run this project is stored in `environment.yml`.

## Data Processing Pipeline

### 1. DE Calling
DE labels are computed per perturbation via Wilcoxon rank-sum test with Benjamini–Hochberg correction. Current DE threshold is $p < 0.01$ and $|log2FC| >= 0.00$

```
majority_vote/DE_DOC_stats.py
```

Outputs: `data/DE_DOC_{dataset}.csv`

### 2. Train Test Split (80/20)
Perturbations are split stratified by dominant DE class, written as a `split_80` column into the splitted CSV.

```
majority_vote/train_test_split.py
python train_test_split.py   # test_size=0.20, random_state=305
```

Outputs: `data/DE_DOC_{dataset}_splitted.csv` (adds `split_80` column)

### 3. Calibration Split (60/20/20)
25% of training perturbations (from `split_80 == 'train'`) are carved out as a calibration set for hyperparameter tuning. Written as a `split_calib` column; the test set is untouched.

```
majority_vote/calibration_split.py
python calibration_split.py data/DE_DOC_{dataset}_splitted.csv   # cal_size=0.25
```

Outputs: adds `split_calib` column in-place to `data/DE_DOC_{dataset}_splitted.csv`

### NOTE
Even though there is a calibration split, we did not end up using it in the current version of the benchmark. This split was originally set for tuning calibration-related hyperparameters (ex. sparsity ratio $r$ in sparsity calibration, can see scripts for this in `calibration/tuning/`), but empirical results have found that hyperparameters have negligible effect on calibration performance, so the calibration split was merged into the test split to form the 40% test set for final performance reporting.