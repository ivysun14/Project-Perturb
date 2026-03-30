# Project-Perturb

## Goal

Benchmark and improve post-prediction calibration techniques for continuous perturbation response models in transcriptomics, with the aim of recovering accurate differential expression (DE) classification from model outputs.

---

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