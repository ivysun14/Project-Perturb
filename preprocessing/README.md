# Preprocessing

DE labels and train / calibration / test splits shared by all models. Run from the repo root.

## 1. DE calling

DE labels are computed per perturbation with a Wilcoxon rank-sum test against non-targeting cells
(CP10k + log1p normalization, done inside the script), Benjamini–Hochberg correction within each
perturbation, threshold adjusted $p < 0.01$, no log2FC cutoff.

```
python preprocessing/DE_DOC_stats.py --adata {dataset}_raw.h5ad --output data/DE_DOC_{dataset}.csv
```

Output `data/DE_DOC_{dataset}.csv`, one row per (perturbation, gene): `label` (0 non-DE, 1 up,
2 down), `p_value` (BH-adjusted), `logFC`, `log2FC`.

## 2. Train / test split (80/20)

Perturbations are stratified by their dominant DE class and split 80/20, written as a `split_80`
column.

```
python preprocessing/train_test_split.py   # test_size=0.20, random_state=305
```

Output: `data/DE_DOC_{dataset}_splitted.csv` (adds `split_80`).

## 3. Calibration split (60/20/20)

25% of the training perturbations (`split_80 == 'train'`) become a calibration set, written as a
`split_calib` column; the test set is unchanged.

```
python preprocessing/calibration_split.py data/DE_DOC_{dataset}_splitted.csv   # cal_size=0.25
```

Output: `split_calib` added in place to `data/DE_DOC_{dataset}_splitted.csv`.

**Note.** The calibration split was meant for tuning calibration hyperparameters (e.g. the sparsity
ratio $r$ in sparsity calibration, `calibration/tuning/`). These turned out to have negligible
effect, so for final reporting the calibration perturbations are merged into the test set, giving a
**60/40 train / test split**.

## Helpers

`data_utility.py`: `resolve_repeats` (average duplicated gene columns of an AnnData) and
`update_DE_DOC_labels` (re-label DE at another p-value / log2FC threshold).

Other code imports these modules as `from preprocessing.<module> import ...` with the repo root on
`sys.path`.
