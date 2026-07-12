# Calibration

Post-prediction calibration methods applied to continuous model outputs.

## Sparsity Calibration

`calibrate_sparsity.py`

Corrects under/over-predicted dropout. For each gene, computes the zero-expression rate in control cells, derives a quantile threshold from it (scaled by a `ratio` hyperparameter, which in the manuscript we set this to be 1), and re-zeros predicted perturbed values that fall below that threshold — pulling the predicted zero-inflation rate back toward what's actually observed in matched controls.

## Other Calibrations

These are calibrations attempted other than sparsity adjustment.

### Variance Calibration

`var_calibration/`

Corrects under/over-predicted variance. For each gene, rescales predicted perturbed expression around its mean toward the variance observed in control cells, with a `ratio` hyperparameter interpolating between no change (0) and fully matching control variance (1).

### Logistic Regression Calibration

`logreg_calibration/`

Corrects biased DE (differential expression) calling. Trains a multinomial logistic regression on the calibration split to remap predicted DE features (p-value, log2FC, gene-level DE frequency) to corrected 3-class labels (Non-DE / Up / Down). Includes a within-cell-line variant (`calibrate_de_logreg_within.py`) and a cross-cell-line variant (`calibrate_de_logreg_cross.py`, fit on training cell lines' calibration perts and evaluated on the held-out cell line).
