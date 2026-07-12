# Cross-Dataset Concordance Analysis

**Method**: Only comparing (perturbation, gene) pairs that exist in BOTH datasets.

**Datasets**: adamson, hepg2, jurkat, replogle_k562, replogle_rpe1, tcell_rest

## Summary Table

| Dataset 1 | Dataset 2 | Common Perts | Pairs | Accuracy | Kappa | Up Agr | Down Agr | NotDE Agr | LogFC r | LogFC rho |
|-----------|-----------|--------------|-------|----------|-------|--------|----------|-----------|---------|----------|
| adamson | hepg2 | 60 | 134,040 | 0.625 | 0.104 | 0.069 | 0.155 | 0.937 | 0.214 | 0.142 |
| adamson | jurkat | 60 | 121,260 | 0.601 | 0.086 | 0.036 | 0.141 | 0.958 | 0.230 | 0.150 |
| adamson | replogle_k562 | 41 | 78,310 | 0.606 | 0.218 | 0.191 | 0.275 | 0.917 | 0.529 | 0.406 |
| adamson | replogle_rpe1 | 45 | 72,720 | 0.606 | 0.164 | 0.146 | 0.307 | 0.849 | 0.294 | 0.225 |
| adamson | tcell_rest | 43 | 99,943 | 0.649 | 0.014 | 0.027 | 0.020 | 0.971 | 0.060 | 0.047 |
| hepg2 | jurkat | 2392 | 18,253,352 | 0.949 | 0.204 | 0.118 | 0.227 | 0.979 | 0.221 | 0.128 |
| hepg2 | replogle_k562 | 1091 | 4,660,752 | 0.883 | 0.266 | 0.270 | 0.357 | 0.931 | 0.310 | 0.238 |
| hepg2 | replogle_rpe1 | 1543 | 6,494,487 | 0.888 | 0.289 | 0.376 | 0.508 | 0.915 | 0.405 | 0.287 |
| hepg2 | tcell_rest | 1218 | 8,385,640 | 0.968 | 0.015 | 0.042 | 0.025 | 0.980 | 0.036 | 0.031 |
| jurkat | replogle_k562 | 1091 | 4,647,660 | 0.895 | 0.203 | 0.298 | 0.365 | 0.923 | 0.302 | 0.239 |
| jurkat | replogle_rpe1 | 1543 | 6,026,958 | 0.886 | 0.165 | 0.329 | 0.383 | 0.905 | 0.257 | 0.195 |
| jurkat | tcell_rest | 1218 | 8,366,780 | 0.965 | 0.032 | 0.066 | 0.039 | 0.980 | 0.050 | 0.046 |
| replogle_k562 | replogle_rpe1 | 848 | 2,401,536 | 0.823 | 0.228 | 0.309 | 0.396 | 0.873 | 0.338 | 0.275 |
| replogle_k562 | tcell_rest | 346 | 1,287,209 | 0.937 | 0.045 | 0.059 | 0.040 | 0.977 | 0.061 | 0.056 |
| replogle_rpe1 | tcell_rest | 725 | 2,541,171 | 0.907 | 0.037 | 0.051 | 0.039 | 0.970 | 0.070 | 0.070 |

## Interpretation

- **Accuracy**: Fraction of (pert,gene) pairs with same direction label
- **Kappa**: Cohen's kappa (agreement beyond chance)
- **Per-class Agreement**: Given class X in dataset1, P(class X in dataset2)
- **LogFC r/rho**: Correlation of logFC values for same (pert,gene) pairs

## Files

- `concordance_*.pdf`: Confusion matrix + logFC scatter + per-class bars
- `shared_pairs_*.csv`: Merged data with columns: perturbation, gene, direction_*, logfc_*
- `summary_heatmap.pdf`: Overview heatmaps
