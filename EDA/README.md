# Exploratory Data Analysis (EDA)

Visualising ground-truth Perturb-Seq data and comparing with model predictions. All paths are centralised in `config.py`.

---

## Pipeline overview

```
compute_gene_stats.py          ──► gene_stats_{dataset}.csv
compute_de_profile_stats.py    ──► gene_de_stats_{dataset}.csv
                                   pert_de_stats_{dataset}.csv
compute_covariance.py          ──► gene_covariance_{dataset}.npz
                                   pert_covariance_{dataset}.npz

plot_expression_distributions.py   (reads gene_stats)
plot_de_distributions.py           (reads gene_de_stats, pert_de_stats)
plot_covariance.py                 (reads gene_covariance, pert_covariance)
plot_model_prediction_eda.py       (reads h5ad prediction files directly)
```

---

## compute_gene_stats.py

Computes per-gene expression statistics from the ground-truth h5ad after
log TP10k normalisation, separately for control and pooled-perturbed cells.

**Output:** `results/gene_stats_{dataset}.csv`

| Column | Description |
|--------|-------------|
| `ctrl_mean` / `pert_mean` | Per-gene mean expression across control / all perturbed cells |
| `ctrl_var` / `pert_var` | Per-gene variance |
| `ctrl_zero_prop` / `pert_zero_prop` | Fraction of cells with zero expression |
| `ctrl_q25/50/75/90/99` | Percentiles of per-gene expression in control cells |
| `pert_q25/50/75/90/99` | Percentiles of per-gene expression in perturbed cells |

**Submission:** `sbatch compute_gene_stats.sh` (array job, one task per dataset)

---

## compute_de_profile_stats.py

Computes per-gene and per-perturbation differential expression statistics
from the DE_DOC CSV files (Wilcoxon + BH, p < 0.01).

**Output:** `results/gene_de_stats_{dataset}.csv` and `results/pert_de_stats_{dataset}.csv`

| Column | Description |
|--------|-------------|
| `n_de` | Number of DE events (up or down) |
| `de_rate` | Fraction of tested pairs that are DE |
| `n_up` / `n_down` | Directional breakdown |
| `mean_logFC` / `var_logFC` | Mean and variance of logFC across pairs |
| `mean_abs_logFC` | Mean absolute logFC (effect size) |
| `q25/50/75/90_abs_logFC` | Distribution of absolute logFC |

**Submission:** `sbatch compute_de_profile_stats.sh` (array job, one task per dataset)

---

## compute_covariance.py

Computes two covariance matrices from ground-truth data.

**Outputs:** `results/gene_covariance_{dataset}.npz`, `results/pert_covariance_{dataset}.npz`

Each `.npz` archive contains keys `cov` (N×N float64 matrix) and `genes`/`perts`
(N-length string array of labels matching matrix rows/columns).
Load with: `data = np.load(path, allow_pickle=True); cov = data['cov']`.

**Submission:** `sbatch compute_covariance.sh` (array job, one task per dataset)

---

## plot_expression_distributions.py

Visualises the gene stats CSVs. Run after `compute_gene_stats.py`.

| Figure | Interpretation |
|--------|----------------|
| `gene_marginal_distributions_{dataset}.pdf` | Per-gene mean, variance, and zero proportion in control cells. Reveals the baseline expression landscape. |
| `gene_mean_variance.pdf` | Mean–variance scatter per dataset. scRNA-seq data shows a Poisson floor (var ≈ mean) with overdispersion for highly expressed genes. |
| `gene_ctrl_vs_pert_mean.pdf` | Control vs pooled-perturbed per-gene mean. Points above y = x are globally up-regulated genes; below = globally down-regulated. Most genes should lie near the diagonal. |
| `gene_ctrl_vs_pert_distributions.pdf` | Overlaid histograms of mean, variance, and zero proportion for control vs perturbed cells. Distributional shifts indicate systematic perturbation effects. |
| `gene_ctrl_vs_pert_distributions_violin.pdf` | Same comparison as violin plots. |

---

## plot_de_distributions.py

Visualises the DE profile stats CSVs. Run after `compute_de_profile_stats.py`.

| Figure | Interpretation |
|--------|----------------|
| `gene_logfc_distributions.pdf` | Distribution of per-gene mean logFC and mean \|logFC\|. A symmetric distribution near zero means effects are small and balanced across perturbations. |
| `gene_de_rate_distribution.pdf` | Distribution of per-gene DE rates. Heavy concentration near zero is expected — a long right tail indicates hub genes that are broadly responsive. |
| `gene_up_down_asymmetry.pdf` | Scatter of n_up vs n_down per gene. Points off the diagonal indicate genes with a systematic directional bias across perturbations. |
| `pert_de_distributions.pdf` | Distribution of DE gene counts and DE rates per perturbation. Heterogeneity reveals that perturbations vary widely in strength, which is important context for calibration evaluation. |

---

## plot_covariance.py

Visualises covariance matrices as Pearson correlation heatmaps.
Run after `compute_covariance.py`.

| Figure | Interpretation |
|--------|----------------|
| `gene_covariance_{dataset}.pdf` | Correlation heatmap of the top-N most variable genes in control cells, ordered by hierarchical clustering. Block structure reveals co-expression modules. |
| `pert_covariance_{dataset}.pdf` | Correlation heatmap of perturbations based on their logFC profiles. Clustered blocks indicate groups of perturbations with similar transcriptome-wide effects. |

---

## plot_model_prediction_eda.py

Evaluates prediction quality for a single model × dataset × calibration combination.
All outputs are saved to a dedicated subdirectory named `{model}_{dataset}_{calibration}`.

For cross-cell-line models (TranScouter, STATE), both h5ads are subsetted before
analysis: perturbations listed in `dropped_test_perts.json` are dropped and genes
are aligned to the common gene universe.

**Arguments:**

| Argument | Choices | Default |
|----------|---------|---------|
| `--model` | `Scouter`, `TranScouter`, `STATE` | required |
| `--dataset` | `k562`, `hepg2`, `jurkat`, `rpe1` | required |
| `--calibration` | `none`, `sparsity` | `none` |
| `--tag` | any string | `within` (Scouter) / `cross` (others) |

**Example calls:**
```bash
python plot_model_prediction_eda.py --model Scouter     --dataset k562
python plot_model_prediction_eda.py --model TranScouter --dataset rpe1
python plot_model_prediction_eda.py --model STATE       --dataset hepg2
python plot_model_prediction_eda.py --model Scouter     --dataset k562 --calibration sparsity
```

**Figures** (`figures/{model}_{dataset}_{calibration}/`):

| Figure | Interpretation |
|--------|----------------|
| `real_vs_pred_mean.pdf` | Scatter of per-gene mean expression computed by pooling all perturbed cells, real (x) vs predicted (y). Pearson r and Spearman ρ in title. Systematic offset reveals over- or under-prediction bias; low correlation means the model fails to rank genes by average expression level. |
| `per_gene_pearsonr_hist.pdf` | Histogram of per-gene Pearson r values. Each gene's r is computed by correlating its real vs predicted per-perturbation mean expression vector across all perturbations, then averaged — this is the `pearson_per_gene` metric from `perturb_metrics.py`. The mean r shown in the title is directly comparable to the metrics CSV. A distribution shifted left indicates the model fails to capture gene-specific perturbation response profiles. |
| `ctrl_vs_Per-gene_mean__rea.pdf` | Per-gene mean in control cells (x) vs real perturbed cells (y). Most genes should lie near y = x; deviations indicate globally shifted genes across all perturbations. Sanity check on the dataset. |
| `ctrl_vs_Per-gene_mean__pre.pdf` | Same as above but y-axis is predicted cells. Compare directly with the real counterpart to assess whether the model captures global expression shifts relative to controls. |
| `distributions.pdf` | Three overlaid histograms (real vs predicted) of mean expression, variance, and zero proportion across perturbed cells. A shifted mean = systematic bias; under-dispersed predicted variance = regression to the mean; lower predicted zero proportion = over-smoothing of sparse expression. |

**Results** (`results/{model}_{dataset}_{calibration}/`):

| File | Description |
|------|-------------|
| `gene_summary.csv` | One row per gene with columns: `gene`, `real_mean`, `pred_mean` (pooled), `real_mean_of_means`, `pred_mean_of_means` (averaged over perturbations), `per_gene_pearsonr` (Pearson r across perturbations), `real_var`, `pred_var`, `real_zero_prop`, `pred_zero_prop`, `ctrl_mean`, `ctrl_var`, `ctrl_zero_prop`. |
