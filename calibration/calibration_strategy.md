# Post-Prediction Calibration

## 1. Zero-Calibration

Deep generative models trained to predict single-cell transcriptomic responses to genetic perturbations are typically optimized under reconstruction objectives — such as mean squared error or the evidence lower bound — that minimize average prediction error across all expressed genes. This encourages the decoder to produce small but non-zero values for lowly or intermittently expressed genes, resulting in predicted expression matrices that are systematically denser than observed single-cell RNA-sequencing data. Real scRNA-seq profiles are characterized by pronounced sparsity arising from both biological absence of transcription and technical dropout during library preparation, and this mismatch between model output and observed data structure inflates false-positive differential expression calls and distorts downstream analyses.

To correct this systematic bias, we applied a post-hoc, gene-level zero calibration anchored to the sparsity of unperturbed control cells. For each gene $g$, we computed the zero-expression fraction $z_g$ across all control cells as an estimate of natural sparsity in the given experimental context. A gene-specific threshold $T_g$ was then defined as the $q_g$-th quantile of predicted expression values pooled across all perturbed cells, where $q_g = r \cdot z_g$ and $r \in [0, 1]$ is a global scaling factor (for now default $r = 0.8$). Predicted values at or below $T_g$ were set to zero; all other values and all control cell measurements were left unchanged. The scaling factor $r < 1$ reflects the biological reality that perturbations can genuinely alter the expressed fraction of a gene, so enforcing exact agreement with control sparsity would be overly conservative(?)

## 2. Variance Calibration

**Motivation:** Exploratory data analysis revealed a second systematic miscalibration different than sparsity: both STATE and Scouter underestimate per-gene expression variance in perturbed cells, with Scouter exhibiting a more severe deflation than STATE. Crucially, per-gene means are unbiased in both models — the miscalibration is entirely in the second moment rather than the first. Could be that models fail to reproduce the cell-to-cell heterogeneity of real single-cell data.

To correct this, we apply a post-hoc, per-gene variance scaling that preserves the predicted mean while inflating deviations from it. For each gene $g$, let $\sigma^2_{\text{ctrl}}(g)$ be the variance observed in control cells and $\sigma^2_{\text{pred}}(g)$ be the variance of predicted perturbed cells. A scale factor is computed as:

$$s(g) = 1 + r \cdot \left( \sqrt{\frac{\sigma^2_{\text{ctrl}}(g)}{\sigma^2_{\text{pred}}(g)}} - 1 \right)$$

where $r \in [0, 1]$ is a global calibration strength (default $r = 0.8$). Each perturbed cell's predicted expression is then adjusted as:

$$\hat{X}_{\text{cal}}[i, g] = \hat{\mu}(g) + s(g) \cdot \left( \hat{X}[i, g] - \hat{\mu}(g) \right)$$

where $\hat{\mu}(g)$ is the per-gene predicted mean across perturbed cells. The result is clipped to $[0, \infty)$ to preserve the non-negativity of log-normalised expression. At $r = 0$ the predictions are unchanged; at $r = 1$ the per-gene variance is fully scaled to match control variance. Scale factors are capped at 10 to prevent numerical explosion for genes predicted as nearly constant ($\sigma^2_{\text{pred}} \approx 0$).

Control-cell variance serves as the calibration anchor for consistency with zero-calibration and because it represents the baseline variability expected in the absence of a perturbation signal. Critically, the method never requires access to real perturbed expression — only control cells are needed, which are always available in any Perturb-seq experimental design. This makes the approach applicable to any continuous prediction model without modification.

The scale factor $s(g)$ is bidirectional by construction: when $\sigma^2_{\text{pred}}(g) < \sigma^2_{\text{ctrl}}(g)$ (underdispersion, as observed in Scouter and STATE), $s(g) > 1$ and deviations are expanded; when $\sigma^2_{\text{pred}}(g) > \sigma^2_{\text{ctrl}}(g)$ (overdispersion), $s(g) < 1$ and deviations are contracted. A model that already matches control-level variance receives $s(g) = 1$ and is left unchanged. No prior knowledge of the direction of miscalibration is required.

Variance calibration should probably be applied before zero-calibration since zero-calibration sets reduces variance?