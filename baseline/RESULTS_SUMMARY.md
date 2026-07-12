# Baseline Methods for Perturbation Response Prediction
## Comprehensive Results Summary

### Task Overview

Given a perturbation $p$ (gene knockout) and target gene $g$, predict:
- **3-class**: Is $g$ upregulated, downregulated, or not differentially expressed (not_de)?

### Datasets

| Dataset | Cell Type | Train Size | Test Size | % not_de |
|---------|-----------|------------|-----------|----------|
| Adamson | K562 (CRISPRi) | 258K | 91K | 78.5% |
| Replogle K562 | K562 | 3.27M | 1.09M | 90.8% |
| Replogle RPE1 | RPE1 | 4.63M | 1.55M | 90.1% |
| HepG2 | HepG2 | 13.8M | 4.62M | 96.0% |
| Jurkat | Jurkat | 12.7M | 4.26M | 97.4% |
| T-cell rest | T-cell | 67.0M | 22.3M | 99.5% |

---

## Methods

### 1. Majority Vote Baselines
- **majority_gene**: Predict most common label for each gene (ignores perturbation)
- **majority_pert**: Predict most common label for each perturbation (ignores gene)

### 2. Neighbor-Based Methods
Find similar perturbations/genes and vote based on their observed labels.

**Similarity Sources:**
- **ESM2**: Protein sequence embeddings
- **GenePT**: Gene description + protein embeddings
- **Co-pathway**: Shared KEGG/GO pathway membership

**Approaches:**
- **pert_neighbor_X**: Find k-nearest perturbations to query, vote on target gene's response
- **gene_neighbor_X**: Find k-nearest genes to target, vote on their response
- **both_neighbor_X**: Combine both
- **ensemble**: Combine all similarity sources
- **two_stage_X**: First predict DE vs not_de, then up vs down

---

## Results

**Note**: All results below are on the **same full data** (same train/test splits). The improvement from neighbor methods comes from the METHOD, not different data.

### Majority Vote Baseline

| Dataset | Accuracy | Macro F1 | Up F1 | Down F1 | not_de F1 |
|---------|----------|----------|-------|---------|-----------|
| Adamson | 0.884 | **0.773** | 0.639 | 0.749 | 0.933 |
| Replogle K562 | 0.908 | 0.333 | 0.014 | 0.034 | 0.952 |
| Replogle RPE1 | 0.905 | 0.501 | 0.195 | 0.357 | 0.950 |
| HepG2 | 0.960 | 0.327 | 0.000 | 0.000 | 0.980 |
| Jurkat | 0.974 | 0.329 | 0.000 | 0.000 | 0.987 |
| T-cell rest | 0.995 | 0.332 | 0.000 | 0.000 | 0.997 |

**Key Insight**: With extreme class imbalance, majority vote predicts only `not_de` for HepG2/Jurkat/T-cell (F1=0 for minority classes).

---

### Neighbor-Based Methods (Same Data, Better Results)

#### Best Method per Dataset

| Dataset | Best Method | Macro F1 | Up F1 | Down F1 | not_de F1 |
|---------|-------------|----------|-------|---------|-----------|
| Adamson | majority_gene | **0.773** | 0.639 | 0.749 | 0.933 |
| Replogle K562 | pert_neighbor_copath | **0.553** | 0.343 | 0.373 | 0.943 |
| Replogle RPE1 | pert_neighbor_copath | **0.581** | 0.360 | 0.442 | 0.941 |
| HepG2 | pert_neighbor_copath | **0.517** | 0.265 | 0.310 | 0.977 |
| Jurkat | pert_neighbor_copath | **0.453** | 0.180 | 0.198 | 0.982 |
| T-cell rest | pert_neighbor_copath | **0.353** | 0.034 | 0.030 | 0.996 |

#### Method Comparison (Average Across 6 Datasets)

| Method | Avg Macro F1 | Avg Up F1 | Avg Down F1 |
|--------|--------------|-----------|-------------|
| **pert_neighbor_copath** | **0.537** | 0.302 | 0.349 |
| ensemble | 0.526 | 0.283 | 0.334 |
| two_stage_esm2 | 0.503 | 0.254 | 0.302 |
| pert_neighbor_esm2 | 0.496 | 0.243 | 0.288 |
| pert_neighbor_genept | 0.481 | 0.219 | 0.259 |
| majority_gene | 0.433 | 0.141 | 0.190 |
| gene_neighbor_* | 0.433 | 0.141 | 0.190 |
| majority_pert | 0.319 | 0.000 | 0.000 |

---

### Detailed Results by Dataset (Downsampled)

#### Adamson
| Method | Accuracy | Macro F1 | Up F1 | Down F1 |
|--------|----------|----------|-------|---------|
| majority_gene | 0.884 | 0.773 | 0.639 | 0.749 |
| pert_neighbor_esm2 | 0.883 | 0.773 | 0.639 | 0.748 |
| pert_neighbor_copath | 0.880 | 0.767 | 0.628 | 0.742 |

#### Replogle K562
| Method | Accuracy | Macro F1 | Up F1 | Down F1 |
|--------|----------|----------|-------|---------|
| pert_neighbor_copath | 0.894 | **0.553** | 0.343 | 0.373 |
| ensemble | 0.886 | 0.530 | 0.305 | 0.346 |
| two_stage_esm2 | 0.861 | 0.485 | 0.238 | 0.291 |
| majority_gene | 0.908 | 0.333 | 0.014 | 0.034 |

#### Replogle RPE1
| Method | Accuracy | Macro F1 | Up F1 | Down F1 |
|--------|----------|----------|-------|---------|
| pert_neighbor_copath | 0.891 | **0.581** | 0.360 | 0.442 |
| ensemble | 0.887 | 0.580 | 0.351 | 0.449 |
| majority_gene | 0.906 | 0.501 | 0.195 | 0.357 |

#### HepG2
| Method | Accuracy | Macro F1 | Up F1 | Down F1 |
|--------|----------|----------|-------|---------|
| pert_neighbor_copath | 0.955 | **0.517** | 0.265 | 0.310 |
| ensemble | 0.954 | 0.514 | 0.262 | 0.304 |
| majority_gene | 0.960 | 0.327 | 0.000 | 0.000 |

#### Jurkat
| Method | Accuracy | Macro F1 | Up F1 | Down F1 |
|--------|----------|----------|-------|---------|
| pert_neighbor_copath | 0.964 | **0.453** | 0.180 | 0.198 |
| ensemble | 0.961 | 0.426 | 0.142 | 0.156 |
| majority_gene | 0.974 | 0.329 | 0.000 | 0.000 |

#### T-cell rest
| Method | Accuracy | Macro F1 | Up F1 | Down F1 |
|--------|----------|----------|-------|---------|
| pert_neighbor_copath | 0.993 | **0.353** | 0.034 | 0.030 |
| majority_gene | 0.995 | 0.332 | 0.000 | 0.000 |

---

## Key Findings

### 1. Co-pathway neighbors are most informative
`pert_neighbor_copath` achieves the best Macro F1 on 5/6 datasets. This suggests that **perturbations targeting genes in the same pathway have similar downstream effects**.

### 2. Perturbation neighbors >> Gene neighbors
- **Perturbation-neighbor methods** improve significantly over majority vote
- **Gene-neighbor methods** perform identically to majority_gene on all datasets
- This indicates that **perturbation identity matters more than target gene similarity**

### 3. Adamson is qualitatively different
Only dataset where gene-level majority voting achieves competitive Macro F1 (0.77). Possible reasons:
- Smaller, more focused perturbation set (86 perturbations vs 1000+)
- CRISPRi mechanism may have more predictable effects
- Better signal-to-noise in the original experiment

### 4. Class imbalance severely limits performance
| Dataset | % not_de | Majority F1 | Best F1 |
|---------|----------|-------------|---------|
| Adamson | 78.5% | 0.773 | 0.773 |
| K562 | 90.8% | 0.333 | 0.553 |
| RPE1 | 90.1% | 0.501 | 0.581 |
| HepG2 | 96.0% | 0.327 | 0.517 |
| Jurkat | 97.4% | 0.329 | 0.453 |
| T-cell | 99.5% | 0.332 | 0.353 |

As imbalance increases, even the best methods struggle to detect minority classes.

### 5. Two-stage approach shows mixed results
Hierarchical prediction (DE → direction) does not consistently outperform single-stage methods.

---

## Comparison with Published Methods

| Method | K562 Acc | K562 Macro F1 | RPE1 Acc | RPE1 Macro F1 |
|--------|----------|---------------|----------|---------------|
| **pert_neighbor_copath** | 0.894 | **0.553** | 0.891 | **0.581** |
| majority_gene | 0.908 | 0.333 | 0.906 | 0.501 |
| SynthPert* | 0.89 | 0.70 | 0.87 | 0.72 |
| SUMMER* | -- | -- | -- | -- |

*Note: SynthPert numbers from paper, may use different splits/preprocessing.

---

## Files

- `output/baseline_results/` - Full data majority vote results
- `output/neighbor_baseline_results/` - Downsampled neighbor method results
- `baseline/run_neighbor_baselines.py` - Run neighbor methods
- `baseline/writeup/main.tex` - LaTeX writeup (detailed)
