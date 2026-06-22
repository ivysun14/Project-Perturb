# KNN Vote

Weighted k-nearest neighbours (KNN) classifier for 3-class differential expression (DE) prediction (up-regulated / down-regulated / non-DE) in perturb-seq experiments.

---

## Embedding types

Four perturbation representation types are supported:

| Type | Description | Coverage |
|---|---|---|
| **GenePT** | Text embeddings of gene descriptions (NCBI/UniProt) via OpenAI `text-embedding-3-large` (3072-d) | Full — all perturbations |
| **ESM2** | Protein sequence embeddings | Partial — ~57 perturbations missing |
| **WikiCrow** | Embeddings from WikiCrow gene summaries | Partial — ~180 perturbations missing |
| **Co-path** | No embedding vectors; neighbours pre-ranked by Jaccard similarity on shared MSigDB/KEGG/Reactome pathway memberships | Partial — ~150 perturbations missing |

Perturbations with no representation for a given type are dropped before training and evaluation. Pass `--coverage_filter` to drop all perturbations missing from *any* of the four types, so every mode is evaluated on the same set (used in sensitivity analyses).

---

## Neighbour search and voting

### Within-cell-line

**Embedding modes:** exactly k cosine-nearest training perturbations are retrieved upfront. Because every training perturbation has a label for every gene in the dataset, all k neighbours are used directly — no per-gene expansion is needed.

**Co-path mode:** up to k Jaccard-highest training perturbations from the pre-computed JSON are used. Same complete label lookup as embedding modes, so no per-gene expansion needed. Only co-path mode can produce a zero-neighbour default: when a query perturbation has zero co-path neighbours in the training set it is predicted as class 0 (non-DE), flagged `zero_neighbor=True`.

### Cross-cell-line

Because training and test cell lines may have different gene panels, a gene-aware expansion is used:

**Embedding modes:** all n_train training perturbations are retrieved sorted by cosine distance. For each query (pert, gene) pair the sorted list is scanned in order, collecting only neighbours that have a label for the target gene in the merged training dictionary. The scan stops once k valid neighbours are found or the list is exhausted.

**Co-path mode:** the same per-gene scan, but the candidate pool is restricted to co-path-connected training perturbations for that query pert (filtered to training set). Neighbours outside this pool are never considered, even if they have a label for the target gene.

In both modes, if j < k valid neighbours are found, the j available are used; if 0 are found, the prediction defaults to class 0 and the pair is flagged `zero_neighbor=True`.

### Voting

Each valid neighbour casts a weighted vote: weight = cosine similarity (1 − cosine dissimilarity) for embedding modes, or Jaccard similarity for co-path. The class with the highest total weight wins; ties broken uniformly at random.

---

## Experiments

All experiment outputs live under `KNN_vote/`. Predictions files always include a `zero_neighbor` boolean column. Error analysis outputs are written to `{experiment_dir}/error_analysis/`.

### `STATE_eval_set` convention

The `STATE_eval_set` column in collected metric files records whether gene-universe and dropped-perturbation filters were applied before metric computation:

- `NA` — no post-hoc filtering was applied; metrics are computed on the raw evaluation set produced at run time.
- `True` — the common gene universe and the dataset-specific dropped-perturbation list were applied before computing metrics.

Metrics files in `results/` are always unfiltered. Collected summary CSVs may be filtered (`STATE_eval_set=True`) or unfiltered depending on the experiment.

---

### 1. Full dataset — within-cell-line (GenePT only)

**Script:** `knn_classifier.sh` (8 array tasks: 4 datasets × 2 k)  
**Python script:** `knn_classifier.py`  
**Output directory:** `KNN_full_within_expr/`

**Data setup:**
- Input: dense DE_DOC matrix (`{DS}_splitted.csv`), 60/20/20 train/calib/test split by perturbation.
- Training on the `train` split; calib and test evaluated independently; a `combined` file concatenates both.

**Output files** (`results/`):
```
{DS}_GenePT_k{K}_knn_metrics_{calib,test,combined}.csv
{DS}_GenePT_k{K}_knn_predictions_{calib,test,combined}.csv
{DS}_GenePT_k{K}_data_summary.png
```

**Caveats:**
- `STATE_eval_set=NA` — no gene universe or pert-drop filter is applied; metrics are on the full evaluation set.
- Only GenePT embeddings. ESM2/WikiCrow/co-path within-cell-line on the full dataset are run only under the sensitivity analysis (`--coverage_filter`).

---

### 2. Full dataset — cross-cell-line (GenePT only)

**Script:** `knn_classifier_transfer.sh` (8 array tasks: 4 held-out datasets × 2 k)  
**Python script:** `knn_classifier_transfer.py`  
**Output directory:** `KNN_full_cross_expr/`

**Data setup:**
- Training: the `train` (60%) and `test` (20%) splits of all 3 source cell lines — i.e., 80% of each source dataset. The source calibration splits (20%) are excluded from training.
- Test: `calib` (20%) and `test` (20%) of the held-out cell line, evaluated separately.
- (pert, gene) pairs appearing in multiple source cell lines are resolved by **majority vote** before the training dictionary is built.

**Output files** (`results/`):
```
{TestDS}_k{K}_knn_transfer_metrics_{calib,test,combined}.csv
{TestDS}_k{K}_knn_transfer_predictions_{calib,test,combined}.csv
{TestDS}_k{K}_transfer_unseen.json
{TestDS}_k{K}_data_summary.png
```

**Filtered metrics:** `compute_cross_filtered_metrics.py` reads the raw calib+test predictions, restricts to the common gene universe (`common_gene_universe.csv`) and drops perturbations in `dropped_test_perts.json`, then recomputes metrics. Outputs to `KNN_full_cross_expr/cross_genept_metrics_filtered.csv` (`STATE_eval_set=True`).

**Caveats:**
- Raw metrics in `results/` are unfiltered (`STATE_eval_set=NA`). Use `cross_genept_metrics_filtered.csv` for filtered metrics.
- The `*_combined` prediction files are not stored; the combined set is reconstructed by concatenating `_calib` + `_test` in `compute_cross_filtered_metrics.py`.
- `common_gene_universe.csv` is the common gene universe across the 4 full-dataset cell lines (not the PerturbQA universe).

---

### 3. PerturbQA — within-cell-line (GenePT only)

**Script:** `knn_classifier_perturbQA.sh` (8 array tasks: 4 datasets × 2 k)  
**Python script:** `knn_classifier_perturbQA.py`  
**Output directory:** `KNN_perturbQA_within_expr/`

**Data setup:**
- Input: sparse two-file format — `{ds}-de.csv` (binary DE/non-DE) and `{ds}-dir.csv` (direction for DE pairs) from `perturbqa_datasets/`. Merged into 3-class labels at load time.
- Train/test split by perturbation (~83%/~17%); no calibration split.
- Training on `train`; test set evaluated once.

**Output files** (`results/`):
```
{DS}_GenePT_k{K}_knn_metrics.csv
{DS}_GenePT_k{K}_knn_predictions.csv
{DS}_GenePT_k{K}_data_summary.png
```

**Caveats:**
- `STATE_eval_set=NA` — no gene universe or pert-drop filter applied.
- No calibration split → single metrics/predictions file per run (no `_calib`/`_test` suffix).
- The zero-neighbour fallback is only triggered in co-path mode (not run here). In embedding modes, every test gene appears in at least one training pair (verified empirically for all 4 datasets), so `zero_neighbor` is always `False` in these outputs.

---

### 4. PerturbQA — cross-cell-line (GenePT only)

**Script:** `knn_classifier_transfer_perturbQA.sh` (8 array tasks: 4 held-out datasets × 2 k)  
**Python script:** `knn_classifier_transfer_perturbQA.py`  
**Output directory:** `KNN_perturbQA_cross_expr/`

**Data setup:**
- Training: ALL (perturbation, gene) pairs from the 3 source cell lines — both `train` and `test` splits — restricted to the 1337-gene common gene universe (`common_gene_universe_QA.csv`). No calibration split to exclude.
- Test: `test`-split pairs of the held-out cell line, restricted to the same 1337-gene universe.
- (pert, gene) duplicates across source cell lines resolved by majority vote before building the training dictionary.

**Output files** (`results/`):
```
{TestDS}_genept_k{K}_knn_transfer_metrics.csv
{TestDS}_genept_k{K}_knn_transfer_predictions.csv     ← gene-filtered at run time
{TestDS}_genept_k{K}_data_summary.png
```

**Filtered metrics:** `compute_cross_filtered_metrics.py` additionally drops perturbations in `dropped_test_perts_QA.json` from the already gene-filtered predictions, then recomputes metrics. Outputs to `KNN_perturbQA_cross_expr/cross_genept_metrics_filtered.csv` (`STATE_eval_set=True`).

**Caveats:**
- Gene universe filter is applied **at run time**; prediction files in `results/` are already restricted to 1337 common genes.
- Pert-drop filter is applied **post-hoc** by `compute_cross_filtered_metrics.py`; the raw prediction files still contain the dropped perturbations.
- GenePT predictions have been copied to `temp_predictions/cross_QA/{1NN,10NN}/` for convenience; those files are gene-filtered but NOT pert-drop-filtered (see that directory's README).
- Individual dataset gene counts before intersection: K562=4136, RPE1=4760, HepG2=7435, Jurkat=6842.

---

### 5. Sensitivity analysis — full dataset (all 4 embedding types)

Uses `--coverage_filter` on every run so all modes are evaluated on the same perturbation set (union drop set across ESM2 + WikiCrow + co-path).

#### 5a. Within-cell-line

**Script:** `knn_classifier_sensitivity.sh` (32 array tasks: 4 datasets × 4 modes × 2 k)  
**Output directory:** `KNN_sensitivity_full_within_expr/`

**Output files** (`results/`):
```
{DS}_{mode}_k{K}_covfilt_knn_metrics_{calib,test,combined}.csv
{DS}_{mode}_k{K}_covfilt_knn_predictions_{calib,test,combined}.csv
```

**Collected summary:** `collect_within_sensitivity_metrics.py` reads the `_combined` metrics CSV for each run and assembles `within_sensitivity_metrics.csv`. Also reads `_combined` predictions to compute `zero_neighbor_pcts.csv`.

**Caveats:**
- `STATE_eval_set=NA` — collected metrics come directly from the run-time metrics CSV (no additional gene/pert filter beyond `--coverage_filter`).
- Only the `combined` split appears in the collected CSV.

#### 5b. Cross-cell-line

**Script:** `knn_classifier_transfer_sensitivity.sh` (32 array tasks: 4 held-out datasets × 4 modes × 2 k)  
**Output directory:** `KNN_sensitivity_full_cross_expr/`

**Output files** (`results/`):
```
{TestDS}_{mode}_k{K}_covfilt_knn_transfer_metrics_{calib,test,combined}.csv
{TestDS}_{mode}_k{K}_covfilt_knn_transfer_predictions_{calib,test,combined}.csv
```

**Collected summary:** `collect_cross_sensitivity_metrics.py` reads predictions files (`_combined`), applies the common gene universe filter and dropped-pert filter, recomputes metrics, and outputs `cross_sensitivity_metrics.csv` and `zero_neighbor_pcts.csv`.

**Caveats:**
- `STATE_eval_set=True` — collected metrics are recomputed after gene/pert filtering.
- The pre-computed metrics CSVs in `results/` are unfiltered; they are not used by the collection script.

---

### 6. Sensitivity analysis — PerturbQA (all 4 embedding types)

Same `--coverage_filter` logic as the full-dataset sensitivity analysis.

#### 6a. Within-cell-line

**Script:** `knn_classifier_perturbQA_sensitivity.sh` (32 array tasks: 4 datasets × 4 modes × 2 k)  
**Output directory:** `KNN_sensitivity_perturbQA_within_expr/`

**Output files** (`results/`):
```
{DS}_{mode}_k{K}_covfilt_knn_metrics.csv
{DS}_{mode}_k{K}_covfilt_knn_predictions.csv
```

**Collected summary:** `collect_within_sensitivity_metrics_perturbQA.py` reads metrics and predictions, assembles `within_sensitivity_metrics.csv` and `zero_neighbor_pcts.csv`.

**Caveats:**
- `STATE_eval_set=NA` — no gene universe or pert-drop filter.
- No `_combined` suffix in filenames (no calib split in PerturbQA).
- Co-path mode is the only mode where `zero_neighbor=True` can appear. For embedding modes, every test gene appears in at least one training pair (verified empirically); `zero_neighbor` is always `False` in those files.

#### 6b. Cross-cell-line

**Script:** `knn_classifier_transfer_perturbQA_sensitivity.sh` (32 array tasks: 4 held-out datasets × 4 modes × 2 k)  
**Output directory:** `KNN_sensitivity_perturbQA_cross_expr/`

**Output files** (`results/`):
```
{TestDS}_{mode}_k{K}_covfilt_knn_transfer_metrics.csv
{TestDS}_{mode}_k{K}_covfilt_knn_transfer_predictions.csv   ← gene-filtered at run time
```

**Collected summary:** `collect_cross_sensitivity_metrics_perturbQA.py` reads prediction files, applies the 1337-gene universe filter and dropped-pert filter (`dropped_test_perts_QA.json`), recomputes metrics, and outputs `cross_sensitivity_metrics.csv` and `zero_neighbor_pcts.csv`.

**Caveats:**
- Gene universe filter is applied **at run time** (predictions in `results/` are already gene-filtered).
- Pert-drop filter is applied **post-hoc** in the collection script; `STATE_eval_set=True`.
- The pre-computed metrics CSVs in `results/` are unfiltered; they are not used by the collection script.

---

## Collection and filtering scripts

| Script | Reads from | Produces | `STATE_eval_set` |
|---|---|---|---|
| `collect_within_sensitivity_metrics.py` | `KNN_sensitivity_full_within_expr/results/` | `within_sensitivity_metrics.csv`, `zero_neighbor_pcts.csv` | NA |
| `collect_cross_sensitivity_metrics.py` | `KNN_sensitivity_full_cross_expr/results/` | `cross_sensitivity_metrics.csv`, `zero_neighbor_pcts.csv` | True |
| `collect_within_sensitivity_metrics_perturbQA.py` | `KNN_sensitivity_perturbQA_within_expr/results/` | `within_sensitivity_metrics.csv`, `zero_neighbor_pcts.csv` | NA |
| `collect_cross_sensitivity_metrics_perturbQA.py` | `KNN_sensitivity_perturbQA_cross_expr/results/` | `cross_sensitivity_metrics.csv`, `zero_neighbor_pcts.csv` | True |
| `compute_cross_filtered_metrics.py` | `KNN_full_cross_expr/results/`, `KNN_perturbQA_cross_expr/results/` | `cross_genept_metrics_filtered.csv` (in each dir) | True |

All scripts are run from `KNN_vote/`.

---

## Coverage check

```bash
python check_coverage.py
# Produces: coverage_summary.pdf
```

Reports the fraction of perturbations in each dataset covered by ESM2, WikiCrow, and co-path neighbours (GenePT assumed full). Shows a 4-bar group per dataset, with uncovered counts labelled.

---

## Data preparation utilities

```bash
python fetch_gene_descriptions.py  # fetch gene descriptions from NCBI/UniProt
python generate_embeddings.py      # generate OpenAI embeddings
python precompute_pathways.py      # pre-compute co-pathway Jaccard neighbours
```

---

## Utilities

- **`analysis_utility.py`** — `load_embedding_from_pkl()`, `load_copath_neighbors()`, `build_embedding_matrix()`, `find_k_nearest_neighbors()`; static drop sets `PERTS_NO_ESM2_EMBEDDING`, `PERTS_NO_WIKICROW_EMBEDDING`, `PERTS_NO_COPATH_NEIGHBORS`.
