# KNN Vote

Weighted k-nearest neighbors (KNN) for 3-class differential expression (DE) prediction (up-regulated / down-regulated / non-DE) in perturb-seq experiments.

Perturbations are represented by 4 different embedding types:
- GenePT
- ESM2
- WikiCrow
- Co-path

Distance between perturbations is measured by cosine similarity. 

If j < k valid neighbors are found, the j available are used; if 0 neighbor is found, the prediction defaults to class 0 and the (perturbation, gene) pair is flagged `zero_neighbor=True`.

### Voting

Each valid neighbour casts a weighted vote: weight = cosine similarity (1 − cosine dissimilarity) for embedding modes, or Jaccard similarity for co-path. The class with the highest total weight wins; ties broken uniformly at random.

## Experiments

### 1. Full dataset — within-cell-line (GenePT only)

Train on the 60% train split using GenePT embeddings, report performance on the 40% test set within the same cell line.

```
# example: K562, k=10 (k ∈ {1, 10} both run)
python knn_classifier.py \
    --de_doc_file     ${DATA_DIR}/DE_DOC_K562_splitted.csv \
    --embedding_file  ${EMBED_DIR}/GenePT_gene_protein_embedding_model_3_text.pickle. \
    --embedding_file2 ${EMBED_DIR}/additional_genePT_embeddings.pkl \
    --embedding_mode  genept \
    --dataset_name    K562_GenePT_k10 \
    --output_dir      KNN_full_within_expr/results \
    --skip_cv --k     10
```

### 2. Full dataset — cross-cell-line (GenePT only)

Train using the `train` and `test` splits of 3 non-target cell lines (duplicate (pert, gene) pairs resolved by majority vote) with GenePT embeddings, report performance on the 40% test set of the target cell line.

```
# example: held-out cell line = K562, trained on the other 3, k=10
python knn_classifier_transfer.py \
    --train_de_doc_file  ${DATA_DIR}/DE_DOC_RPE1_splitted.csv ${DATA_DIR}/DE_DOC_HepG2_splitted.csv ${DATA_DIR}/DE_DOC_Jurkat_splitted.csv \
    --test_de_doc_file   ${DATA_DIR}/DE_DOC_K562_splitted.csv \
    --embedding_file     ${EMBED_DIR}/GenePT_gene_protein_embedding_model_3_text.pickle. \
    --embedding_file2    ${EMBED_DIR}/additional_genePT_embeddings.pkl \
    --embedding_mode     genept \
    --train_dataset_name RPE1 HepG2 Jurkat \
    --test_dataset_name  K562 \
    --k                  10 \
    --output_dir         KNN_full_cross_expr/results

# recompute metrics after filtering to the common gene universe and dropping low-coverage perturbations
python compute_cross_filtered_metrics.py
```

- Raw metrics in `results/` are unfiltered (`STATE_eval_set=NA`); use `cross_genept_metrics_filtered.csv` for filtered metrics.

### 3. PerturbQA — within-cell-line (GenePT only)

Train on the PerturbQA train split using GenePT embeddings, report performance on the PerturbQA test split within the same cell line.

```
# example: K562, k=10 (k ∈ {1, 10} both run)
python knn_classifier_perturbQA.py \
    --de_file         ${PQADATA}/k562-de.csv \
    --doc_file        ${PQADATA}/k562-dir.csv \
    --embedding_file  ${EMBED_DIR}/GenePT_gene_protein_embedding_model_3_text.pickle. \
    --embedding_file2 ${EMBED_DIR}/additional_genePT_embeddings.pkl \
    --dataset_name    K562_GenePT_k10 \
    --output_dir      KNN_perturbQA_within_expr/results \
    --skip_cv --k     10
```

### 4. PerturbQA — cross-cell-line (GenePT only)

Train using all (perturbation, gene) pairs from 3 non-target cell lines restricted to the common gene universe with GenePT embeddings, report performance on the PerturbQA test split of the target cell line.

```
# example: held-out cell line = K562, trained on the other 3, k=10
python knn_classifier_transfer_perturbQA.py \
    --train_de_file       ${PQADATA}/rpe1-de.csv ${PQADATA}/hepg2-de.csv ${PQADATA}/jurkat-de.csv \
    --train_dir_file      ${PQADATA}/rpe1-dir.csv ${PQADATA}/hepg2-dir.csv ${PQADATA}/jurkat-dir.csv \
    --train_dataset_name  RPE1 HepG2 Jurkat \
    --test_de_file        ${PQADATA}/k562-de.csv \
    --test_dir_file       ${PQADATA}/k562-dir.csv \
    --test_dataset_name   K562 \
    --gene_universe_file  ${GENE_UNIVERSE} \
    --embedding_file      ${EMBED_DIR}/GenePT_gene_protein_embedding_model_3_text.pickle. \
    --embedding_file2     ${EMBED_DIR}/additional_genePT_embeddings.pkl \
    --embedding_mode      genept \
    --output_dir          KNN_perturbQA_cross_expr/results \
    --k                   10

# recompute metrics after additionally dropping low-coverage test perturbations
python compute_cross_filtered_metrics.py
```

- Gene universe filter is applied at run time; prediction files in `results/` are already restricted to the common gene universe.

### 5. Sensitivity analysis — full dataset (all 4 embedding types)

Uses `--coverage_filter` on every run so all modes are evaluated on the same perturbation set (union drop set across ESM2 + WikiCrow + co-path).

#### 5a. Within-cell-line

Same as Experiment 1, restricted to perturbations covered by all 4 embedding types.

```
# example: K562, GenePT mode, k=10
python knn_classifier.py \
    --de_doc_file     ${DATA_DIR}/DE_DOC_K562_splitted.csv \
    --embedding_file  ${EMBED_DIR}/GenePT_gene_protein_embedding_model_3_text.pickle. \
    --embedding_file2 ${EMBED_DIR}/additional_genePT_embeddings.pkl \
    --embedding_mode  genept \
    --dataset_name    K562_genept_k10_covfilt \
    --output_dir      KNN_sensitivity_full_within_expr/results \
    --skip_cv --k     10 \
    --coverage_filter

# swap --embedding_file (and --embedding_mode/--copath_file) for esm2, wikicrow, copath; then collect all 4 modes into one summary
python collect_within_sensitivity_metrics.py
```

- `STATE_eval_set=NA` in the collected summary (no gene/pert filter beyond `--coverage_filter`); only the `combined` split appears.

#### 5b. Cross-cell-line

Same as Experiment 2, restricted to perturbations covered by all 4 embedding types.

```
# example: held-out cell line = K562, trained on the other 3, GenePT mode, k=10
python knn_classifier_transfer.py \
    --train_de_doc_file  ${DATA_DIR}/DE_DOC_RPE1_splitted.csv ${DATA_DIR}/DE_DOC_HepG2_splitted.csv ${DATA_DIR}/DE_DOC_Jurkat_splitted.csv \
    --test_de_doc_file   ${DATA_DIR}/DE_DOC_K562_splitted.csv \
    --train_dataset_name RPE1 HepG2 Jurkat \
    --test_dataset_name  K562 \
    --embedding_file     ${EMBED_DIR}/GenePT_gene_protein_embedding_model_3_text.pickle. \
    --embedding_file2    ${EMBED_DIR}/additional_genePT_embeddings.pkl \
    --embedding_mode     genept \
    --k                  10 \
    --output_dir         KNN_sensitivity_full_cross_expr/results \
    --coverage_filter

# swap the embedding flags for esm2, wikicrow, copath; recompute metrics with gene/pert filtering and collect all 4 modes into one summary
python collect_cross_sensitivity_metrics.py
```

- `STATE_eval_set=True` in the collected summary (metrics recomputed after gene/pert filtering); pre-computed metrics CSVs in `results/` are unfiltered and unused by the collection script.

### 6. Sensitivity analysis — PerturbQA (all 4 embedding types)

#### 6a. Within-cell-line

Same as Experiment 3, restricted to perturbations covered by all 4 embedding types.

```
# example: K562, GenePT mode, k=10
python knn_classifier_perturbQA.py \
    --de_file         ${PQADATA}/k562-de.csv \
    --doc_file        ${PQADATA}/k562-dir.csv \
    --embedding_file  ${EMBED_DIR}/GenePT_gene_protein_embedding_model_3_text.pickle. \
    --embedding_file2 ${EMBED_DIR}/additional_genePT_embeddings.pkl \
    --dataset_name    K562_genept_k10_covfilt \
    --output_dir      KNN_sensitivity_perturbQA_within_expr/results \
    --skip_cv --k     10 \
    --coverage_filter

# swap --embedding_file for esm2/--esm2_file, wikicrow/--wikicrow_file, copath/--copath_file; then collect all 4 modes into one summary
python collect_within_sensitivity_metrics_perturbQA.py
```

#### 6b. Cross-cell-line

Same as Experiment 4, restricted to perturbations covered by all 4 embedding types.

```
# example: held-out cell line = K562, trained on the other 3, GenePT mode, k=10
python knn_classifier_transfer_perturbQA.py \
    --train_de_file       ${PQADATA}/rpe1-de.csv ${PQADATA}/hepg2-de.csv ${PQADATA}/jurkat-de.csv \
    --train_dir_file      ${PQADATA}/rpe1-dir.csv ${PQADATA}/hepg2-dir.csv ${PQADATA}/jurkat-dir.csv \
    --train_dataset_name  RPE1 HepG2 Jurkat \
    --test_de_file        ${PQADATA}/k562-de.csv \
    --test_dir_file       ${PQADATA}/k562-dir.csv \
    --test_dataset_name   K562 \
    --gene_universe_file  ${GENE_UNIVERSE} \
    --embedding_file      ${EMBED_DIR}/GenePT_gene_protein_embedding_model_3_text.pickle. \
    --embedding_file2     ${EMBED_DIR}/additional_genePT_embeddings.pkl \
    --embedding_mode      genept \
    --output_dir          KNN_sensitivity_perturbQA_cross_expr/results \
    --k                   10 \
    --coverage_filter

# swap the embedding flags for esm2, wikicrow, copath; recompute metrics with gene/pert filtering and collect all 4 modes into one summary
python collect_cross_sensitivity_metrics_perturbQA.py
```

- Gene universe filter applied at run time; pert-drop filter applied post-hoc in the collection script (`STATE_eval_set=True`).

## Other Scripts

```bash
python check_coverage.py
# Produces: coverage_summary.pdf
```

Reports the fraction of perturbations in each dataset covered by ESM2, WikiCrow, and co-path neighbours (GenePT assumed full). Shows a 4-bar group per dataset, with uncovered counts labelled.

```bash
python fetch_gene_descriptions.py  # fetch gene descriptions from NCBI/UniProt
python generate_embeddings.py      # generate OpenAI embeddings
python precompute_pathways.py      # pre-compute co-pathway Jaccard neighbours
```

## Utilities

- **`analysis_utility.py`** — `load_embedding_from_pkl()`, `load_copath_neighbors()`, `build_embedding_matrix()`, `find_k_nearest_neighbors()`; static drop sets `PERTS_NO_ESM2_EMBEDDING`, `PERTS_NO_WIKICROW_EMBEDDING`, `PERTS_NO_COPATH_NEIGHBORS`.