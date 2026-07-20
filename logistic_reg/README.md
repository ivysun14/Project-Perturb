# Logistic Regression Baseline

Per-gene multinomial logistic regression for 3-class differential expression (DE) prediction (up / down / non-DE) in perturb-seq experiments.

Each perturbation is represented as a GenePT embedding vector (3072-d). A separate `LogisticRegression` model is trained for each gene independently, using the perturbation embeddings as features and that gene's DE labels as targets. To predict the label for a test (perturbation, gene) pair, the perturbation's embedding is passed through the gene's model, which outputs a 3-class probability via softmax.

All experiments use a fixed regularisation strength C = 0.01 (L2 penalty).

## Experiments

### 1. Full dataset — within-cell-line

Train on the 60% train split, report performance on the 40% test set within the same cell line.

```
python logistic_reg.py \
    --de_doc_file     ${DATA_DIR}/DE_DOC_K562_splitted.csv \
    --embedding_file  ${EMBED_DIR}/GenePT_gene_protein_embedding_model_3_text.pickle. \
    --embedding_file2 ${EMBED_DIR}/additional_genePT_embeddings.pkl \
    --dataset_name    K562_logreg \
    --output_dir      logreg_0.01_full_within_expr/results \
    --Cs 0.01 \
    --max_iter 2000
```

### 2. Full dataset — cross-cell-line

Train using the `train` and `test` splits of 3 non-target cell lines, report performance on the 40% test set of the target cell line.

```
# example: held-out cell line = K562, trained on the other 3
python logistic_reg.py \
    --train_de_doc_files ${DATA_DIR}/DE_DOC_RPE1_splitted.csv,${DATA_DIR}/DE_DOC_HepG2_splitted.csv,${DATA_DIR}/DE_DOC_Jurkat_splitted.csv \
    --test_de_doc_file   ${DATA_DIR}/DE_DOC_K562_splitted.csv \
    --embedding_file     ${EMBED_DIR}/GenePT_gene_protein_embedding_model_3_text.pickle. \
    --embedding_file2    ${EMBED_DIR}/additional_genePT_embeddings.pkl \
    --dataset_name       K562_logreg_transfer \
    --output_dir         logreg_0.01_full_cross_expr/results \
    --Cs 0.01 \
    --max_iter 2000

# recompute metrics after filtering to the common gene universe and dropping low-coverage perturbations
python metric_drop_perts.py
```

### 3. PerturbQA — within-cell-line

Train on the PerturbQA train split, report performance on the PerturbQA test split within the same cell line.

```
python logistic_reg_perturbQA.py \
    --de_file         ${PQADATA}/k562-de.csv \
    --dir_file        ${PQADATA}/k562-dir.csv \
    --dataset_name    K562 \
    --embedding_file  ${EMBED_DIR}/GenePT_gene_protein_embedding_model_3_text.pickle. \
    --embedding_file2 ${EMBED_DIR}/additional_genePT_embeddings.pkl \
    --output_dir      logreg_0.01_perturbQA_within_expr/results \
    --Cs 0.01 \
    --max_iter 2000
```

### 4. PerturbQA — cross-cell-line

Train using all (perturbation, gene) pairs from 3 non-target cell lines restricted to the common gene universe, report performance on the PerturbQA test split of the target cell line.

```
# example: held-out cell line = K562, trained on the other 3
python logistic_reg_perturbQA.py \
    --de_file_train        ${PQADATA}/rpe1-de.csv ${PQADATA}/hepg2-de.csv ${PQADATA}/jurkat-de.csv \
    --dir_file_train       ${PQADATA}/rpe1-dir.csv ${PQADATA}/hepg2-dir.csv ${PQADATA}/jurkat-dir.csv \
    --de_file_test         ${PQADATA}/k562-de.csv \
    --dir_file_test        ${PQADATA}/k562-dir.csv \
    --gene_universe_file   ${GENE_UNIVERSE} \
    --dataset_name_test    K562 \
    --embedding_file       ${EMBED_DIR}/GenePT_gene_protein_embedding_model_3_text.pickle. \
    --embedding_file2      ${EMBED_DIR}/additional_genePT_embeddings.pkl \
    --output_dir           logreg_0.01_perturbQA_cross_expr/results \
    --Cs 0.01 \
    --max_iter 2000

# recompute metrics after additionally dropping low-coverage test perturbations
python metric_drop_perts.py
```

## Utilities

- **`metric_drop_perts.py`** — recomputes cross-cell-line metrics after gene universe and pert-drop filtering for both full (`logreg_0.01_full_cross_expr/`) and PerturbQA (`logreg_0.01_perturbQA_cross_expr/`) experiments; outputs `cross_logreg_metrics_filtered.csv` in each experiment directory