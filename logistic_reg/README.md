# Logistic Regression Baseline

Per-gene multinomial logistic regression for 3-class differential expression (DE) prediction (up / down / non-DE) in perturb-seq experiments.

## Algorithm

Each perturbation is represented as a GenePT embedding vector (3072-d). A separate logistic regression model is trained for each gene independently, using the perturbation embeddings as features and that gene's DE labels as targets. To predict the label for a test (perturbation, gene) pair, the perturbation's embedding is passed through the gene's model, which outputs a 3-class probability via softmax.

**Edge cases:**

| Situation | Handling |
|---|---|
| Gene has < 2 training observations | Skipped; excluded from test predictions |
| Gene has only one class in training | Always predicts that class with probability 1 |
| Gene model saw only 2 of 3 classes | Probability for the unseen class is set to 0 at prediction time |

## Modes

### Within-cell-line
Train and test on splits from the same cell line. Pass a single CSV with a `split` column.

### Cross-cell-line
Train on the `train` splits of N cell lines, test on the `test` split of a held-out cell line. When the same `(perturbation, gene)` pair appears in multiple training sets, labels are resolved by **majority vote**. Ties are broken at random (seeded by `--random_seed`) and printed as warnings.

## Scripts

### Within-cell-line — fixed C (no cross-validation)

```bash
python -u logistic_reg.py \
    --de_doc_file     ../data/DE_DOC_hepg2_splitted.csv \
    --embedding_file  ../KNN_vote/data/GenePT_gene_protein_embedding_model_3_text.pickle \
    --embedding_file2 ../KNN_vote/data/additional_genePT_embeddings.pkl \
    --dataset_name    hepg2 \
    --output_dir      results \
    --Cs              0.01 \
    --max_iter        1000 \
    --random_seed     305
```

### Within-cell-line — CV-selected C (longer run time)

```bash
python -u logistic_reg.py \
    --de_doc_file     ../data/DE_DOC_jurkat_splitted.csv \
    --embedding_file  ../KNN_vote/data/GenePT_gene_protein_embedding_model_3_text.pickle \
    --embedding_file2 ../KNN_vote/data/additional_genePT_embeddings.pkl \
    --dataset_name    jurkat \
    --output_dir      results \
    --Cs              0.01,0.1,1.0,10.0 \
    --cv              5 \
    --max_iter        1000 \
    --random_seed     305
```

### Cross-cell-line — train on 3 cell lines, test on the 4th

```bash
python -u logistic_reg.py \
    --train_de_doc_files ../data/DE_DOC_jurkat_splitted.csv,../data/DE_DOC_K562_Replogle_splitted.csv,../data/DE_DOC_rpe1_splitted.csv \
    --test_de_doc_file   ../data/DE_DOC_hepg2_splitted.csv \
    --embedding_file     ../KNN_vote/data/GenePT_gene_protein_embedding_model_3_text.pickle \
    --embedding_file2    ../KNN_vote/data/additional_genePT_embeddings.pkl \
    --dataset_name       cross_hepg2_holdout \
    --output_dir         results \
    --Cs                 0.01 \
    --max_iter           1000 \
    --random_seed        305
```

To switch the hold-out cell line, change `--train_de_doc_files` and `--test_de_doc_file` accordingly. The `--train_de_doc_files` argument accepts a comma-separated list of any number of files.

## Outputs

Saved to `--output_dir`:

- `{dataset_name}_logreg_predictions.csv` — per-pair predictions, class probabilities, and `best_C` (the C used for that gene's model; NaN when C is fixed)
- `{dataset_name}_logreg_metrics.csv` — overall and per-class metrics (accuracy, balanced accuracy, precision, recall, F1, AUROC)
