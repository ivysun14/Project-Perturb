# Perturbation-transfer VAE Baseline

## Overview

This is a conditional VAE perturbation prediction baseline inspired by [scGen](https://github.com/theislab/scgen). Given control cells and perturbation embeddings, the model learns perturbation effects in latent space and transfers them to either a held-out cell line or held-out perturbations.

For detailed script arguments, run:

```bash
python main.py --help
```

## Prerequisites

Data inputs:

- One or more `.h5ad` files named `<CELL_LINE>_raw.h5ad` under `--dataset_dir` with expression matrix in `.X`.
- Perturbation labels in `adata.obs["gene"]`; controls should be named `non-targeting`.
- A split CSV with one row per perturbation. It must contain `pert` and a split column such as `split_calib`.
- Perturbation embedding pickle files.

Default paths:

```bash
--dataset_dir data/h5ad_objects
--split_csv_path "data/csv/DE_DOC_{cell}_splitted.csv"
--embedding_path data/embedding/GenePT_gene_protein_embedding_model_3_text.pickle.
--missing_embedding_path data/embedding/additional_genePT_embeddings.pkl
--output_root results
```

Python/package requirements:

- Python 3.10+ is recommended.
- Required packages include `torch`, `anndata`, `scanpy`, `numpy`, `pandas`, and `scipy`.

## Train

Cross-cell-line training:

```bash
python main.py --cell_line HepG2 --mode cross --stage train
```

Within-cell-line training:

```bash
python main.py --cell_line Jurkat --mode within --stage train \
  --split_set_train train \
  --split_set_test test
```

PerturbQA-style runs use the same entry point with an alternate split CSV and split column:

```bash
python main.py --cell_line RPE1 --mode cross --stage train \
  --split_csv_path "data/perturbqa_datasets/{cell_lower}-de.csv" \
  --split_column split \
  --split_set_test test \
  --split_set_calibration "" \
  --use_split_gene_space true \
  --dropped_test_perts_json /path/to/dropped_test_perts_QA.json
```

Use explicit paths when the data is not under the default folders:

```bash
python main.py --cell_line HepG2 --mode cross --stage train \
  --dataset_dir /path/to/h5ad_objects \
  --split_csv_path "/path/to/csv/DE_DOC_{cell}_splitted.csv" \
  --embedding_path /path/to/embedding/GenePT_gene_protein_embedding_model_3_text.pickle. \
  --missing_embedding_path /path/to/embedding/additional_genePT_embeddings.pkl \
  --output_root /path/to/results
```

## Test

Cross-cell-line test prediction on the held-out cell line:

```bash
python main.py --cell_line HepG2 --mode cross --stage test
```

Within-cell-line test prediction:

```bash
python main.py --cell_line Jurkat --mode within --stage test \
  --split_set_train train \
  --split_set_test test
```

PerturbQA-style prediction is also a split override:

```bash
python main.py --cell_line RPE1 --mode cross --stage test \
  --split_csv_path "data/perturbqa_datasets/{cell_lower}-de.csv" \
  --split_column split \
  --split_set_test test \
  --use_split_gene_space true
```

If training used custom paths, pass the same path arguments again for testing.

## *Prediction on Calibration Set

This run exports predictions on perturbations in the calibration set for the follow-up post-calibration step.

Cross-cell-line prediction on calibration set:

```bash
python main.py --cell_line HepG2 --mode cross --stage calibration
```

Within-cell-line prediction on calibration set uses `--stage test` with `--split_set_test calibration`:

```bash
python main.py --cell_line Jurkat --mode within --stage test \
  --split_set_train train \
  --split_set_test calibration
```

## Output

Each run writes to:

```text
<output_root>/<CELL_LINE>_<mode>/
```

Main outputs:

```text
models/best_model.pt
prediction/<CELL_LINE>_cross_test.h5ad
prediction/<CELL_LINE>_within_test.h5ad
prediction/<CELL_LINE>_within_test_true.h5ad
```
