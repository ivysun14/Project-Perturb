The scVI implementation code is inspired by
[ArcInstitute/state-reproduce baselines](https://github.com/ArcInstitute/state-reproduce/tree/main/baselines).

# scVI Baseline

## Overview

This is a perturbation prediction baseline based on scVI. Given a basal/control cell expression profile, a perturbation identity, and optional cell-line/batch covariates, the model predicts the gene expression profile of the perturbed cell. Prediction can be exported either as the distribution mean or as sampled counts from the learned ZINB distribution.

For detailed script arguments, run:

```bash
python scripts/train.py --help
python scripts/predict.py --help
python scripts/predict_calib.py --help
```

## Prerequisites

Data inputs:

- One or more `.h5ad` files named `<CELL_LINE>_raw.h5ad` with expression matrix in `.X`.
- A split CSV with one row per perturbation. It must contain `pert` and a split column.
- Optional perturbation embeddings.

Default paths:

```bash
--dataset-dir data/h5ad_objects
--split-csv "data/csv/DE_DOC_<CELL_LINE>_splitted.csv"
--output-dir outputs
```

Each run writes to `<output-dir>/<run-name>/`; the default run name is `<CELL_LINE>_<mode>`.

Optional perturbation embeddings can be passed explicitly or through an environment variable:

```bash
--pert-embedding-path data/embedding/GenePT_gene_protein_embedding_model_3_text.pickle. \
--pert-embedding-path data/embedding/additional_genePT_embeddings.pkl
```

```bash
export SCVI_PERT_EMBEDDING_PATHS="data/embedding/GenePT_gene_protein_embedding_model_3_text.pickle.,data/embedding/additional_genePT_embeddings.pkl"
```

You can also change defaults with environment variables:

```bash
export SCVI_DATA_ROOT=/path/to/data
export SCVI_DATASET_DIR=/path/to/h5ad_objects
export SCVI_SPLIT_CSV=/path/to/csv/DE_DOC_HepG2_splitted.csv
export SCVI_SPLIT_COLUMN=split_calib
export SCVI_OUTPUT_ROOT=/path/to/run_root
```

Python/package requirements:

- Python 3.10+ is recommended.
- Required packages include `torch`, `lightning`, `hydra-core`, `omegaconf`,
  `anndata`, `scanpy`, `numpy`, `pandas`, `pyyaml`, `tqdm`, `h5py`,
  `torchmetrics`, `transformers`, and `cell_load`.

## Train

Cross-cell-line training:

```bash
python scripts/train.py --cell-line HepG2 --mode cross
```

Within-cell-line training:

```bash
python scripts/train.py --cell-line Jurkat --mode within \
  --train-split-groups train \
  --test-split-groups test
```

PerturbQA-style runs use the same entry point with an alternate split CSV and split column. Add other QA-specific options as needed:

```bash
python scripts/train.py --cell-line RPE1 --mode cross \
  --split-csv data/perturbqa_datasets/rpe1-de.csv \
  --split-column split \
  --test-split-groups test \
  --calib-split-groups "" \
  --use-split-gene-space true \
  --dropped-test-perts-json /path/to/dropped_test_perts_QA.json
```

Use explicit paths when the data is not under the default folders:

```bash
python scripts/train.py --cell-line HepG2 --mode cross \
  --dataset-dir /path/to/h5ad_objects \
  --split-csv /path/to/csv/DE_DOC_HepG2_splitted.csv \
  --output-dir /path/to/outputs \
  --pert-embedding-path /path/to/embedding/GenePT_gene_protein_embedding_model_3_text.pickle. \
  --pert-embedding-path /path/to/embedding/additional_genePT_embeddings.pkl
```

## Test

The `--prediction-modes` argument controls which prediction export to run:
`mean` exports deterministic distribution-mean predictions, `sampling` exports
sampled-count predictions, and `both` runs both modes. The default is `mean`.

Cross-cell-line test prediction:

```bash
python scripts/predict.py --cell-line HepG2 --mode cross \
  --prediction-modes both
```

Within-cell-line test prediction:

```bash
python scripts/predict.py --cell-line Jurkat --mode within \
  --prediction-modes both
```

PerturbQA-style prediction is also just a split override:

```bash
python scripts/predict.py --cell-line RPE1 --mode cross \
  --split-csv data/perturbqa_datasets/rpe1-de.csv \
  --split-column split \
  --test-split-groups test \
  --use-split-gene-space true \
  --prediction-modes both
```

If training used custom paths, pass the same `--output-dir`/`--run-name` or a concrete `--run-dir` for prediction:

```bash
python scripts/predict.py --cell-line HepG2 --mode cross \
  --output-dir /path/to/outputs \
  --split-csv /path/to/csv/DE_DOC_HepG2_splitted.csv \
  --prediction-modes both
```

## Prediction on Calibration Set

This run is to export predictions on perturbations in the calibration set (with split label `calibration`) for the follow-up post-calibration.

Cross-cell-line prediction on calibration set:

```bash
python scripts/predict_calib.py --cell-line HepG2 \
  --prediction-modes both
```

Within-cell-line prediction on calibration set comes from `scripts/predict.py` :

```bash
python scripts/predict.py --cell-line Jurkat --mode within \
  --run-name Jurkat_within_calib \
  --test-split-groups calibration \
  --prediction-modes both
```
