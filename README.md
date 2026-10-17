# Benchmarking Differential Expression Classification in Single-Cell Perturbation Models

This repository contains the code to reproduce the analyses in the manuscript. We benchmark how well
perturbation models predict which genes are differentially expressed (up, down or non-DE) after a
genetic perturbation, comparing simple baselines with deep learning models on four Perturb-seq
datasets (K562, RPE1, HepG2, Jurkat), within and across cell lines.

## Environment

From the repository root:

```
conda env create -f environment.yml
conda activate Project-Perturb
```

VAE and scVI list additional package requirements in their READMEs.

## Repository layout

Method and analysis details and commands are in each folder's README.

| Folder | Contents |
| --- | --- |
| `preprocessing/` | DE calling and train / calibration / test splits shared by all models |
| `majority_vote/` | Perturbation majority vote |
| `KNN_vote/` | KNN classifier |
| `logistic_reg/` | Regularized logistic regression |
| `Scouter/` | Scouter |
| `TranScouter/` | TranScouter |
| `VAE/` | Variational autoencoder |
| `scVI/` | scVI |
| `STATE/` | STATE |
| `metrics/` | Evaluation metrics |
| `calibration/` | Post-prediction distributional calibration |
| `EDA/` | Exploratory data analysis |

## Data access

Raw K562, RPE1, Jurkat, and HepG2 essential gene screen h5ad files and processed train-test splits with DE labels can be downloaded [here](https://drive.google.com/file/d/1ZVLPp0dw3URB-kq_xM1S4ULOsHN67ZRt/view?usp=drive_link). PerturbQA downsampled data are available for download [here](https://drive.google.com/file/d/1DQdo-2qhRusCYytc29hToiiUbireV3xb/view?usp=drive_link) or at the original PerturbQA paper's [github repo](https://github.com/genentech/PerturbQA). Embeddings used in this study can be downloaded [here](https://drive.google.com/drive/folders/19UTD1Znm4iYnikyUjnHZ7nM4bChTnyeE?usp=drive_link).
