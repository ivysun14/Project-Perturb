"""
main.py
=======

Entry script for training and evaluating the perturbation VAE model.
This script supports two experimental settings:

1. Cross-Cell-Line Generalization (mode="cross")
   - Hold out one entire cell line.
   - Train VAE on remaining cell lines.
   - Transfer perturbation effect to held-out cell line.

2. Within-Cell-Line Unseen Perturbation Prediction (mode="within")
   - Split perturbations within a cell line into train/test sets.
   - Train VAE on training perturbations.
   - Use embedding similarity to find nearest perturbations.
   - Transfer weighted latent delta to unseen perturbations.

"""

import anndata
import os
from utils import data_read_split, get_embeddings, train_valid_split, train_test_split, similarity_emb
from train_test import train_cell_line, reconstruct_cross_cell_line, reconstruct_within_cell_line

def main(z_dim, alpha,  cell_line, mode, n_epochs, batch_size, dropout_rate, learning_rate,train_continue=False, downsample=False, calibrate=False, seed=42):
    """
    Main experiment pipeline for perturbation VAE training and evaluation.

    Important args:
    alpha : float
        Weight of KL divergence term in VAE loss, Final loss = recon_loss + alpha * KL_loss.
    cell_line : str
        Target cell line for experiment.
    mode : str
        Experiment mode:"cross" or "within"
    train_continue : bool, default=False
        Whether to resume training from saved checkpoint.
    downsample : bool, default=False
        Whether to apply PerturbQA-based filtering (cross mode only).
    calibrate : bool, default=False
        Whether to apply zero calibration after reconstruction.
    """
    data_path_dict = {
        "K562": "path/to/h5ad",
        "RPE1": "path/to/h5ad",
        "Jurkat": "path/to/h5ad", 
        "HepG2": "path/to/h5ad",
    }
    emb_path = "path/to/embeddings/dict/pkl/file"
    csv_path = "./csv"

    if mode == "cross":
        csv_path = os.path.join(csv_path, f"{cell_line}.csv") if downsample else None # this csv is the result of PerturbQA-based filtering (executable/downsample.py)
        print("Constructing dataset and model...")
        adata_non_holdout, adata_holdout_ctrl, adata_holdout_pert = data_read_split(data_path_dict, cell_line, target_col="gene", control_name="non-targeting",
                                                                                    cell_line_key="cell_line", downsample=downsample, csv_path = csv_path)
        adata_train, adata_valid = train_valid_split(anndata.concat([adata_non_holdout, adata_holdout_ctrl],join="outer"),
                                                    0.05, seed, "gene", "cell_line", min_count_per_group = 50)
        pert_set = set(adata_non_holdout.obs["gene"])
        embeddings = get_embeddings(emb_path, pert_set)

        print("Start training...")
        train_cell_line(adata_train, adata_valid, 
                        z_dim, alpha, n_epochs, batch_size, dropout_rate, learning_rate,
                        cell_line, embeddings, mode, train_continue)
        
        print("Start Reconstructing for cross_cell_line data...")
        reconstruct_cross_cell_line(adata_non_holdout, adata_holdout_ctrl, adata_holdout_pert, z_dim, cell_line, embeddings, calibrate, seed)

    elif mode == "within":
        print("Constructing dataset and model...")
        adata = anndata.read_h5ad(data_path_dict[cell_line])
        adata.obs["cell_line"] = cell_line
        csv_path = os.path.join(csv_path, f"{cell_line}_splitted.csv") # this csv is the result of splitting perturbations into train/test sets (executable/train_test_split.py)
        adata_train, adata_test = train_test_split(adata, csv_path, "gene", "non-targeting")
        adata_train, adata_valid = train_valid_split(adata_train, 0.05, seed, "gene", "cell_line", min_count_per_group = 50)

        pert_set_train = set(adata_train.obs["gene"])
        pert_set_test = set(adata_test.obs["gene"])
        pert_union = set(pert_set_train) | set(pert_set_test)

        embeddings_train = get_embeddings(emb_path, pert_set_train)
        embeddings_test = get_embeddings(emb_path, pert_set_test)
        embeddings = get_embeddings(emb_path, pert_union)

        similarity_df = similarity_emb(embeddings_train, embeddings_test, n=20)

        print("Start training...")
        train_cell_line(adata_train, adata_valid, 
                        z_dim, alpha, n_epochs, batch_size, dropout_rate, learning_rate,
                        cell_line, embeddings_train, mode, train_continue)
        
        print("Start Reconstructing for within_cell_line data...")
        reconstruct_within_cell_line(adata_train, adata_test, similarity_df, z_dim, cell_line, embeddings, calibrate, seed)

    print("All done!")

if __name__ == '__main__':
    main(z_dim=64, alpha=0.05, cell_line="K562", mode="within",
         downsample=False, calibrate=False, train_continue=False, 
         n_epochs=100, batch_size=256, dropout_rate=0.05, learning_rate=0.0001, seed=42)