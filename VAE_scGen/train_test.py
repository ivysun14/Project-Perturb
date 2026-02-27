from calibration import calibrate_zeros
from dataset import Dataset_vae
from solver import PerTrainer
from models.vae_ import VAEArith
import os

def train_cell_line(adata_train, adata_valid, z_dim, alpha, n_epochs, batch_size, dropout_rate,
                          learning_rate, cell_line, embeddings, mode, train_continue=False):
    if mode == "cross":
        model_path = f"./results/models/{cell_line}_holdout"
    elif mode == "within":
        model_path = f"./results/models/{cell_line}_within"
    train_dataset = Dataset_vae(adata_train, target_col="gene", control_name="non-targeting", embeddings=embeddings)
    valid_dataset = Dataset_vae(adata_valid, target_col="gene", control_name="non-targeting", embeddings=embeddings)
    VAE = VAEArith(x_dimension=adata_train.X.shape[1], z_dimension=z_dim, emb_dimension=3072, alpha=alpha,
                   dropout_rate=dropout_rate, restore=train_continue, model_path=model_path)
    trainer = PerTrainer(VAE, train_set=train_dataset, valid_set=valid_dataset, learning_rate=learning_rate, model_path=model_path)
    trainer.train(n_epochs=n_epochs, batch_size=batch_size)
    print("Training done!")

def reconstruct_cross_cell_line(adata_within, adata_ctrl, adata_to_predict ,z_dimension, cell_hold_out, embeddings, calibrate, seed):
    model_path = f"./results/models/{cell_hold_out}_holdout"
    network = VAEArith(x_dimension=adata_within.X.shape[1], z_dimension=z_dimension, emb_dimension=3072,
                       ez_dimension=z_dimension, restore=True, model_path=model_path)
    trainer = PerTrainer(network, model_path=model_path)
    adata = trainer.predict_across_cell(adata_within, adata_ctrl, adata_to_predict, cell_line_key="cell_line",
                                             target_col="gene", control_name="non-targeting", embeddings=embeddings, seed=seed)
    print(f"Finish Reconstructing for OOD cell line!")

    out_dir = os.path.join(f"./results/predictions")
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{cell_hold_out}_holdout.h5ad"
    adata.write_h5ad(out_path)
    if calibrate:
        adata = calibrate_zeros(adata, target_col="gene", control_name="non-targeting", ratio=0.8)
        out_path = f"{out_dir}/{cell_hold_out}_holdout_calibrated.h5ad"
        adata.write_h5ad(out_path)

def reconstruct_within_cell_line(adata_train, adata_test, similarity_df, z_dimension, cell_line, embeddings, calibrate, seed):
    model_path = f"./results/models/{cell_line}_within"
    network = VAEArith(x_dimension=adata_test.X.shape[1], z_dimension=z_dimension, emb_dimension=3072,
                       ez_dimension=z_dimension, restore=True, model_path=model_path)
    trainer = PerTrainer(network, model_path=model_path)
    adata = trainer.predict_within_cell(adata_train, adata_test, similarity_df,
                                        target_col="gene", control_name="non-targeting", embeddings=embeddings, seed=seed)
    print(f"Finish Reconstructing for OOD perturbation!")

    out_dir = os.path.join(f"./results/predictions")
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{cell_line}_within.h5ad"
    adata.write_h5ad(out_path)
    if calibrate:
        adata = calibrate_zeros(adata, target_col="gene", control_name="non-targeting", ratio=0.8)
        out_path = f"{out_dir}/{cell_line}_within_calibrated.h5ad"
        adata.write_h5ad(out_path)