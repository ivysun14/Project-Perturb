from dataset import Dataset_vae
from solver import PerTrainer
from vae import VAEArith
import anndata
import os, pickle


def _replace_controls_for_export(adata, full_control_adata, target_col, control_name):
    if full_control_adata is None:
        return adata

    non_control_mask = adata.obs[target_col].astype(str) != control_name
    pred_only = adata[non_control_mask].copy()
    control_export = full_control_adata.copy()
    control_export.obs[target_col] = control_name
    return anndata.concat(
        [pred_only, control_export],
        axis=0,
        join="inner",
        merge="same",
        index_unique=None,
    )

def train_cell_line(adata_train, adata_valid, log, z_dim, alpha, n_epochs, batch_size, dropout_rate,
                          learning_rate, cell_line, embeddings, mode, train_continue=False, run_dir=None):
    if run_dir is not None:
        model_path = os.path.join(run_dir, "models")
    elif mode == "cross":
        model_path = f"./results/models/{cell_line}_holdout"
    elif mode == "within":
        model_path = f"./results/models/{cell_line}_within"
    
    log.info(f"Training Mode: {mode}")
    log.info(f"Target Cell Line: {cell_line}")

    log.info("[Train Dataset]")
    log.info(f"Number of cells: {adata_train.n_obs}")
    log.info(f"Number of genes: {adata_train.n_vars}")
    log.info(f"Data shape: {adata_train.shape}")
    log.info(f"Cell line distribution:\n{adata_train.obs['cell_line'].value_counts()}")

    log.info("[Valid Dataset]")
    log.info(f"Number of cells: {adata_valid.n_obs}")
    log.info(f"Number of genes: {adata_valid.n_vars}")
    log.info(f"Data shape: {adata_valid.shape}")
    log.info(f"Cell line distribution:\n{adata_valid.obs['cell_line'].value_counts()}")

    train_dataset = Dataset_vae(adata_train, target_col="gene", control_name="non-targeting", embeddings=embeddings)
    valid_dataset = Dataset_vae(adata_valid, target_col="gene", control_name="non-targeting", embeddings=embeddings)
    VAE = VAEArith(x_dimension=adata_train.X.shape[1], z_dimension=z_dim, emb_dimension=3072, alpha=alpha,
                   dropout_rate=dropout_rate, restore=train_continue, model_path=model_path)
    trainer = PerTrainer(VAE, train_set=train_dataset, valid_set=valid_dataset, log=log, learning_rate=learning_rate, model_path=model_path)
    trainer.train(n_epochs=n_epochs, batch_size=batch_size)
    print("Training done!")

def reconstruct_cross_cell_line(
    adata_within,
    adata_ctrl,
    adata_to_predict,
    log,
    similarity_df,
    z_dimension,
    cell_hold_out,
    embeddings,
    seed,
    output_tag=None,
    run_dir=None,
):
    if run_dir is not None:
        model_path = os.path.join(run_dir, "models")
    else:
        model_path = f"./results/models/{cell_hold_out}_holdout"

    log.info("[Test Dataset]")
    log.info(f"Number of cells: {adata_to_predict.n_obs}")
    log.info(f"Number of genes: {adata_to_predict.n_vars}")
    log.info(f"Data shape: {adata_to_predict.shape}")
    log.info(f"Cell line distribution:\n{adata_to_predict.obs['cell_line'].value_counts()}")

    network = VAEArith(x_dimension=adata_within.X.shape[1], z_dimension=z_dimension, emb_dimension=3072,
                       ez_dimension=z_dimension, restore=True, model_path=model_path)
    trainer = PerTrainer(network, model_path=model_path, log=log)
    delta_dict, adata = trainer.predict_across_cell(adata_within, adata_ctrl, adata_to_predict, similarity_df, cell_line_key="cell_line",
                                        target_col="gene", control_name="non-targeting", embeddings=embeddings, seed=seed)
    log.info(f"Finish Reconstructing for OOD cell line!")
    out_dir = os.path.join(run_dir, "prediction") if run_dir is not None else os.path.join(f"./results/predictions")
    os.makedirs(out_dir, exist_ok=True)
    suffix = f"_cross_{output_tag}" if output_tag else "_cross"
    out_path = f"{out_dir}/{cell_hold_out}{suffix}.h5ad"
    adata.write_h5ad(out_path)
    # out_path = f"{out_dir}/{cell_hold_out}_delta.pkl"
    # with open(out_path, "wb") as f:
    #     pickle.dump(delta_dict, f)

def reconstruct_cross_cell_line_calib(
    adata_train,
    adata_calibration,
    log,
    similarity_df,
    z_dimension,
    cell_hold_out,
    embeddings,
    seed,
    output_tag=None,
    run_dir=None,
):
    if run_dir is not None:
        model_path = os.path.join(run_dir, "models")
    else:
        model_path = f"./results/models/{cell_hold_out}_holdout"

    log.info("[Calibration Dataset]")
    log.info(f"Number of cells: {adata_calibration.n_obs}")
    log.info(f"Number of genes: {adata_calibration.n_vars}")
    log.info(f"Data shape: {adata_calibration.shape}")
    log.info(f"Cell line distribution:\n{adata_calibration.obs['cell_line'].value_counts()}")
    target_col = "gene"
    control_name = "non-targeting"
    adata_ctrl = adata_train[adata_train.obs[target_col].astype(str) == control_name].copy()

    network = VAEArith(x_dimension=adata_train.X.shape[1], z_dimension=z_dimension, emb_dimension=3072,
                       ez_dimension=z_dimension, restore=True, model_path=model_path)
    trainer = PerTrainer(network, model_path=model_path, log=log)
    delta_dict, adata = trainer.predict_across_cell(adata_train, adata_ctrl, adata_calibration, similarity_df, cell_line_key="cell_line",
                                        target_col="gene", control_name="non-targeting", embeddings=embeddings, seed=seed)
    log.info(f"Finish Reconstructing for calibration set")
    out_dir = os.path.join(run_dir, "prediction") if run_dir is not None else os.path.join(f"./results/predictions")
    os.makedirs(out_dir, exist_ok=True)
    pred_cells = sorted(adata.obs["cell_line"].astype(str).unique())
    for pred_cell in pred_cells:
        sub = adata[adata.obs["cell_line"].astype(str) == pred_cell, :].copy()
        out_path = f"{out_dir}/{cell_hold_out}_{pred_cell}_calibration.h5ad"
        sub.write_h5ad(out_path)

def reconstruct_within_cell_line(
    adata_train,
    adata_test,
    log,
    similarity_df,
    z_dimension,
    cell_line,
    embeddings,
    seed,
    output_tag=None,
    full_control_adata=None,
    run_dir=None,
):
    if run_dir is not None:
        model_path = os.path.join(run_dir, "models")
    else:
        model_path = f"./results/models/{cell_line}_within"

    log.info("[Test Dataset]")
    log.info(f"Number of cells: {adata_test.n_obs}")
    log.info(f"Number of genes: {adata_test.n_vars}")
    log.info(f"Data shape: {adata_test.shape}")
    log.info(f"Cell line distribution:\n{adata_test.obs['cell_line'].value_counts()}")

    network = VAEArith(x_dimension=adata_test.X.shape[1], z_dimension=z_dimension, emb_dimension=3072,
                       ez_dimension=z_dimension, restore=True, model_path=model_path)
    trainer = PerTrainer(network, model_path=model_path, log=log)
    pred_adata, true_adata = trainer.predict_within_cell(adata_train, adata_test, similarity_df,
                                        target_col="gene", control_name="non-targeting", embeddings=embeddings, seed=seed)
    log.info(f"Finish Reconstructing for OOD perturbation!")

    if full_control_adata is not None:
        pred_adata = _replace_controls_for_export(
            pred_adata,
            full_control_adata,
            target_col="gene",
            control_name="non-targeting",
        )
        true_adata = _replace_controls_for_export(
            true_adata,
            full_control_adata,
            target_col="gene",
            control_name="non-targeting",
        )
        log.info(
            "Replaced exported controls with full control set: %d cells",
            full_control_adata.n_obs,
        )

    out_dir = os.path.join(run_dir, "prediction") if run_dir is not None else os.path.join(f"./results/predictions")
    os.makedirs(out_dir, exist_ok=True)
    suffix = f"_within_{output_tag}" if output_tag else "_within"
    out_path_pred = f"{out_dir}/{cell_line}{suffix}.h5ad"
    out_path_true = f"{out_dir}/{cell_line}{suffix}_true.h5ad"
    pred_adata.write_h5ad(out_path_pred)
    true_adata.write_h5ad(out_path_true)
