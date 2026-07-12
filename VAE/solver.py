import torch
from torch.utils.data import DataLoader
import numpy as np
import anndata
import os
from scipy import sparse
from utils import balancer

class PerTrainer:
    """
    Trainer class for perturbation-aware Variational Autoencoder (VAE), providing utilities for:
        1. Training and validating a conditional VAE model.
        2. Performing perturbation prediction:
            - Within cell line (unseen perturbation prediction via similarity-based delta transfer)
            - Across cell lines (cross-cell-line generalization, maybe including unseen perturbations as well)
        3. Performing bias calculation grouped by perturbations and genes in training cell lines.
    """
    def __init__(self, model, train_set = None, valid_set = None, **kwargs):

        self.model = model
        self.learning_rate = kwargs.get("learning_rate", 0.001)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.model_to_use = kwargs.get("model_path", "./models")
        self.log = kwargs.get("log")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_set = train_set
        self.valid_set = valid_set

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        KL = 0
        RECON = 0
        for x, e in dataloader:
            x, e = x.to(self.device), e.to(self.device)
            self.optimizer.zero_grad()
            kl_loss, recon_loss, vae_loss = self.model(x, e)
            loss = vae_loss
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            KL += kl_loss.item()
            RECON += recon_loss.item()
        return total_loss / len(dataloader), KL / len(dataloader), RECON / len(dataloader)

    def valid_epoch(self, dataloader):
        self.model.eval()
        total_loss = 0
        for x, e in dataloader:
            x, e = x.to(self.device), e.to(self.device)
            self.optimizer.zero_grad()
            _, _, vae_loss = self.model(x, e)
            loss = vae_loss
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def train(self, n_epochs, batch_size, shuffle=True, initial_run=True, save=True):
        if initial_run:
            self.log.info("--------Training--------")

        best_loss = np.inf
        self.log.info("Training VAE...")
        self.train_loader = DataLoader(
            self.train_set,
            batch_size=batch_size,
            shuffle=shuffle
        )
        self.valid_loader = DataLoader(
            self.valid_set,
            batch_size=batch_size,
            shuffle=False
        )

        for it in range(n_epochs):
            self.model.train()
            vae_loss, kl_loss, recon_loss = self.train_epoch(self.train_loader)
            valid_loss = self.valid_epoch(self.valid_loader)

            if best_loss > valid_loss:
                best_loss = valid_loss
                os.makedirs(self.model_to_use, exist_ok=True)
                torch.save(self.model.state_dict(),  # save best model
                            os.path.join(self.model_to_use, "best_model.pt"))

            os.makedirs(self.model_to_use, exist_ok=True)
            if save and (it+1) % 10 == 0:
                torch.save(self.model.state_dict(), os.path.join(self.model_to_use, f"model_vae_epoch{it+1}.pt"))          
            self.log.info(f"Epoch {it+1}: KL loss: {kl_loss},\t Recon loss: {recon_loss},\n Train Loss: {vae_loss},\t Valid Loss: {valid_loss}, \n")

        self.log.info(f"Training finished!")

    @staticmethod
    def _to_dense(x):
        return x.toarray() if sparse.issparse(x) else np.asarray(x)

    @torch.no_grad()
    def predict_across_cell(self, adata_within, adata_ctrl, adata_to_predict, similarity_df, cell_line_key, target_col, control_name, embeddings, seed):
        self.model.eval()
        np.random.seed(seed)
        ctrl_x = adata_within[adata_within.obs[target_col] == control_name, :] # unperturbed cells in training cell lines
        stim_x = adata_within[adata_within.obs[target_col] != control_name, :] # perturbed cells in training cell lines
        train_perts = stim_x.obs[target_col].unique() # perturbations in training cell lines
        e_dim = embeddings[next(iter(embeddings))].shape[0]
        pred_adatas = []
        ctrl_adatas = []

        # 1. Calculate the average delta in latent space for each perturbation across the training cell lines
        self.log.info("Calculate delta_p for each perturbation across training cell lines...")
        delta_dict = {}
        for p in train_perts:
            stim_x_p = stim_x[stim_x.obs[target_col] == p, :]
            # Upsample to eliminate the cell type bias
            stim_x_p = balancer(stim_x_p, cell_line_key, target_col)
            emb_p = embeddings[p]

            delta_list = []
            for cl in stim_x_p.obs[cell_line_key].unique():
                stim_cl = stim_x_p[stim_x_p.obs[cell_line_key] == cl, :]
                ctrl_cl = ctrl_x[ctrl_x.obs[cell_line_key] == cl, :]
                if stim_cl.shape[0] == 0 or ctrl_cl.shape[0] == 0:
                    continue
                # Downsample to eliminate the population bias
                eq = min(stim_cl.shape[0], ctrl_cl.shape[0])
                stim_ind = np.random.choice(range(stim_cl.shape[0]), eq, replace=True)
                ctrl_ind = np.random.choice(range(ctrl_cl.shape[0]), eq, replace=True)
                stim_bal = stim_cl[stim_ind, :]
                ctrl_bal = ctrl_cl[ctrl_ind, :]
                # Encode and obtain latent average
                latent_sim_p = self.model._avg_vector(self._to_dense(stim_bal.X), np.repeat(emb_p[None, :], stim_bal.shape[0], axis=0))
                latent_sim_ctrl = self.model._avg_vector(self._to_dense(ctrl_bal.X), np.zeros((ctrl_bal.shape[0], e_dim)))
                delta_list.append(latent_sim_p - latent_sim_ctrl)

            if not delta_list:
                continue
            delta_dict[p] = np.mean(delta_list, axis=0)

        # 2. Perform similarity-based delta transfer for each perturbation in each target cell line
        target_cell_lines = adata_to_predict.obs[cell_line_key].astype(str).unique()
        self.log.info("Predicting every perturbation for each target cell line...")
        for cl in target_cell_lines:
            pred_cl = adata_to_predict[adata_to_predict.obs[cell_line_key].astype(str) == str(cl), :]
            ctrl_cl = adata_ctrl[adata_ctrl.obs[cell_line_key].astype(str) == str(cl), :]
            if pred_cl.shape[0] == 0 or ctrl_cl.shape[0] == 0:
                self.log.warning(
                    "Skip cell line %s in prediction because prediction/control cells are empty: pred=%d ctrl=%d",
                    cl, pred_cl.shape[0], ctrl_cl.shape[0]
                )
                continue

            all_perts = pred_cl.obs[target_col].unique()
            for pert in all_perts:
                # Similar downsampling
                pred_x_p = pred_cl[pred_cl.obs[target_col] == pert, :]
                eq = min(ctrl_cl.X.shape[0], pred_x_p.X.shape[0])
                cd_ind = np.random.choice(range(ctrl_cl.shape[0]), size=eq, replace=True)
                pred_ind = np.random.choice(range(pred_x_p.shape[0]), size=eq, replace=True)
                pred_x_p = pred_x_p[pred_ind, :]
                pred_x_ctrl = ctrl_cl[cd_ind, :]

                # Select most similar perturbations
                row = similarity_df[similarity_df["test_pert"] == pert].iloc[0]
                weight_cols = sorted(
                    [c for c in row.index if c.startswith("weight_")],
                    key=lambda x: int(x.split("_")[1])
                )
                nn_cols = sorted(
                    [c for c in row.index if c.startswith("nn_")],
                    key=lambda x: int(x.split("_")[1])
                )
                weights = row[weight_cols].astype(float).values
                nn_perts = row[nn_cols].values
                valid_pairs = [(p, w) for p, w in zip(nn_perts, weights) if p in delta_dict]
                if not valid_pairs:
                    continue
                nn_perts = np.array([p for p, _ in valid_pairs], dtype=object)
                weights = np.array([w for _, w in valid_pairs], dtype=float)
                wsum = weights.sum()
                if wsum == 0:
                    weights = np.repeat(1.0 / len(weights), len(weights))
                else:
                    weights = weights / wsum
                deltas = np.stack([delta_dict[p] for p in nn_perts])  # (k, latent_dim)
                weights = np.asarray(weights).reshape(-1)
                delta_weighted = (weights[:, None] * deltas).sum(axis=0)

                # Transfer the delta_p and predict in target cell line
                N = pred_x_ctrl.shape[0]
                pred_z_ctrl = self.model.to_latent(self._to_dense(pred_x_ctrl.X), np.zeros((N, e_dim)))
                pred_z_p = pred_z_ctrl + delta_weighted[None, :]
                emb_p = embeddings[pert]
                emb_t = torch.tensor(np.repeat(emb_p[None, :], N, axis=0), dtype=torch.float32, device=self.device)
                predicted_cell = self.model.reconstruct(pred_z_p, emb_t)

                pred_adata= anndata.AnnData(
                    predicted_cell,
                    obs={
                        target_col: [f"{pert}"] * len(predicted_cell),
                        cell_line_key: [str(cl)] * len(predicted_cell),
                    },
                    var=pred_x_p.var[["gene_name"]].copy()
                )
                pred_adatas.append(pred_adata)

            ctrl_export = anndata.AnnData(
                X=ctrl_cl.X,
                obs={
                    target_col: [control_name] * ctrl_cl.n_obs,
                    cell_line_key: [str(cl)] * ctrl_cl.n_obs,
                },
                var=ctrl_cl.var[["gene_name"]].copy()
            )
            ctrl_adatas.append(ctrl_export)
            
        self.log.info(f"All done for the cross_cell_line prediction!")
        pred_adatas.extend(ctrl_adatas)
        if len(pred_adatas) == 0:
            raise ValueError("No predictions generated in predict_across_cell.")
        adata = anndata.concat(
            pred_adatas,
            axis=0,
            join="inner",
            merge="same",
            index_unique=None,   
        )
        return delta_dict, adata
    
    @torch.no_grad()
    def predict_within_cell(self, adata_train, adata_test, similarity_df, target_col, control_name, embeddings, seed):
        self.model.eval()
        np.random.seed(seed)
        ctrl_x = adata_train[adata_train.obs[target_col] == control_name, :]
        stim_x = adata_train[adata_train.obs[target_col] != control_name, :]
        e_dim = embeddings[next(iter(embeddings))].shape[0]
        all_perts = [
            p for p in adata_test.obs[target_col].unique()
            if p != control_name
        ] 
        pred_adatas = []
        true_adatas = []

        # 1. Calculate the average delta in latent space for each perturbation in the training data
        self.log.info("Calculate delta_p for each perturbation in training dataset...")
        delta_dict = {}
        train_perts = stim_x.obs[target_col].unique()
        for p in train_perts:
            x_p = stim_x[stim_x.obs[target_col] == p, :]
            # Downsample to eliminate the population bias
            eq = min(x_p.shape[0], ctrl_x.shape[0])
            stim_ind = np.random.choice(range(x_p.shape[0]), eq, replace=True)
            ctrl_ind = np.random.choice(range(ctrl_x.shape[0]), eq, replace=True)
            stim_bal = x_p[stim_ind, :]
            ctrl_bal = ctrl_x[ctrl_ind, :]
            # Encode and obtain latent average
            emb_p = embeddings[p]
            latent_sim_p = self.model._avg_vector(self._to_dense(stim_bal.X), np.repeat(emb_p[None, :], stim_bal.shape[0], axis=0))
            latent_sim_ctrl = self.model._avg_vector(self._to_dense(ctrl_bal.X), np.zeros((ctrl_bal.shape[0], e_dim)))
            delta_dict[p] = latent_sim_p - latent_sim_ctrl

        # 2. Perform similarity-based delta transfer for each perturbation in the test data
        self.log.info("Predicting unseen perturbations within cell line...")
        for pert in all_perts:
            # Downsampling
            pred_x_p = adata_test[adata_test.obs[target_col] == pert, :]
            eq = min(ctrl_x.X.shape[0], pred_x_p.X.shape[0])
            cd_ind = np.random.choice(range(ctrl_x.shape[0]), size=eq, replace=True)
            pred_ind = np.random.choice(range(pred_x_p.shape[0]), size=eq, replace=True)
            pred_x_p = pred_x_p[pred_ind, :]
            pred_x_ctrl = ctrl_x[cd_ind, :]

            # Predict based on the similar perturbation
            row = similarity_df[similarity_df["test_pert"] == pert]
            row = row.iloc[0]
            weight_cols = sorted(
                [c for c in row.index if c.startswith("weight_")],
                key=lambda x: int(x.split("_")[1])
            )
            nn_cols = sorted(
                [c for c in row.index if c.startswith("nn_")],
                key=lambda x: int(x.split("_")[1])
            )
            weights = row[weight_cols].astype(float).values
            nn_perts = row[nn_cols].values
            deltas = np.stack([delta_dict[p] for p in nn_perts])  # (k, latent_dim)
            weights = np.asarray(weights, dtype=float).reshape(-1)
            wsum = weights.sum()
            if wsum == 0:
                weights = np.repeat(1.0 / len(weights), len(weights))
            else:
                weights = weights / wsum
            delta_weighted = (weights[:, None] * deltas).sum(axis=0)
            N = pred_x_ctrl.shape[0]
            pred_z_ctrl = self.model.to_latent(self._to_dense(pred_x_ctrl.X), np.zeros((N, e_dim)) )  # (N, latent_dim)
            pred_z_p = pred_z_ctrl + delta_weighted[None, :]

            emb_p = embeddings[pert]
            emb_t = torch.tensor(np.repeat(emb_p[None, :], N, axis=0), dtype=torch.float32, device=self.device)
            predicted_cell = self.model.reconstruct(pred_z_p, emb_t)

            pred_adata = anndata.AnnData(
                predicted_cell,
                obs={target_col: [f"{pert}"] * len(predicted_cell)},
                var=pred_x_p.var[["gene_name"]].copy()
            )

            true_adata = anndata.AnnData(
                pred_x_p.X,
                obs={target_col: [f"{pert}"] * len(predicted_cell)},
                var=pred_x_p.var[["gene_name"]].copy()
            )

            true_adatas.append(true_adata)
            pred_adatas.append(pred_adata)

        self.log.info(f"All done for the within_cell_line prediction!")
        
        adata_ctrl = anndata.AnnData(
            X=ctrl_x.X,
            obs={target_col: [control_name] * ctrl_x.n_obs},
            var=ctrl_x.var[["gene_name"]].copy()
        )
        
        true_adatas.append(adata_ctrl)
        pred_adatas.append(adata_ctrl)
        pred_adata = anndata.concat(
            pred_adatas,
            axis=0,
            join="inner",
            merge="same",
            index_unique=None,   
        )
        true_adata = anndata.concat(
            true_adatas,
            axis=0,
            join="inner",
            merge="same",
            index_unique=None,   
        )
        return pred_adata, true_adata
    
    @torch.no_grad()
    def bias(self, adata_within, cell_line_key, target_col, control_name, embeddings, non_zero=False):
        self.model.eval()
        stim_x = adata_within[adata_within.obs[target_col] != control_name, :] # perturbed cells in training cell lines
        train_perts = stim_x.obs[target_col].unique() # perturbations in training cell lines
        self.log.info(f"{len(train_perts)} Perturbations in training cell lines!")
        e_dim = embeddings[next(iter(embeddings))].shape[0]
        pred_adatas = []

        self.log.info("Calculate reconstruction bias for each perturbation across training cell lines...")
        bias_dict = {}
        for p in train_perts:
            stim_x_p = stim_x[stim_x.obs[target_col] == p, :]
            # Upsample to eliminate the cell type bias
            stim_x_p = balancer(stim_x_p, cell_line_key, target_col)
            N = stim_x_p.shape[0]
            emb_p = embeddings[p]
            pred_z_p = self.model.to_latent(stim_x_p.X, np.repeat(emb_p[None, :], stim_x_p.shape[0], axis=0))
            emb_t = torch.tensor(np.repeat(emb_p[None, :], N, axis=0), dtype=torch.float32, device=self.device)
            pred_x_p = self.model.reconstruct(pred_z_p, emb_t)
            if non_zero:
                mask = (stim_x_p.X != 0)
                bias = stim_x_p.X - pred_x_p
                bias_masked = np.where(mask, bias, np.nan)
                bias = np.nanmean(bias_masked, axis=0)
                bias_dict[p] = np.nan_to_num(bias, nan=0.0)
            else:
                bias = stim_x_p.X - pred_x_p
                bias_dict[p] = np.mean(bias, axis=0)
        self.log.info("Bias calculation completed!")
        first_key = next(iter(bias_dict))
        self.log.info(f"Example key: {first_key}, bias shape: {bias_dict[first_key].shape}")
        self.log.info(f"Total perturbations: {len(bias_dict)}")
        
        return bias_dict
