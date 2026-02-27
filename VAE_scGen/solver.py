import torch
from torch.utils.data import DataLoader
import numpy as np
import anndata
import os
from utils import balancer, log_setting
log = log_setting(log_dir="./logs")

class PerTrainer:
    """
    Trainer class for perturbation-aware Variational Autoencoder (VAE), providing utilities for:
        1. Training and validating a conditional VAE model.
        2. Performing perturbation prediction:
            - Across cell lines (cross-cell-line generalization)
            - Within cell line (unseen perturbation prediction via similarity-based delta transfer)
    """
    def __init__(self, model, train_set = None, valid_set = None, **kwargs):
        self.model = model
        self.learning_rate = kwargs.get("learning_rate", 0.001)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.model_to_use = kwargs.get("model_path", "./models")
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
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def train(self, n_epochs=25, batch_size=32, shuffle=True, initial_run=True, save=True):
        if initial_run:
            log.info("----Training----")

        best_loss = np.inf
        log.info("Training VAE...")
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
            if save and (it+1) % 5 == 0:
                torch.save(self.model.state_dict(), os.path.join(self.model_to_use, f"model_vae_epoch{it+1}.pt"))          
            log.info(f"Epoch {it+1}: KL loss: {kl_loss},\t Recon loss: {recon_loss},\n Train Loss: {vae_loss},\t Valid Loss: {valid_loss}, \n")

        log.info(f"Training finished!")

    @torch.no_grad()
    def predict_across_cell(self, adata_within, adata_ctrl, adata_to_predict, cell_line_key, target_col, control_name, embeddings, seed, biased=False):
        self.model.eval()
        np.random.seed(seed)
        ctrl_x = adata_within[adata_within.obs[target_col] == control_name, :]
        stim_x = adata_within[adata_within.obs[target_col] != control_name, :]
        all_perts = adata_to_predict.obs[target_col].unique()
        pred_adatas = []

        log.info(f"Predicting every perturbation for held out cell line...")
        for pert in all_perts:
            # Downsample the control and perturbed data of held out cell line to have the same number of samples
            pred_x_p = adata_to_predict[adata_to_predict.obs[target_col] == pert, :]
            eq = min(adata_ctrl.X.shape[0], pred_x_p.X.shape[0])
            cd_ind = np.random.choice(range(adata_ctrl.shape[0]), size=eq, replace=True)
            pred_ind = np.random.choice(range(pred_x_p.shape[0]), size=eq, replace=True)
            pred_x_p = pred_x_p[pred_ind, :]
            pred_x_ctrl = adata_ctrl[cd_ind, :]
            
            stim_x_p = stim_x[stim_x.obs[target_col] == pert, :]
            # Upsample to eliminate the cell type bias and then downsample to eliminate the pop[ulation bias]
            if not biased:
                stim_x_p = balancer(stim_x_p, cell_line_key, target_col)
                eq = min(ctrl_x.X.shape[0], stim_x_p.X.shape[0])
                cd_ind = np.random.choice(range(ctrl_x.shape[0]), size=eq, replace=True)
                stim_ind = np.random.choice(range(stim_x_p.shape[0]), size=eq, replace=True)
                stim_x_p = stim_x_p[stim_ind, :]
                stim_x_ctrl = ctrl_x[cd_ind, :]

            # Predict the held out cell line in latent space 
            predicted_cell = self.model.inference_p(pred_x_ctrl, stim_x_p, stim_x_ctrl, embeddings[pert])

            pred_adata= anndata.AnnData(
                predicted_cell,
                obs={target_col: [f"{pert}"] * len(predicted_cell)},
                var=pred_x_p.var[["gene_name"]].copy()
            )
            pred_adatas.append(pred_adata)
            
        log.info(f"All done for the cross_cell_line prediction!")
        
        adata_ctrl = anndata.AnnData(
            X=adata_ctrl.X,
            obs={target_col: [control_name] * adata_ctrl.n_obs},
            var=adata_ctrl.var[["gene_name"]].copy()
        )
        
        pred_adatas.append(adata_ctrl)
        adata = anndata.concat(
            pred_adatas,
            axis=0,
            join="inner",
            merge="same",
            index_unique=None,   
        )
        return adata
    
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

        # 1. Calculate the average delta in latent space for each perturbation in the training data
        delta_dict = {}
        latent_sim_ctrl = self.model._avg_vector(ctrl_x.X, np.zeros((ctrl_x.shape[0], e_dim)))
        train_perts = stim_x.obs[target_col].unique()
        for p in train_perts:
            x_p = stim_x[stim_x.obs[target_col] == p, :]
            emb_p = embeddings[p]
            latent_sim_p = self.model._avg_vector(x_p.X, np.repeat(emb_p[None, :], x_p.shape[0], axis=0))
            delta_dict[p] = latent_sim_p - latent_sim_ctrl

        # 2. Perform similarity-based delta transfer for each perturbation in the test data
        log.info("Predicting unseen perturbations within cell line...")
        for pert in all_perts:
            # Downsample the control and perturbed data of held out cell line to have the same number of samples
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
            weights = np.asarray(weights).reshape(-1)
            delta_weighted = (weights[:, None] * deltas).sum(axis=0)
            N = pred_x_ctrl.shape[0]
            pred_z_ctrl = self.model.to_latent(pred_x_ctrl.X, np.zeros((N, e_dim)) )  # (N, latent_dim)
            pred_z_p = pred_z_ctrl + delta_weighted[None, :]

            emb_p = embeddings[pert]
            emb_t = torch.tensor(np.repeat(emb_p[None, :], N, axis=0), dtype=torch.float32, device=self.device)
            predicted_cell = self.model.reconstruct(pred_z_p, emb_t)

            pred_adata= anndata.AnnData(
                predicted_cell,
                obs={target_col: [f"{pert}"] * len(predicted_cell)},
                var=pred_x_p.var[["gene_name"]].copy()
            )
            pred_adatas.append(pred_adata)

        log.info(f"All done for the within_cell_line prediction!")
        
        adata_ctrl = anndata.AnnData(
            X=ctrl_x.X,
            obs={target_col: [control_name] * ctrl_x.n_obs},
            var=ctrl_x.var[["gene_name"]].copy()
        )
        
        pred_adatas.append(adata_ctrl)
        adata = anndata.concat(
            pred_adatas,
            axis=0,
            join="inner",
            merge="same",
            index_unique=None,   
        )
        return adata
    