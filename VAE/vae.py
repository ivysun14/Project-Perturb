import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse

class VAEArith(nn.Module):
    """
    Conditional Variational Autoencoder with Latent Arithmetic Transfer.
    This model implements a conditional VAE designed for multi-perturbation modeling in single-cell gene expression.

    The model learns a conditional latent representation:
        z ~ q(z | x, e)
    where:
        x : gene expression vector
        e : perturbation embedding
    Decoder reconstructs:
        x_hat = f(z, e)
    """

    def __init__(self, x_dimension, z_dimension, emb_dimension, **kwargs):
        super().__init__()
        self.x_dim = x_dimension
        self.z_dim = z_dimension
        self.e_dim = emb_dimension
        self.dropout_rate = kwargs.get("dropout_rate", 0.1)
        self.model_to_use = kwargs.get("model_path", "./models")
        self.alpha = kwargs.get("alpha", 0.1)
        restore = kwargs.get("restore", False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Embedding encoder
        self.ce_dim = self.z_dim * 4
        self.cond_encoder = nn.Sequential(
            nn.Linear(self.e_dim, self.ce_dim),
            nn.GELU(),
        )

        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Linear(x_dimension, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(1024, 512),
        )
        
        self.encoder2 = nn.Sequential(
            nn.Linear(512+self.ce_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(512, 128),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, self.z_dim),
        )

        self.fc_mu = nn.Linear(self.z_dim, self.z_dim)
        self.fc_logvar = nn.Linear(self.z_dim, self.z_dim)

        # Decoder
        self.decoder1 = nn.Sequential(
            nn.Linear(self.z_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, 512),
        )

        self.decoder2 = nn.Sequential(
            nn.Linear(512+self.ce_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(512, 1024),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(1024, self.x_dim)
        )

        self.to(self.device)
        if restore:
            self.restore_model()

    def _encoder(self, x_in, cond_in):
        h = self.encoder1(x_in)
        h = torch.cat([h, cond_in], dim=1)
        h = self.encoder2(h)
        mean = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        return mean, log_var

    def _decoder(self, z_in, cond_in):
        h = self.decoder1(z_in)
        h = torch.cat([h, cond_in], dim=1)
        h = self.decoder2(h)
        return h
    
    def _cond_encoder(self, cond, scope="Emb_encoder"):
        h = self.cond_encoder(cond)
        return h
    
    def _sample_z(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z
    
    def _loss_function(self, x, x_hat, mu, log_var):
        kl_loss = torch.mean(0.5 * torch.sum(torch.exp(log_var) + mu * mu - 1.0 - log_var, dim=1))
        recon_loss = torch.mean(0.5 * torch.sum((x - x_hat) ** 2, dim=1))
        vae_loss = recon_loss + self.alpha * kl_loss
        return kl_loss, recon_loss, vae_loss
    
    def _avg_vector(self, x_np, cond_np):
        """
        Computes the average of points which computed from mapping `data` to encoder part of VAE.
        """
        latent = self.to_latent(x_np, cond_np)
        latent_avg = np.average(latent, axis=0)
        return latent_avg

    def restore_model(self):
        path = os.path.join(self.model_to_use, "best_model.pt")
        ckpt = torch.load(path, map_location=self.device)
        self.load_state_dict(ckpt)
        self.to(self.device)
    
    def forward(self, x, cond, **kwargs):
        cond = self._cond_encoder(cond)
        mu, log_var = self._encoder(x, cond)
        z = self._sample_z(mu, log_var)
        x_hat = self._decoder(z, cond)
        return self._loss_function(x, x_hat, mu, log_var)
    
    @torch.no_grad()
    def to_latent(self, x_np, cond_np, batch_size=2048):
        """
        Map origianl gene expression data into latent space.
        """
        n = x_np.shape[0]
        zs = []
        for i in range(0, n, batch_size):
            xb = torch.as_tensor(x_np[i:i+batch_size], dtype=torch.float32, device=self.device)
            cb = torch.as_tensor(cond_np[i:i+batch_size], dtype=torch.float32, device=self.device)
            cb = self._cond_encoder(cb)
            mu, log_var = self._encoder(xb, cb)
            z = mu
            zs.append(z.detach().cpu())

        return torch.cat(zs, dim=0).numpy()

    @torch.no_grad()
    def reconstruct(self, x_or_z_np, cond):
        """
        Map back from the latent space encoding via the decoder.
        """
        self.eval()
        z = torch.tensor(x_or_z_np, dtype=torch.float32, device=self.device)
        cond = self._cond_encoder(cond)
        x_hat = self._decoder(z, cond)
        return x_hat.detach().cpu().numpy()    
    
    def inference_p(self, pred_x_ctrl, stim_x_p, stim_x_ctrl, emb_p):
        """
        Perform perturbation effect inference via latent delta transfer.

        Step 1: Compute latent mean of control samples:
            z_ctrl = E[z | control]
        Step 2: Compute latent mean of perturbed samples:
            z_p = E[z | perturbation p]
        Step 3: Compute perturbation effect in latent space:
            delta_p = z_p - z_ctrl
        Step 4: Transfer perturbation effect to target control samples:
            z_pred = z_target_ctrl + delta_p
        Step 5: Decode:
            x_pred = Decoder(z_pred, e_p)
        """
        self.eval()
        def _maybe_dense(X):
                return X.toarray() if sparse.issparse(X) else np.asarray(X)
        N = pred_x_ctrl.shape[0]
        N_stim = stim_x_ctrl.shape[0]
        pred_z_ctrl = self.to_latent(_maybe_dense(pred_x_ctrl.X), np.zeros((N, self.e_dim)))
        emb_t = torch.tensor(np.repeat(emb_p[None, :], N, axis=0), dtype=torch.float32, device=self.device)

        latent_sim_ctrl = self._avg_vector(_maybe_dense(stim_x_ctrl.X), np.zeros((N_stim, self.e_dim)))

        latent_sim_p = self._avg_vector(_maybe_dense(stim_x_p.X), np.repeat(emb_p[None, :], N_stim, axis=0))

        delta_p = latent_sim_p - latent_sim_ctrl
        
        pred_z_p = pred_z_ctrl + delta_p

        predicted_cell = self.reconstruct(pred_z_p, emb_t)

        return predicted_cell

