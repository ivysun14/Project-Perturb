from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence as kl

from ._base_modules import VariationalEncoder, CountDecoder
from ._dists import NegativeBinomial, ZeroInflatedNegativeBinomial

from torchmetrics.functional import pairwise_euclidean_distance
try:
    from torch_scatter import scatter_mean
except Exception:
    def scatter_mean(src, index):
        """Fallback when torch_scatter is unavailable."""
        if src.numel() == 0:
            return src
        unique_ids = torch.unique(index, sorted=True)
        means = []
        for uid in unique_ids:
            mask = index == uid
            means.append(src[mask].mean(dim=0, keepdim=False))
        return torch.stack(means, dim=0)


def knn_purity(data, labels, n_neighbors=15):
    """Computes KNN Purity for ``data`` given the labels.
    Parameters
    ----------
    data:
        torch tensor of data (n_samples, n_features)
    labels
        torch tensor of labels (n_samples,)
    n_neighbors: int
        Number of nearest neighbors.
    Returns
    -------
    score: float
        KNN purity score. A float between 0 and 1.
    """
    distances = pairwise_euclidean_distance(data)
    # sort each row in distances to get nearest neighbors

    _, indices = torch.topk(
        distances, k=n_neighbors + 1, dim=1, largest=False, sorted=True
    )
    indices = indices[:, 1:]  # remove self
    # neighbors_labels = np.vectorize(lambda i: labels[i])(indices)
    neighbors_labels = labels[indices]  # (n_samples, n_neighbors)

    # pre cell purity scores
    scores = (
        ((neighbors_labels - labels.reshape(-1, 1)) == 0).float().mean(axis=1)
    )  # (n_samples,)
    res = scatter_mean(scores, labels).mean()  # per category purity

    return res


class scVIModule(nn.Module):
    """
    scVI module using NegativeBinomial/Zero-Inflated NegativeBinomial Likelihood objectives

    Parameters
    ----------
        n_genes: int
        n_treatments: int
        covars_encoder: dict
            Dictionary of covariates with keys as each covariate name and values as
                number of unique values of the corresponding covariate
        n_latent: int
            Latent Dimension
        loss_ae: str
            Count likelihood, either "nb" or "zinb".
        doser_type: str
            # Type of doser network, either `mlp` or `linear`.
        autoencoder_width: int
        autoencoder_depth: int
        use_batch_norm: bool
        use_layer_norm: bool
        variational: bool
    """

    def __init__(
        self,
        n_genes: int,
        n_perts: int,
        n_cell_types: int,
        n_batches: int = 1,
        pert_embeddings: Optional[np.ndarray] = None,
        use_cell_type_embeddings: bool = True,
        use_batch_embeddings: bool = True,
        input_is_log1p: bool = True,
        n_latent: int = 128,
        n_pert_latent: int = 64,
        n_cell_type_latent: int = 32,
        n_batch_latent: int = 16,
        recon_loss: str = "nb",
        n_hidden_encoder: int = 256,
        n_layers_encoder: int = 3,
        n_hidden_decoder: int = 256,
        n_layers_decoder: int = 3,
        use_batch_norm: str = "both",
        use_layer_norm: str = "none",
        dropout_rate_encoder: float = 0.0,
        dropout_rate_decoder: float = 0.0,
        seed: int = 0,
    ):
        super().__init__()

        torch.manual_seed(seed)
        np.random.seed(seed)

        recon_loss = recon_loss.lower()

        assert recon_loss in ["nb", "zinb"]

        self.n_genes = n_genes
        self.n_latent = n_latent
        self.n_cell_type_latent = n_cell_type_latent
        self.n_batch_latent = n_batch_latent
        self.n_perts = n_perts
        self.recon_loss = recon_loss
        self.use_cell_type_embeddings = use_cell_type_embeddings and n_cell_types > 1
        self.use_batch_embeddings = use_batch_embeddings and n_batches > 1
        self.input_is_log1p = input_is_log1p

        effective_pert_latent = (
            int(pert_embeddings.shape[1]) if pert_embeddings is not None else n_pert_latent
        )
        self.n_pert_latent = effective_pert_latent

        self.encoder = VariationalEncoder(
            n_genes,
            n_latent,
            var_activation=nn.Softplus(),
            n_hidden=n_hidden_encoder,
            n_layers=n_layers_encoder,
            use_batch_norm=use_batch_norm in ["both", "encoder"],
            use_layer_norm=use_layer_norm in ["both", "encoder"],
            dropout_rate=dropout_rate_encoder,
            activation_fn=nn.ReLU,
            return_dist=True,
        )

        decoder_cell_type_latent = n_cell_type_latent if self.use_cell_type_embeddings else 0
        decoder_batch_latent = n_batch_latent if self.use_batch_embeddings else 0
        n_input_decoder = (
            n_latent
            + decoder_cell_type_latent
            + decoder_batch_latent
            + self.n_pert_latent
        )

        # Decoder components
        self.px_r = torch.nn.Parameter(torch.randn(self.n_genes))
        self.decoder = CountDecoder(
            n_input=n_input_decoder,
            n_output=n_genes,
            n_layers=n_layers_decoder,
            n_hidden=n_hidden_decoder,
            use_norm=(
                "batch"
                if use_batch_norm in ["both", "decoder"]
                else "layer" if use_layer_norm in ["both", "decoder"] else "none"
            ),
        )

        # Embeddings
        if pert_embeddings is not None:
            self.pert_embeddings = nn.Embedding.from_pretrained(
                torch.FloatTensor(pert_embeddings), freeze=True
            )
        else:
            self.pert_embeddings = nn.Embedding(n_perts, self.n_pert_latent)

        if self.use_batch_embeddings:
            self.batch_embeddings = nn.Embedding(n_batches, n_batch_latent)
        else:
            self.batch_embeddings = None

        if self.use_cell_type_embeddings:
            self.cell_type_embeddings = nn.Embedding(n_cell_types, n_cell_type_latent)
        else:
            self.cell_type_embeddings = None

    def forward(self, x_basal, perts, cell_types, batch_ids, n_samples: int = 1):
        enc_outputs = self.forward_encoder(
            x=x_basal,
            perts=perts,
            cell_types=cell_types,
            batch_ids=batch_ids,
            n_samples=n_samples,
        )

        dec_outputs = self.forward_decoder(
            z_basal=enc_outputs["z_basal"],
            z=enc_outputs["z"],
            library=enc_outputs["library"],
        )

        return enc_outputs, dec_outputs

    def forward_encoder(
        self,
        x,
        perts,
        cell_types,
        batch_ids: Optional[torch.Tensor] = None,
        n_samples: int = 1,
    ):
        batch_size = x.shape[0]

        if self.input_is_log1p:
            x_ = x
            pseudo_counts = torch.exp(x) - 1
            library = torch.log(pseudo_counts.sum(1).clamp_min(1e-8)).unsqueeze(1)
        else:
            # log the input to the variational distribution for numerical stability
            x_ = torch.log(1 + x)
            library = torch.log(x.sum(1).clamp_min(1e-8)).unsqueeze(1)

        qz, z_basal = self.encoder(x_)

        if n_samples > 1:
            sampled_z = qz.sample((n_samples,))
            z_basal = self.encoder.z_transformation(sampled_z)
            library = library.unsqueeze(0).expand(
                (n_samples, library.size(0), library.size(1))
            )

        z_cell_type = []
        if self.cell_type_embeddings is not None:
            z_cell_type.append(self.cell_type_embeddings(cell_types.long()))

        z_batch = []
        if self.batch_embeddings is not None:
            z_batch.append(self.batch_embeddings(batch_ids.long()))

        z_pert = self.pert_embeddings(perts.long())

        z = torch.cat([z_basal] + z_cell_type + z_batch + [z_pert], dim=-1)
        z_corrected = torch.cat([z_basal] + z_cell_type + [z_pert], dim=-1)

        return dict(
            z=z,
            z_basal=z_basal,
            z_corrected=z_corrected,
            library=library,
            qz=qz,
        )

    def forward_decoder(
        self,
        z_basal,
        z,
        library,
    ):
        if self.recon_loss == "nb":
            px_scale, _, px_rate, px_dropout = self.decoder("gene", z, library)
            px_r = torch.exp(self.px_r)

            px = NegativeBinomial(mu=px_rate, theta=px_r, scale=px_scale)

        elif self.recon_loss == "zinb":
            px_scale, _, px_rate, px_dropout = self.decoder("gene", z, library)
            px_r = torch.exp(self.px_r)

            px = ZeroInflatedNegativeBinomial(
                mu=px_rate,
                theta=px_r,
                zi_logits=px_dropout,
                scale=px_scale,
            )

        else:
            raise ValueError(f"Unsupported count likelihood: {self.recon_loss}")

        pz = Normal(torch.zeros_like(z_basal), torch.ones_like(z_basal))
        return dict(px=px, pz=pz)

    def loss(self, x_pert, encoder_outputs, decoder_outputs):
        """Computes the reconstruction loss (AE) or the ELBO (VAE)"""
        px = decoder_outputs["px"]
        recon_loss = -px.log_prob(x_pert).sum(dim=-1).mean()

        qz = encoder_outputs["qz"]
        pz = decoder_outputs["pz"]

        kl_divergence_z = kl(qz, pz).sum(dim=1)
        kl_loss = kl_divergence_z.mean()

        return recon_loss, kl_loss

    def sample(self, x_basal, perts, cell_types, batch_ids, n_samples=1):
        """Computes knockout gene expression

        Parameters
        ----------
        tensors : dict
            dictionary of input tensors

        """
        _, decoder_outputs = self.forward(
            x_basal=x_basal,
            perts=perts,
            cell_types=cell_types,
            batch_ids=batch_ids,
            n_samples=n_samples,
        )

        px = decoder_outputs["px"]

        return px.sample()
