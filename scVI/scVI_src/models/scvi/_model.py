import logging
import os
import pickle
from typing import Dict, Iterable, Optional, Tuple
import anndata as ad
import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import StepLR

from ..base import PerturbationModel
from ._module import scVIModule
from ._utils import build_pert_embedding_matrix

logger = logging.getLogger(__name__)


class SCVIPerturbationModel(PerturbationModel):
    """
    Implementation of the scVI model. The outputs are always in
    gene expression space.

    Args:
        input_dim: Dimension of input embeddings (either number of genes or latent dim from obsm key)
        hidden_dim: Dimension of hidden layers
        output_dim: Number of genes to predict
        pert_dim: Dimension of perturbation inputs (usually one-hot size)
        decode_intermediate_dim: Optional intermediate dimension for decoder
        n_encoder_layers: Number of layers in encoder (default: 2)
        n_decoder_layers: Number of layers in encoder (default: 2)
        dropout: Dropout rate (default: 0.1)
        learning_rate: Learning rate for optimizer (default: 1e-3)
        loss_fn: Loss function (default: 'nn.MSELoss()')
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        pert_dim: int,
        n_cell_types: int,
        n_perts: int,
        n_batches: int,
        output_space: str = "gene",
        decoder=None,
        lr=5e-4,
        wd=1e-6,
        n_steps_kl_warmup: int = None,
        n_epochs_kl_warmup: int = None,
        step_size_lr: int = 45,
        do_clip_grad: bool = False,
        gradient_clip_value: float = 3.0,
        check_val_every_n_epoch: int = 5,
        pert_names: Optional[Iterable[str]] = None,
        pert_embeddings: Optional[np.ndarray] = None,
        pert_embedding_paths: Optional[Iterable[str]] = None,
        strict_pert_embeddings: bool = False,
        use_cell_type_embeddings: bool = False,
        use_batch_embeddings: bool = True,
        **kwargs,
    ):
        # Register with parent constructor
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            pert_dim=pert_dim,
            output_space=output_space,
            decoder=decoder,
            **kwargs,
        )

        # Set class specific parameters before registering with parent constructor
        self.n_cell_types = n_cell_types
        self.n_perts = n_perts
        self.n_batches = n_batches

        self.n_layers_encoder = kwargs.get("n_layers_encoder", 2)
        self.n_layers_decoder = kwargs.get("n_layers_decoder", 2)
        self.n_hidden_encoder = kwargs.get("n_hidden_encoder", 256)
        self.n_hidden_decoder = kwargs.get("n_hidden_decoder", 256)
        self.n_latent = kwargs.get("n_latent", 64)
        self.recon_loss = kwargs.get("recon_loss", "nb")

        self.use_batch_norm = kwargs.get("use_batch_norm", "both")
        self.use_layer_norm = kwargs.get("use_layer_norm", "none")

        self.pert_names = [str(name) for name in pert_names] if pert_names is not None else None
        self.filtered_pert_embeddings = {}
        self.missing_pert_embeddings = tuple()
        self.pert_embeddings = None  # will be set before _build_networks if provided
        self.use_cell_type_embeddings = use_cell_type_embeddings
        self.use_batch_embeddings = use_batch_embeddings

        self.dropout_rate_encoder = kwargs.get("dropout_rate_encoder", 0.0)
        self.dropout_rate_decoder = kwargs.get("dropout_rate_decoder", 0.0)
        self.seed = kwargs.get("seed", 0)

        # training params
        self.lr = lr
        self.wd = wd
        self.n_steps_kl_warmup = n_steps_kl_warmup
        self.n_epochs_kl_warmup = n_epochs_kl_warmup

        self.step_size_lr = step_size_lr
        self.do_clip_grad = do_clip_grad
        self.gradient_clip_value = gradient_clip_value
        self.check_val_every_n_epoch = check_val_every_n_epoch

        self.kwargs = kwargs
        self.training_logger = kwargs.get("log")
        self.mode = kwargs.get("mode", "within")
        self.cell_line = kwargs.get("cell_line", kwargs.get("name", "scvi"))
        mode_suffix = "cross" if self.mode == "cross" else "within"
        self.model_path = kwargs.get("model_path") or os.path.join(
            "./results/models", f"{self.cell_line}_{mode_suffix}"
        )
        self.prediction_output_dir = kwargs.get("prediction_output_dir") or "./results/predictions"
        self.save_epoch_checkpoints = kwargs.get("save_epoch_checkpoints", True)
        self.save_predictions = kwargs.get("save_predictions", True)
        self.save_bias = kwargs.get("save_bias", False)
        self.prediction_use_sampling = kwargs.get("prediction_use_sampling", False)
        self.bias_non_zero = kwargs.get("bias_non_zero", False)
        self.best_val_loss = np.inf
        self._predict_outputs = []
        self.input_is_log1p = kwargs.get("input_is_log1p", True)
        self.dataset_dir = kwargs.get("dataset_dir")
        self.dataset_name = kwargs.get("dataset_name")
        self.pert_col = kwargs.get("pert_col", "gene")

        if pert_embeddings is not None:
            self.pert_embeddings = np.asarray(pert_embeddings, dtype=np.float32)
        elif pert_embedding_paths is not None:
            if self.pert_names is None:
                raise ValueError(
                    "pert_names must be provided when using pretrained perturbation embeddings."
                )
            (
                self.pert_embeddings,
                self.filtered_pert_embeddings,
                self.missing_pert_embeddings,
            ) = build_pert_embedding_matrix(
                pert_names=self.pert_names,
                embedding_paths=pert_embedding_paths,
                control_pert=self.control_pert,
                strict=strict_pert_embeddings,
            )

        assert self.output_space in [
            "gene",
            "all",
        ], "scVI model only supports gene-level or all-level output"

        # Build model components
        self._build_networks()

    def _log_message(self, message: str):
        logger.info(message)
        if self.training_logger is not None and self.training_logger is not logger:
            try:
                self.training_logger.info(message)
            except Exception:
                pass

    def _save_state_dict(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def _restore_from_log_if_needed(self, X: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if X is None:
            return None
        if self.input_is_log1p:
            return torch.exp(X) - 1
        return X

    def _prediction_to_output_space(self, X: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if X is None:
            return None
        if self.input_is_log1p:
            return torch.log1p(X)
        return X

    def _to_numpy(self, value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _flatten_outputs(self, outputs):
        flat = []
        for item in outputs:
            if isinstance(item, list):
                flat.extend(self._flatten_outputs(item))
            elif item is not None:
                flat.append(item)
        return flat

    def _concat_output_field(self, outputs, key: str):
        values = [out.get(key) for out in outputs if out.get(key) is not None]
        if not values:
            return None
        first = values[0]
        if isinstance(first, torch.Tensor):
            return torch.cat([v.detach().cpu() for v in values], dim=0).numpy()
        merged = []
        for value in values:
            if isinstance(value, (list, tuple)):
                merged.extend(list(value))
            else:
                merged.extend(list(np.asarray(value)))
        return np.asarray(merged, dtype=object)

    def _build_obs_frame(self, outputs, n_rows: int):
        obs = {}
        for key in ("pert_name", "cell_type", "batch_name", "ctrl_cell_barcode"):
            values = self._concat_output_field(outputs, key)
            if values is not None:
                obs[key] = values
        if not obs:
            obs["pert_name"] = np.array(["unknown"] * n_rows, dtype=object)
        return pd.DataFrame(obs)

    def _build_var_frame(self, n_genes: int):
        gene_names = self.gene_names if self.gene_names is not None else [f"gene_{i}" for i in range(n_genes)]
        return pd.DataFrame({"gene_name": list(gene_names)})

    def _build_prediction_anndata(self, outputs, key: str):
        matrix = self._concat_output_field(outputs, key)
        if matrix is None:
            return None
        obs = self._build_obs_frame(outputs, matrix.shape[0])
        var = self._build_var_frame(matrix.shape[1])
        return ad.AnnData(matrix, obs=obs, var=var)

    def _load_control_adata_for_export(self):
        if not self.dataset_dir or not self.dataset_name:
            return None
        dataset_path = os.path.join(self.dataset_dir, f"{self.dataset_name}.h5ad")
        if not os.path.exists(dataset_path):
            self._log_message(f"Control export skipped: dataset file not found at {dataset_path}")
            return None

        adata = ad.read_h5ad(dataset_path, backed="r")
        try:
            if self.pert_col not in adata.obs.columns:
                self._log_message(
                    f"Control export skipped: obs[{self.pert_col!r}] not found in {dataset_path}"
                )
                return None
            control_mask = adata.obs[self.pert_col].astype(str).to_numpy() == str(self.control_pert)
            control_view = adata[control_mask]
            control_adata = control_view.to_memory()
        finally:
            file_manager = getattr(adata, "file", None)
            if file_manager is not None:
                file_manager.close()

        control_obs = pd.DataFrame(
            {self.pert_col: [str(self.control_pert)] * control_adata.n_obs},
            index=control_adata.obs_names,
        )
        control_var = self._build_var_frame(control_adata.n_vars)
        return ad.AnnData(X=control_adata.X, obs=control_obs, var=control_var)

    def _append_controls_for_within(self, adata_obj):
        if adata_obj is None:
            return None
        control_adata = self._load_control_adata_for_export()
        if control_adata is None:
            return adata_obj
        converted_obs = pd.DataFrame(
            {self.pert_col: adata_obj.obs["pert_name"].astype(str).tolist()},
            index=adata_obj.obs.index,
        )
        converted = ad.AnnData(X=adata_obj.X, obs=converted_obs, var=adata_obj.var.copy())
        return ad.concat(
            [converted, control_adata],
            axis=0,
            join="inner",
            merge="same",
            index_unique=None,
        )

    def _save_prediction_artifacts(self, outputs):
        if not self.save_predictions:
            return
        os.makedirs(self.prediction_output_dir, exist_ok=True)
        if self.mode == "cross":
            pred_adata = self._build_prediction_anndata(outputs, "preds")
            if pred_adata is None:
                return
            out_path = os.path.join(self.prediction_output_dir, f"{self.cell_line}_cross.h5ad")
            pred_adata.write_h5ad(out_path)
        else:
            pred_adata = self._build_prediction_anndata(outputs, "preds")
            true_adata = self._build_prediction_anndata(outputs, "X")
            pred_adata = self._append_controls_for_within(pred_adata)
            true_adata = self._append_controls_for_within(true_adata)
            if pred_adata is not None:
                pred_adata.write_h5ad(
                    os.path.join(self.prediction_output_dir, f"{self.cell_line}_within.h5ad")
                )
            if true_adata is not None:
                true_adata.write_h5ad(
                    os.path.join(self.prediction_output_dir, f"{self.cell_line}_within_true.h5ad")
                )

    def compute_bias_from_outputs(self, outputs, non_zero: bool = False):
        outputs = self._flatten_outputs(outputs)
        preds = self._concat_output_field(outputs, "preds")
        targets = self._concat_output_field(outputs, "X")
        pert_names = self._concat_output_field(outputs, "pert_name")
        if preds is None or targets is None or pert_names is None:
            return {}

        bias_dict = {}
        unique_perts = [p for p in np.unique(pert_names) if p != self.control_pert]
        self._log_message(f"{len(unique_perts)} Perturbations in training cell lines!")
        self._log_message("Calculate reconstruction bias for each perturbation across training cell lines...")
        for pert in unique_perts:
            mask = pert_names == pert
            pred_x_p = preds[mask]
            stim_x_p = targets[mask]
            if non_zero:
                nz_mask = stim_x_p != 0
                bias = stim_x_p - pred_x_p
                bias_masked = np.where(nz_mask, bias, np.nan)
                bias = np.nanmean(bias_masked, axis=0)
                bias_dict[pert] = np.nan_to_num(bias, nan=0.0)
            else:
                bias_dict[pert] = np.mean(stim_x_p - pred_x_p, axis=0)

        self._log_message("Bias calculation completed!")
        if bias_dict:
            first_key = next(iter(bias_dict))
            self._log_message(f"Example key: {first_key}, bias shape: {bias_dict[first_key].shape}")
            self._log_message(f"Total perturbations: {len(bias_dict)}")
        return bias_dict

    def bias(self, outputs, non_zero: bool = False):
        return self.compute_bias_from_outputs(outputs, non_zero=non_zero)

    @torch.no_grad()
    def compute_bias_from_dataloader(self, dataloader, non_zero: bool = False):
        if dataloader is None:
            return {}

        was_training = self.training
        self.eval()
        outputs = []
        device = self.device

        for batch in dataloader:
            batch = self.transfer_batch_to_device(batch, device, 0)
            x_pert, x_basal, pert, cell_type, batch_ids = self.extract_batch_tensors(batch)
            x_pert = self._restore_from_log_if_needed(x_pert)

            encoder_outputs, decoder_outputs = self.module.forward(
                x_basal, pert, cell_type, batch_ids
            )

            x_pred = decoder_outputs["px"].mu

            outputs.append(
                {
                    "preds": self._prediction_to_output_space(x_pred).detach().cpu(),
                    "X": batch.get("pert_cell_emb", None),
                    "pert_name": batch.get("pert_name", None),
                }
            )

        if was_training:
            self.train()

        return self.compute_bias_from_outputs(outputs, non_zero=non_zero)

    def save_bias_dict(self, bias_dict):
        if not self.save_bias or not bias_dict:
            return
        os.makedirs(self.prediction_output_dir, exist_ok=True)
        suffix = "cross" if self.mode == "cross" else "within"
        out_path = os.path.join(self.prediction_output_dir, f"{self.cell_line}_{suffix}_bias.pkl")
        with open(out_path, "wb") as handle:
            pickle.dump(bias_dict, handle)

    def _get_train_dataloader_for_bias(self):
        dataloader = getattr(self.trainer, "train_dataloader", None)
        if callable(dataloader):
            dataloader = dataloader()
        if isinstance(dataloader, (list, tuple)):
            dataloader = dataloader[0] if len(dataloader) > 0 else None
        return dataloader

    def on_train_start(self):
        self._log_message("--------Training--------")
        self._log_message("Training VAE...")

    def on_validation_epoch_end(self):
        return

    def on_train_epoch_end(self):
        epoch = int(self.current_epoch) + 1
        metrics = self.trainer.callback_metrics
        kl_loss = float(metrics.get("kl_loss", torch.tensor(0.0)).detach().cpu()) if metrics.get("kl_loss") is not None else 0.0
        recon_loss = float(metrics.get("recon_loss", torch.tensor(0.0)).detach().cpu()) if metrics.get("recon_loss") is not None else 0.0
        train_loss = float(metrics.get("train_loss", torch.tensor(recon_loss)).detach().cpu()) if metrics.get("train_loss") is not None else recon_loss
        val_loss = float(metrics.get("val_loss", torch.tensor(float("nan"))).detach().cpu()) if metrics.get("val_loss") is not None else float("nan")
        self._log_message(
            f"Epoch {epoch}: KL loss: {kl_loss},\t Recon loss: {recon_loss},\n Train Loss: {train_loss},\t Valid Loss: {val_loss}, \n"
        )

    def on_train_end(self):
        self._log_message("Training finished!")

    def on_predict_start(self):
        self._predict_outputs = []

    def on_predict_epoch_end(self):
        outputs = self._flatten_outputs(self._predict_outputs)
        self._save_prediction_artifacts(outputs)
        bias_dict = self.compute_bias_from_outputs(outputs, non_zero=self.bias_non_zero)
        self.save_bias_dict(bias_dict)
        self._predict_outputs = []

    @classmethod
    def from_pretrained_pert_embeddings(
        cls,
        pert_names: Iterable[str],
        pert_embedding_paths: Iterable[str],
        control_pert: str = "non-targeting",
        strict_pert_embeddings: bool = False,
        use_cell_type_embeddings: bool = False,
        use_batch_embeddings: bool = True,
        **kwargs,
    ):
        return cls(
            pert_names=pert_names,
            pert_embedding_paths=pert_embedding_paths,
            control_pert=control_pert,
            strict_pert_embeddings=strict_pert_embeddings,
            use_cell_type_embeddings=use_cell_type_embeddings,
            use_batch_embeddings=use_batch_embeddings,
            **kwargs,
        )

    def _build_networks(self):
        """
        Build the core components:
        """
        self.module = scVIModule(
            n_genes=self.input_dim,
            n_perts=self.n_perts,
            n_cell_types=self.n_cell_types,
            n_batches=self.n_batches,
            pert_embeddings=self.pert_embeddings,
            use_cell_type_embeddings=self.use_cell_type_embeddings,
            use_batch_embeddings=self.use_batch_embeddings,
            input_is_log1p=self.input_is_log1p,
            n_latent=self.n_latent,
            recon_loss=self.recon_loss,
            n_hidden_encoder=self.n_hidden_encoder,
            n_layers_encoder=self.n_layers_encoder,
            n_hidden_decoder=self.n_hidden_decoder,
            n_layers_decoder=self.n_layers_decoder,
            use_batch_norm=self.use_batch_norm,
            use_layer_norm=self.use_layer_norm,
            dropout_rate_encoder=self.dropout_rate_encoder,
            dropout_rate_decoder=self.dropout_rate_decoder,
            seed=self.seed,
        )

    def encode_perturbation(self, pert: torch.Tensor) -> torch.Tensor:
        """Map perturbation to an effect vector in embedding space."""
        raise NotImplementedError("Perturbation encoding not supported for scVI model")

    def encode_basal_expression(self, expr: torch.Tensor) -> torch.Tensor:
        """Expression is already in embedding space, pass through."""
        raise NotImplementedError(
            "Basal expression encoding not supported for scVI model"
        )

    def perturb(self, pert: torch.Tensor, basal: torch.Tensor) -> torch.Tensor:
        """
        Given a perturbation and basal embeddings, compute the perturbed embedding.
        """
        # Project perturbation and basal cell state to latent space
        raise NotImplementedError("Perturbation not supported for scVI model")

    @property
    def kl_weight(self):
        slope = 1.0
        if self.n_steps_kl_warmup is not None:
            global_step = self.global_step

            if global_step <= self.n_steps_kl_warmup:
                proportion = global_step / self.n_steps_kl_warmup
                return slope * proportion
            else:
                return slope
        elif self.n_epochs_kl_warmup is not None:
            current_epoch = self.current_epoch

            if current_epoch <= self.n_epochs_kl_warmup:
                proportion = current_epoch / self.n_epochs_kl_warmup
                return slope * proportion
            else:
                return slope
        else:
            return slope

    def extract_batch_tensors(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_pert = batch["pert_cell_emb"]
        x_basal = batch["ctrl_cell_emb"]
        pert = batch["pert_emb"]
        cell_type = batch.get("cell_type_onehot")
        batch_ids = batch.get("batch")

        # if pert is one-hot, convert to index
        if pert.dim() == 2 and pert.size(1) == self.n_perts:
            pert = pert.argmax(1)

        if cell_type is None:
            cell_type = torch.zeros(pert.shape[0], dtype=torch.long, device=pert.device)
        elif cell_type.dim() == 2 and cell_type.size(1) == self.n_cell_types:
            cell_type = cell_type.argmax(1)

        if batch_ids is None:
            batch_ids = torch.zeros(pert.shape[0], dtype=torch.long, device=pert.device)
        elif batch_ids.dim() == 2 and batch_ids.size(1) == self.n_batches:
            batch_ids = batch_ids.argmax(1)

        return x_pert, x_basal, pert, cell_type, batch_ids

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Given

        Args:
            batch: Dictionary containing:
                - pert: Perturbation one-hot
                - basal: Control expression embedding
                - cell_type: Cell type one-hot
                - batch: Batch one-hot
        """
        x_pert, x_basal, pert, cell_type, batch_ids = self.extract_batch_tensors(batch)

        encoder_outputs, decoder_outputs = self.module.forward(
            x_basal, pert, cell_type, batch_ids
        )

        output = decoder_outputs["px"].mu

        return output

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step logic."""
        x_pert, x_basal, pert, cell_type, batch_ids = self.extract_batch_tensors(batch)
        x_pert = self._restore_from_log_if_needed(x_pert)

        encoder_outputs, decoder_outputs = self.module.forward(
            x_basal, pert, cell_type, batch_ids
        )

        recon_loss, kl_loss = self.module.loss(
            x_pert=x_pert,
            encoder_outputs=encoder_outputs,
            decoder_outputs=decoder_outputs,
        )

        loss = recon_loss + self.kl_weight * kl_loss

        self.log(
            "KL_weight",
            self.kl_weight,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "recon_loss",
            recon_loss.item(),
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "kl_loss",
            kl_loss.item(),
            on_epoch=True,
            on_step=False,
            prog_bar=False,
            logger=True,
        )

        self.log(
            "train_loss",
            loss.item(),
            on_epoch=True,
            on_step=False,
            prog_bar=False,
            logger=True,
        )

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step logic."""
        x_pert, x_basal, pert, cell_type, batch_ids = self.extract_batch_tensors(batch)
        x_pert = self._restore_from_log_if_needed(x_pert)

        encoder_outputs, decoder_outputs = self.module.forward(
            x_basal, pert, cell_type, batch_ids
        )

        recon_loss, kl_loss = self.module.loss(
            x_pert=x_pert,
            encoder_outputs=encoder_outputs,
            decoder_outputs=decoder_outputs,
        )

        self.log("val_loss", recon_loss + self.kl_weight * kl_loss, prog_bar=True)

        # is_control = "DMSO_TF" == batch["pert_name"][0] or "non-targeting" == batch["pert_name"][0]
        # if np.random.rand() < 0.1 or is_control:
        #     x_pred = decoder_outputs["px"].mu

        #     self._update_val_cache(batch, x_pred)

    def predict_step(self, batch, batch_idx, **kwargs):
        """
        Typically used for final inference. We'll replicate old logic:
         returning 'preds', 'X', 'pert_name', etc.
        """
        x_pert, x_basal, pert, cell_type, batch_ids = self.extract_batch_tensors(batch)

        encoder_outputs, decoder_outputs = self.module.forward(
            x_basal, pert, cell_type, batch_ids
        )

        px = decoder_outputs["px"]
        use_sampling = kwargs.get("prediction_use_sampling", self.prediction_use_sampling)
        if use_sampling:
            x_pred = px.sample()
        else:
            x_pred = px.mu

        output = {
            "preds": self._prediction_to_output_space(x_pred),
            "X": batch.get("pert_cell_emb", None),
            "ctrl_cell_barcode": batch.get("ctrl_cell_barcode", None),
            "pert_name": batch.get("pert_name", None),
            "cell_type": batch.get("cell_type", None),
            "batch_name": batch.get("batch_name", None),
        }
        self._predict_outputs.append(output)
        return output

    def configure_optimizers(self):
        ae_params = (
            list(filter(lambda p: p.requires_grad, self.module.encoder.parameters()))
            + list(filter(lambda p: p.requires_grad, self.module.decoder.parameters()))
            + list(
                filter(
                    lambda p: p.requires_grad,
                    self.module.pert_embeddings.parameters(),
                )
            )
        )

        if self.module.cell_type_embeddings is not None:
            ae_params += list(
                filter(
                    lambda p: p.requires_grad,
                    self.module.cell_type_embeddings.parameters(),
                )
            )

        if self.module.batch_embeddings is not None:
            ae_params += list(
                filter(
                    lambda p: p.requires_grad,
                    self.module.batch_embeddings.parameters(),
                )
            )

        if self.module.recon_loss in ["zinb", "nb"]:
            ae_params += [self.module.px_r]

        optimizer_autoencoder = torch.optim.Adam(
            ae_params, lr=self.lr, weight_decay=self.wd
        )

        scheduler_autoencoder = StepLR(
            optimizer_autoencoder, step_size=self.step_size_lr, gamma=0.9
        )

        optimizers = [optimizer_autoencoder]
        schedulers = [scheduler_autoencoder]

        if self.step_size_lr is not None:
            return optimizers, schedulers
        else:
            return optimizers
