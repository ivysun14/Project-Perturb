import os
from pathlib import Path
import pickle
import shutil
from os.path import join, exists
from typing import List

import hydra
import anndata as ad
import torch
import sys

import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

from cell_load.utils.modules import get_datamodule

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scVI_src.models import SCVIPerturbationModel
from scVI_src.callbacks import BatchSpeedMonitorCallback
from scVI_src.models.scvi._utils import prepare_cell_load_kwargs, resolve_dataset_h5ad_path
from scVI_src.train.get_predictions import export_predictions_manual_h5ad

import logging


logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision("medium")


def setup_run_file_logger(run_output_dir: str) -> str:
    """Create a per-run train.log file and attach it to the active loggers."""
    log_path = join(run_output_dir, "train.log")
    formatter = logging.Formatter("[%(levelname)s] %(message)s")

    root_logger = logging.getLogger()
    file_handler_exists = any(
        isinstance(handler, logging.FileHandler)
        and os.path.abspath(getattr(handler, "baseFilename", "")) == os.path.abspath(log_path)
        for handler in root_logger.handlers
    )

    if not file_handler_exists:
        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    if root_logger.level > logging.INFO:
        root_logger.setLevel(logging.INFO)

    module_logger = logging.getLogger(__name__)
    if module_logger.level > logging.INFO:
        module_logger.setLevel(logging.INFO)

    module_has_file_handler = any(
        isinstance(handler, logging.FileHandler)
        and os.path.abspath(getattr(handler, "baseFilename", "")) == os.path.abspath(log_path)
        for handler in module_logger.handlers
    )
    if not module_has_file_handler:
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        module_logger.addHandler(file_handler)

    return log_path


def get_lightning_module(
    model_type: str,
    data_config: dict,
    model_config: dict,
    training_config: dict,
    var_dims: dict,
):
    """Create model instance based on config."""
    # combine the model config and training config
    module_config = {**model_config, **training_config}
    module_config["embed_key"] = data_config["embed_key"]
    module_config["output_space"] = data_config["output_space"]
    module_config["gene_names"] = var_dims["gene_names"]

    print(f"Found {len(module_config['gene_names'])} genes in the data")
    module_config["batch_size"] = training_config["batch_size"]

    if data_config["output_space"] == "gene":
        gene_dim = var_dims["hvg_dim"]
    else:
        gene_dim = var_dims["gene_dim"]

    if model_type.lower() == "scvi":
        return SCVIPerturbationModel(
            input_dim=var_dims["input_dim"],
            gene_dim=gene_dim,
            hvg_dim=var_dims["hvg_dim"],
            output_dim=var_dims["output_dim"],
            pert_dim=var_dims["pert_dim"],
            batch_dim=var_dims["batch_dim"],
            pert_names=var_dims.get("pert_names"),
            **module_config,
        )
    else:
        raise ValueError(f"Only scvi is supported in this trimmed baseline, got: {model_type}")


def log_dataset_summary(dataset_path: str, data_kwargs: dict, dm=None) -> None:
    logger.info("Dataset file: %s", dataset_path)
    logger.info("Split CSV: %s", data_kwargs.get("split_csv_path"))
    logger.info("Control perturbation: %s", data_kwargs.get("control_pert"))
    logger.info("Perturbation column: %s", data_kwargs.get("pert_col"))
    logger.info("Batch column: %s", data_kwargs.get("batch_col"))
    logger.info("Cell type column: %s", data_kwargs.get("cell_type_key"))

    adata = ad.read_h5ad(dataset_path, backed="r")
    try:
        logger.info("Dataset shape: cells=%d, genes=%d", adata.n_obs, adata.n_vars)
        obs = adata.obs

        pert_col = data_kwargs.get("pert_col")
        if pert_col in obs.columns:
            n_perts = int(obs[pert_col].astype(str).nunique())
            logger.info("Unique perturbations: %d", n_perts)

        batch_col = data_kwargs.get("batch_col")
        if batch_col in obs.columns:
            n_batches = int(obs[batch_col].astype(str).nunique())
            logger.info("Unique batches: %d", n_batches)

        cell_type_key = data_kwargs.get("cell_type_key")
        if cell_type_key in obs.columns:
            n_cell_types = int(obs[cell_type_key].astype(str).nunique())
            logger.info("Unique cell types: %d", n_cell_types)
    finally:
        file_manager = getattr(adata, "file", None)
        if file_manager is not None:
            file_manager.close()

    if dm is not None:
        logger.info("Datamodule maps: cell_types=%d, perts=%d, batches=%d",
                    len(getattr(dm, "cell_type_onehot_map", {})),
                    len(getattr(dm, "pert_onehot_map", {})),
                    len(getattr(dm, "batch_onehot_map", {})))

        def _dataset_size(loader_fn_name: str):
            try:
                loader = getattr(dm, loader_fn_name)()
                dataset = getattr(loader, "dataset", None)
                return len(dataset) if dataset is not None else None
            except Exception:
                return None

        train_n = _dataset_size("train_dataloader")
        val_n = _dataset_size("val_dataloader")
        test_n = _dataset_size("test_dataloader")
        logger.info("Split sizes: train=%s, val=%s, test=%s", train_n, val_n, test_n)


def get_loggers(
    output_dir: str,
    name: str,
) -> List:
    """Set up local CSV logging."""
    csv_logger = CSVLogger(save_dir=output_dir, name=name, version=0)
    return [csv_logger]


def get_checkpoint_callbacks(
    output_dir: str, name: str, val_freq: int, ckpt_every_n_steps: int
) -> List[ModelCheckpoint]:
    """Create checkpoint callbacks based on validation frequency."""
    checkpoint_dir = join(output_dir, name, "checkpoints")
    callbacks = []

    # Save best checkpoint based on validation loss
    best_ckpt = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="step={step}-val_loss={val_loss:.4f}",
        auto_insert_metric_name=False,
        save_last="link",  # Will create last.ckpt symlink to best checkpoint
        monitor="val_loss",
        mode="min",
        save_top_k=1,  # Only keep the best checkpoint
        every_n_train_steps=val_freq,
    )
    callbacks.append(best_ckpt)

    # Also save periodic checkpoints (without affecting the "last" symlink)
    periodic_ckpt = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{step}",
        save_last=False,  # Don't create/update symlink
        every_n_train_steps=ckpt_every_n_steps,
        save_top_k=-1,  # Keep all periodic checkpoints
    )
    callbacks.append(periodic_ckpt)

    return callbacks


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: DictConfig) -> None:
    """Main training function."""
    # Convert config to YAML for logging
    cfg_yaml = OmegaConf.to_yaml(cfg, resolve=True)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    print(cfg_yaml)

    # Setup output directory
    base_name = cfg["name"]
    cfg["name"] = base_name
    run_output_dir = join(cfg["output_dir"], cfg["name"])
    if os.path.exists(run_output_dir) and cfg["overwrite"]:
        print(f"Output dir {run_output_dir} already exists, overwriting")
        shutil.rmtree(run_output_dir)
    os.makedirs(run_output_dir, exist_ok=True)
    log_path = setup_run_file_logger(run_output_dir)
    logger.info("Run output dir: %s", run_output_dir)
    logger.info("Training log file: %s", log_path)
    logger.info("Resolved config:\n%s", cfg_yaml)

    # Set random seeds
    if "train_seed" in cfg["training"]:
        pl.seed_everything(cfg["training"]["train_seed"])

    # if the provided pert_col is drugname_drugconc, hard code the value of control pert
    # this is because it's surprisingly hard to specify a list of tuples in the config as a string
    if cfg["data"]["kwargs"]["pert_col"] == "drugname_drugconc":
        cfg["data"]["kwargs"]["control_pert"] = "[('DMSO_TF', 0.0, 'uM')]"

    cfg["data"]["kwargs"].setdefault(
        "train_valid_seed",
        cfg["training"].get("train_seed", 42),
    )

    cfg["model"]["kwargs"]["control_pert"] = cfg["data"]["kwargs"]["control_pert"]
    cfg["model"]["kwargs"]["prediction_output_dir"] = join(run_output_dir, "predictions")
    cfg["model"]["kwargs"]["model_path"] = join(run_output_dir, "checkpoints")
    cfg["model"]["kwargs"]["save_epoch_checkpoints"] = False

    # Initialize data module. this is backwards compatible with previous configs
    try:
        sentence_len = cfg["model"]["cell_set_len"]
    except KeyError:
        if cfg["model"]["name"].lower() in [
            "cpa",
            "gears",
            "scvi",
            "lowranklinear",
        ] or cfg["model"]["name"].lower().startswith("scgpt"):
            if (
                "cell_sentence_len" in cfg["model"]["kwargs"]
                and cfg["model"]["kwargs"]["cell_sentence_len"] > 1
            ):
                sentence_len = cfg["model"]["kwargs"]["cell_sentence_len"]
                cfg["training"]["batch_size"] = 1
            else:
                sentence_len = 1
        else:
            sentence_len = cfg["model"]["kwargs"]["transformer_backbone_kwargs"][
                "n_positions"
            ]

    cfg["data"]["kwargs"] = prepare_cell_load_kwargs(
        cfg["data"]["kwargs"],
        output_dir=run_output_dir,
        default_dataset_name=cfg["model"]["kwargs"].get("cell_line", cfg["name"]),
        default_cell_type_name=cfg["model"]["kwargs"].get("cell_line", cfg["name"]),
        mode=cfg["model"]["kwargs"].get("mode", "within"),
        cross_cell_line=cfg["model"]["kwargs"].get("cell_line", cfg["name"]),
    )

    dataset_path = resolve_dataset_h5ad_path(
        cfg["data"]["kwargs"]["dataset_dir"],
        cfg["data"]["kwargs"]["dataset_name"],
    )
    log_dataset_summary(dataset_path, cfg["data"]["kwargs"], dm=None)

    dm = get_datamodule(
        name=cfg["data"]["name"],
        kwargs=cfg["data"]["kwargs"],
        batch_size=cfg["training"]["batch_size"],
        cell_sentence_len=sentence_len,
    )

    dm.setup()
    log_dataset_summary(dataset_path, cfg["data"]["kwargs"], dm=dm)

    if cfg["model"]["name"].lower() == "scvi":
        cfg["model"]["kwargs"]["n_cell_types"] = len(dm.cell_type_onehot_map)
        cfg["model"]["kwargs"]["n_perts"] = len(dm.pert_onehot_map)
        cfg["model"]["kwargs"]["n_batches"] = len(dm.batch_onehot_map)
        cfg["model"]["kwargs"]["dataset_dir"] = cfg["data"]["kwargs"]["dataset_dir"]
        cfg["model"]["kwargs"]["dataset_name"] = cfg["data"]["kwargs"]["dataset_name"]
        cfg["model"]["kwargs"]["pert_col"] = cfg["data"]["kwargs"]["pert_col"]

    with open(join(run_output_dir, "config.yaml"), "w") as f:
        # f.write()
        new_cfg_yaml = OmegaConf.to_yaml(cfg, resolve=True)
        f.write(new_cfg_yaml)

    dm_var_dims = dm.get_var_dims()

    # if cfg["model"]["name"].lower() == "gears":
    # Create model
    model = get_lightning_module(
        cfg["model"]["name"],
        cfg["data"]["kwargs"],
        cfg["model"]["kwargs"],
        cfg["training"],
        dm_var_dims,
    )

    # Set up logging
    loggers = get_loggers(
        output_dir=cfg["output_dir"],
        name=cfg["name"],
    )

    # Set up callbacks
    ckpt_callbacks = get_checkpoint_callbacks(
        cfg["output_dir"],
        cfg["name"],
        cfg["training"]["val_freq"],
        cfg["training"].get("ckpt_every_n_steps", 4000),
    )
    # Log batch throughput metrics through Lightning's configured logger.
    batch_speed_monitor = BatchSpeedMonitorCallback()
    callbacks = ckpt_callbacks + [batch_speed_monitor]

    logger.info("Loggers and callbacks set up.")

    plugins = []

    if torch.cuda.is_available():
        accelerator = "gpu"
    else:
        accelerator = "cpu"

    # Decide on trainer params
    trainer_kwargs = dict(
        accelerator=accelerator,
        devices=1,
        max_steps=cfg["training"].get("max_steps", -1),  # for normal models
        max_epochs=cfg["training"].get("max_epochs", -1),
        check_val_every_n_epoch=None,
        val_check_interval=cfg["training"]["val_freq"],
        logger=loggers,
        plugins=plugins,
        callbacks=callbacks,
        gradient_clip_val=cfg["training"].get("gradient_clip_val", None),
    )

    # Build trainer
    trainer = pl.Trainer(**trainer_kwargs)

    # Load checkpoint if exists
    checkpoint_path = join(ckpt_callbacks[0].dirpath, "last.ckpt")
    if not exists(checkpoint_path):
        checkpoint_path = None
    else:
        logging.info(f"!! Resuming training from {checkpoint_path} !!")

    logger.info("Starting trainer fit.")

    # Train
    trainer.fit(
        model,
        datamodule=dm,
        ckpt_path=checkpoint_path,
    )

    # at this point if checkpoint_path does not exist, manually create one
    checkpoint_path = join(ckpt_callbacks[0].dirpath, "final.ckpt")
    if not exists(checkpoint_path):
        trainer.save_checkpoint(checkpoint_path)

    # save the data_module
    with open(join(run_output_dir, "data_module.pkl"), "wb") as f:
        pickle.dump(dm, f)

    if not cfg["data"]["kwargs"].get("test_split_groups"):
        logger.info("Skipping post-training prediction export because test_split_groups is not set.")
        return

    predict_ckpt_path = ckpt_callbacks[0].best_model_path or checkpoint_path
    test_loader = dm.test_dataloader()
    if test_loader is None:
        logger.info("Skipping prediction export because test dataloader is unavailable.")
        return

    logger.info("Starting post-training prediction export using checkpoint: %s", predict_ckpt_path)
    predict_model = model.__class__.load_from_checkpoint(
        predict_ckpt_path,
        weights_only=False,
        **model.hparams,
    )
    export_predictions_manual_h5ad(
        model=predict_model,
        test_loader=test_loader,
        cfg=cfg,
        dataset_path=dataset_path,
        prediction_dir=join(run_output_dir, "predictions"),
        sample_multiplier=cfg["model"]["kwargs"].get("prediction_sample_multiplier", 1),
        show_progress=False,
    )
    logger.info("Post-training prediction export finished.")


if __name__ == "__main__":
    train()
