#!/usr/bin/env python
"""
Prediction export utilities for trained scVI perturbation models.
This module loads a saved run, filters prediction splits, runs a manual
inference loop, and writes VAE/scGen-style h5ad files for downstream analysis.
"""

import argparse
import logging
import math
import os
import sys
from bisect import bisect_right
from pathlib import Path

import anndata
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Subset
from tqdm import tqdm

from cell_load.utils.modules import get_datamodule
from scVI_src.models.scvi._utils import (
    _load_dropped_test_perts,
    _read_split_genes,
    _subset_adata_gene_space,
    resolve_dataset_h5ad_path,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.multiprocessing.set_sharing_strategy("file_system")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    """
    Parse command-line options for prediction export.

    Returns
    -------
    argparse.Namespace
        Prediction export arguments, including checkpoint and split overrides.
    """
    parser = argparse.ArgumentParser(description="Get predictions from a trained PerturbationModel.")
    parser.add_argument("--output_dir", "-o", type=str, required=True)
    parser.add_argument("--checkpoint", "-c", type=str, default="last.ckpt")
    parser.add_argument(
        "--prediction_sample_multiplier",
        "--sample_size",
        "--samplesize",
        type=float,
        default=None,
        help="Multiplier per test perturbation (1=original).",
    )
    parser.add_argument("--split_csv_path", type=str, default=None)
    parser.add_argument("--split_column", type=str, default=None)
    parser.add_argument(
        "--use_split_gene_space",
        type=str,
        choices=("true", "false", "True", "False", "1", "0", "yes", "no"),
        default=None,
    )
    parser.add_argument("--train_split_groups", type=str, default=None)
    parser.add_argument("--test_split_groups", type=str, default=None)
    parser.add_argument("--calib_split_groups", type=str, default=None)
    parser.add_argument("--dropped_test_perts_json", type=str, default=None)
    parser.add_argument(
        "--toml_name",
        type=str,
        default=None,
        help="Prediction-time Cell Load TOML name/path. Relative paths are resolved under output_dir.",
    )
    parser.add_argument(
        "--prediction_use_sampling",
        type=str,
        choices=("true", "false", "True", "False", "1", "0", "yes", "no"),
        default=None,
    )
    return parser.parse_args()


def parse_bool(value) -> bool:
    """
    Parse common string and boolean forms into a bool.
    """
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Expected a boolean value, got {value!r}")


def load_config(cfg_path: str) -> dict:
    """
    Load the saved YAML config for a trained run.
    """
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Could not find config file: {cfg_path}")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def _resolve_split_col(split_df: pd.DataFrame, split_column: str | None) -> str:
    candidates = []
    for col in (split_column, "split", "split_calib"):
        if col and col not in candidates:
            candidates.append(col)
    resolved = next((c for c in candidates if c in split_df.columns), None)
    if resolved is None:
        raise ValueError(
            "Split CSV must contain a split column. "
            f"Tried {candidates}, available={sorted(split_df.columns.tolist())}."
        )
    return resolved


def select_best_checkpoint(checkpoint_dir: str) -> str:
    """
    Select the checkpoint with the lowest val_loss encoded in its filename.
    """
    best = None
    for name in os.listdir(checkpoint_dir):
        if not name.endswith(".ckpt") or "val_loss" not in name:
            continue
        try:
            loss = float(name.rsplit("val_loss=", 1)[1].removesuffix(".ckpt"))
        except ValueError:
            continue
        path = os.path.join(checkpoint_dir, name)
        if best is None or loss < best[0]:
            best = (loss, path)
    if best is None:
        raise FileNotFoundError(f"No val_loss checkpoint found under {checkpoint_dir}")
    return best[1]


def resolve_checkpoint_path(checkpoint_dir: str, checkpoint_name: str) -> str:
    """
    Resolve a checkpoint alias, absolute path, exact filename, or partial name.
    """
    if os.path.isabs(checkpoint_name) and os.path.exists(checkpoint_name):
        return checkpoint_name
    if checkpoint_name == "best":
        return select_best_checkpoint(checkpoint_dir)
    direct_path = os.path.join(checkpoint_dir, checkpoint_name)
    if os.path.exists(direct_path):
        return direct_path

    files = [n for n in os.listdir(checkpoint_dir) if n.endswith(".ckpt")]
    if checkpoint_name in files:
        return os.path.join(checkpoint_dir, checkpoint_name)

    normalized = checkpoint_name.replace("step=step=", "step=").replace("val_loss=val_loss=", "val_loss=")
    if normalized != checkpoint_name:
        normalized_path = os.path.join(checkpoint_dir, normalized)
        if os.path.exists(normalized_path):
            logger.info("Resolved checkpoint alias %s -> %s", checkpoint_name, normalized)
            return normalized_path

    req_stem = checkpoint_name[:-5] if checkpoint_name.endswith(".ckpt") else checkpoint_name
    norm_stem = normalized[:-5] if normalized.endswith(".ckpt") else normalized
    matching = [n for n in files if req_stem in n or norm_stem in n]
    if len(matching) == 1:
        logger.info("Resolved checkpoint by partial match %s -> %s", checkpoint_name, matching[0])
        return os.path.join(checkpoint_dir, matching[0])

    raise FileNotFoundError(
        f"Could not find checkpoint at {direct_path}.\nAvailable checkpoints: {sorted(files)}"
    )


def _build_var_frame(adata: anndata.AnnData) -> pd.DataFrame:
    if "gene_name" in adata.var.columns:
        return adata.var[["gene_name"]].copy()
    return pd.DataFrame({"gene_name": adata.var_names.astype(str)}, index=adata.var_names)


def load_control_adata_for_export(
    dataset_path: str,
    pert_col: str,
    control_pert: str,
    cell_type_key: str | None = None,
    cell_type_value: str | None = None,
    gene_space: list[str] | None = None,
) -> anndata.AnnData:
    """
    Load control cells used to anchor exported prediction AnnData objects.

    Parameters
    ----------
    dataset_path : str
        Source h5ad dataset.
    pert_col : str
        Observation column containing perturbation labels.
    control_pert : str
        Perturbation label used for control cells.
    """
    adata = anndata.read_h5ad(dataset_path, backed="r")
    try:
        if pert_col not in adata.obs.columns:
            raise KeyError(f"Column {pert_col} not found in dataset obs.")
        control_mask = adata.obs[pert_col].astype(str).to_numpy() == str(control_pert)
        control_adata = adata[control_mask].to_memory()
        if gene_space is not None:
            control_adata = _subset_adata_gene_space(control_adata, gene_space)
    finally:
        file_manager = getattr(adata, "file", None)
        if file_manager is not None:
            file_manager.close()
    obs = pd.DataFrame({pert_col: [control_pert] * control_adata.n_obs}, index=control_adata.obs_names)
    if cell_type_key and cell_type_value is not None:
        obs[cell_type_key] = [str(cell_type_value)] * control_adata.n_obs
    return anndata.AnnData(X=control_adata.X, obs=obs, var=_build_var_frame(control_adata))


def load_var_frame_for_export(dataset_path: str, gene_space: list[str] | None = None) -> pd.DataFrame:
    adata = anndata.read_h5ad(dataset_path, backed="r")
    try:
        if gene_space is not None:
            adata = _subset_adata_gene_space(adata.to_memory(), gene_space)
        if "gene_name" in adata.var.columns:
            return adata.var[["gene_name"]].copy()
        return pd.DataFrame({"gene_name": adata.var_names.astype(str)}, index=adata.var_names)
    finally:
        file_manager = getattr(adata, "file", None)
        if file_manager is not None:
            file_manager.close()


def resolve_export_gene_space(cfg: dict) -> list[str] | None:
    """
    Resolve the optional split-defined gene space for within-cell export.
    """
    data_kwargs = cfg["data"]["kwargs"]
    if not parse_bool(data_kwargs.get("use_split_gene_space", False)):
        return None

    split_csv_path = data_kwargs.get("split_csv_path")
    if not split_csv_path:
        raise ValueError("use_split_gene_space is true, but data.kwargs.split_csv_path is not set.")

    split_gene_col = data_kwargs.get("split_gene_col", "gene")
    gene_space = _read_split_genes(split_csv_path, split_gene_col)
    logger.info(
        "Using split gene space for prediction export: %d genes from %s column %s.",
        len(gene_space),
        split_csv_path,
        split_gene_col,
    )
    return gene_space


def build_vae_scgen_style_adata(
    matrix: np.ndarray,
    pert_names: list,
    control_adata: anndata.AnnData | None,
    pert_col: str,
    cell_types: list[str] | None = None,
    cell_type_key: str | None = None,
    var_frame: pd.DataFrame | None = None,
) -> anndata.AnnData:
    """
    Build an AnnData export containing predictions and optional control cells.

    Parameters
    ----------
    matrix : np.ndarray
        Predicted or true expression matrix.
    pert_names : list
        Perturbation label for each matrix row.
    control_adata : AnnData, optional
        Control cells to append to the exported object.
    """
    if control_adata is not None and matrix.shape[1] != control_adata.n_vars:
        raise ValueError(
            f"Prediction matrix width {matrix.shape[1]} does not match control gene dimension {control_adata.n_vars}."
        )
    pred_obs = pd.DataFrame({pert_col: [str(name) for name in pert_names]})
    if cell_types is not None and cell_type_key:
        if len(cell_types) != matrix.shape[0]:
            raise ValueError(
                f"cell_types length {len(cell_types)} does not match prediction rows {matrix.shape[0]}."
            )
        pred_obs[cell_type_key] = [str(name) for name in cell_types]

    if control_adata is not None:
        pred_adata = anndata.AnnData(X=matrix, obs=pred_obs, var=control_adata.var.copy())
        return anndata.concat([pred_adata, control_adata.copy()], axis=0, join="inner", merge="same", index_unique=None)

    if var_frame is not None:
        pred_var = var_frame.copy()
    else:
        pred_var = pd.DataFrame({"gene_name": [f"gene_{i}" for i in range(matrix.shape[1])]})
    return anndata.AnnData(X=matrix, obs=pred_obs, var=pred_var)


def validate_prediction_sample_multiplier(value) -> float:
    """
    Validate and normalize the prediction sample multiplier.
    """
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"prediction_sample_multiplier must be a positive finite number, got {value!r}")
    return value


def format_prediction_sample_multiplier(value: float) -> str:
    return f"{float(value):g}"


def _as_list(value, expected_len: int) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value] * expected_len


def _normalize_row_values(value, expected_len: int, default_value: str = "unknown") -> list[str]:
    if expected_len <= 0:
        return []
    if value is None:
        return [default_value] * expected_len

    arr = None
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    elif isinstance(value, np.ndarray):
        arr = value

    if arr is not None:
        flat = np.asarray(arr).reshape(-1)
        if flat.size == expected_len:
            return [str(item) for item in flat.tolist()]
        if flat.size == 1:
            return [str(flat.item())] * expected_len

    if isinstance(value, (list, tuple)):
        if len(value) == expected_len:
            return [str(item) for item in value]
        if len(value) == 1:
            return [str(value[0])] * expected_len

    return [str(value)] * expected_len


def post_process_preds(batch_preds, model_class_name):
    del model_class_name
    if "cell_mask" in batch_preds:
        filtered = {}
        cell_mask = batch_preds["cell_mask"]
        for k, v in batch_preds.items():
            if k == "cell_mask":
                continue
            if isinstance(v, torch.Tensor):
                filtered[k] = v[cell_mask]
            elif isinstance(v, list) and v and v[0] is not None:
                filtered[k] = [v[i] for i in range(len(v)) if cell_mask[i]]
            else:
                filtered[k] = v
    else:
        filtered = batch_preds

    gene_mask = None
    if "gene_mask" in batch_preds:
        gm = batch_preds["gene_mask"].detach().cpu().numpy()
        gene_mask = gm[0, :].astype(bool)
    return filtered, gene_mask


def _resolve_sentence_dataset_and_index(concat_dataset, sentence: list[int]):
    if not sentence:
        return None, None
    global_idx = int(sentence[0])
    dataset_idx = bisect_right(concat_dataset.cumulative_sizes, global_idx)
    sample_idx = global_idx if dataset_idx == 0 else global_idx - concat_dataset.cumulative_sizes[dataset_idx - 1]
    subset = concat_dataset.datasets[dataset_idx]
    if isinstance(subset, Subset) or (hasattr(subset, "dataset") and hasattr(subset, "indices")):
        base_dataset = subset.dataset
        file_idx = int(subset.indices[sample_idx])
    else:
        base_dataset = subset
        file_idx = sample_idx
    return base_dataset, file_idx


def _sentence_metadata(concat_dataset, sentence: list[int]) -> tuple[str, str, str]:
    base_dataset, file_idx = _resolve_sentence_dataset_and_index(concat_dataset, sentence)
    if base_dataset is None:
        return "unknown", "unknown", "unknown"
    pert_code = base_dataset.metadata_cache.pert_codes[file_idx]
    pert_name = str(base_dataset.pert_categories[pert_code])
    dataset_name = str(getattr(base_dataset, "name", "unknown"))
    try:
        cell_type_code = int(base_dataset.metadata_cache.cell_type_codes[file_idx])
        cell_type_name = str(base_dataset.cell_type_categories[cell_type_code])
    except Exception:
        cell_type_name = dataset_name.split("_", 1)[0] if dataset_name != "unknown" else "unknown"
    return pert_name, cell_type_name, dataset_name


def _sentence_perturbation_name(concat_dataset, sentence: list[int]) -> str:
    pert_name, _, _ = _sentence_metadata(concat_dataset, sentence)
    return pert_name


def apply_prediction_sample_multiplier_to_loader(test_loader, multiplier: float) -> None:
    """
    Resample test-loader sentences per perturbation before prediction.

    Multipliers below one downsample each perturbation group, while values above
    one repeat and sample sentences to reach the requested approximate size.
    """
    sampler = getattr(test_loader, "batch_sampler", None)
    if sampler is None or not hasattr(sampler, "sentences") or not hasattr(sampler, "_create_batches"):
        raise ValueError("test_loader.batch_sampler does not support prediction sample scaling.")

    original_sentences = [list(s) for s in sampler.sentences]
    if not original_sentences:
        return
    original_batch_count = len(getattr(sampler, "batches", []))
    if math.isclose(multiplier, 1.0):
        logger.info("Prediction sample multiplier is 1; keeping %d sampler batches unchanged.", original_batch_count)
        return

    by_pert: dict[str, list[list[int]]] = {}
    for sentence in original_sentences:
        by_pert.setdefault(_sentence_perturbation_name(sampler.dataset, sentence), []).append(sentence)

    rng = np.random.default_rng(0)
    scaled = []
    for pert_name in sorted(by_pert):
        sents = by_pert[pert_name]
        target = max(1, int(math.floor(len(sents) * multiplier + 0.5)))
        if target <= len(sents):
            idx = rng.choice(len(sents), size=target, replace=False)
            scaled.extend([sents[i] for i in idx])
        else:
            repeats, rem = divmod(target, len(sents))
            scaled.extend(sents * repeats)
            if rem:
                idx = rng.choice(len(sents), size=rem, replace=False)
                scaled.extend([sents[i] for i in idx])

    rng.shuffle(scaled)
    sampler.sentences = [list(s) for s in scaled]
    sampler.batches = sampler._create_batches()
    sampler.tot_num = int(sum(len(s) for s in sampler.sentences))
    logger.info(
        "Applied prediction sample multiplier %s: sentences %d -> %d, batches %d -> %d.",
        multiplier,
        len(original_sentences),
        len(sampler.sentences),
        original_batch_count,
        len(sampler.batches),
    )


def _parse_optional_groups(raw_groups) -> list[str]:
    if raw_groups is None:
        return []
    if isinstance(raw_groups, str):
        parsed = [p.strip().lower() for p in raw_groups.replace(",", " ").split() if p.strip()]
    else:
        parsed = [str(p).strip().lower() for p in raw_groups if str(p).strip()]
    return list(dict.fromkeys(parsed))


def _resolve_branch2_groups(data_kwargs: dict) -> tuple[list[str], list[str]]:
    test_groups = _parse_optional_groups(data_kwargs.get("test_split_groups"))
    calib_groups = _parse_optional_groups(data_kwargs.get("calib_split_groups"))
    return test_groups, calib_groups


def _parse_simple_cell_load_toml(toml_path: str) -> dict[str, dict[str, str]]:
    sections: dict[str, dict[str, str]] = {"datasets": {}, "training": {}, "zeroshot": {}}
    current_section = None
    with open(toml_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                current_section = line[1:-1].strip()
                sections.setdefault(current_section, {})
                continue
            if current_section is None or "=" not in line:
                continue

            key_raw, value_raw = line.split("=", 1)
            key = key_raw.strip()
            value = value_raw.strip()
            if key.startswith('"') and key.endswith('"'):
                key = key[1:-1]
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            sections[current_section][key] = value
    return sections


def _toml_quote(value: str) -> str:
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _write_simple_cell_load_toml(toml_path: str, sections: dict[str, dict[str, str]]) -> None:
    with open(toml_path, "w", encoding="utf-8") as handle:
        handle.write("# Auto-generated prediction Cell Load config.\n")
        for idx, section in enumerate(("datasets", "training", "zeroshot")):
            handle.write(f"[{section}]\n")
            for key, value in sections.get(section, {}).items():
                key_token = _toml_quote(key) if "." in key else key
                handle.write(f"{key_token} = {_toml_quote(value)}\n")
            if idx < 2:
                handle.write("\n")


def _resolve_prediction_toml_path(output_dir: str, toml_name: str | None) -> str:
    if toml_name is None or not str(toml_name).strip():
        return str(Path(output_dir) / "cell_load_prediction_filtered.toml")
    target = Path(str(toml_name).strip())
    if not target.is_absolute():
        target = Path(output_dir) / target
    return str(target)


def prepare_cross_prediction_toml(cfg_dict: dict, output_dir: str, toml_name: str | None) -> str:
    """
    Write a prediction-only Cell Load TOML for cross-cell-line export.

    The generated config selects the held-out test dataset and/or non-held-out
    calibration datasets requested by the CLI.
    """
    data_kwargs = cfg_dict["data"]["kwargs"]
    base_toml = data_kwargs.get("toml_config_path")
    if not base_toml:
        raise ValueError("Cross prediction requires data.kwargs.toml_config_path.")
    if not os.path.exists(base_toml):
        raise FileNotFoundError(f"Cross prediction TOML not found: {base_toml}")

    test_groups, calib_groups = _resolve_branch2_groups(data_kwargs)
    if not test_groups and not calib_groups:
        raise ValueError(
            "Both test_split_groups and calib_split_groups are empty; no cross-cell-line prediction target remains."
        )

    parsed = _parse_simple_cell_load_toml(base_toml)
    datasets = parsed.get("datasets", {})
    training = parsed.get("training", {})
    zeroshot = parsed.get("zeroshot", {})
    if not datasets:
        raise ValueError(f"Invalid Cell Load TOML (missing [datasets]): {base_toml}")

    holdout_cell_line = str(cfg_dict["model"]["kwargs"].get("cell_line", cfg_dict.get("name", "")))
    selected_test_entries: list[tuple[str, str]] = []

    if test_groups:
        holdout_dataset = f"{holdout_cell_line}_test"
        if holdout_dataset not in datasets:
            raise ValueError(
                f"Expected holdout test dataset {holdout_dataset!r} in {base_toml}, "
                f"available={sorted(datasets.keys())}."
            )
        selected_test_entries.append((holdout_dataset, holdout_cell_line))

    if calib_groups:
        non_holdout_calib = []
        for dataset_name in sorted(datasets):
            if not dataset_name.endswith("_calib"):
                continue
            cell_line = dataset_name.rsplit("_", 1)[0]
            if cell_line == holdout_cell_line:
                continue
            non_holdout_calib.append((dataset_name, cell_line))
        if not non_holdout_calib:
            raise ValueError(
                f"Requested calib_split_groups={calib_groups}, but no non-holdout *_calib datasets exist in {base_toml}."
            )
        selected_test_entries.extend(non_holdout_calib)

    if not selected_test_entries:
        raise ValueError(
            "No test entries selected for cross prediction. "
            f"test_split_groups={test_groups}, calib_split_groups={calib_groups}."
        )

    new_zeroshot = {key: value for key, value in zeroshot.items() if str(value).strip().lower() != "test"}
    for dataset_name, cell_line in selected_test_entries:
        new_zeroshot[f"{dataset_name}.{cell_line}"] = "test"

    prediction_toml = _resolve_prediction_toml_path(output_dir, toml_name)
    _write_simple_cell_load_toml(
        prediction_toml,
        {"datasets": datasets, "training": training, "zeroshot": new_zeroshot},
    )

    # Keep exported control cells aligned with the active prediction test source.
    first_dataset_name = selected_test_entries[0][0]
    first_dataset_path = Path(datasets[first_dataset_name]).resolve()
    data_kwargs["dataset_dir"] = str(first_dataset_path.parent)
    data_kwargs["dataset_name"] = first_dataset_path.stem
    data_kwargs["toml_config_path"] = prediction_toml

    logger.info(
        "Prepared cross prediction TOML %s with test entries=%s "
        "(holdout groups=%s, non-holdout calib groups=%s).",
        prediction_toml,
        [f"{dataset}.{cell}" for dataset, cell in selected_test_entries],
        test_groups,
        calib_groups,
    )
    return prediction_toml


def load_model_with_pert_embedding_fallback(ModelClass, checkpoint_path: str, model_init_kwargs: dict):
    """
    Load a checkpoint normally first. If only perturbation embedding table size differs,
    keep the current model's perturbation embeddings and load all other weights.
    """
    try:
        return ModelClass.load_from_checkpoint(checkpoint_path, weights_only=False, **model_init_kwargs)
    except RuntimeError as exc:
        error_text = str(exc)
        if "size mismatch for module.pert_embeddings.weight" not in error_text:
            raise

        logger.warning(
            "Checkpoint perturbation embedding size mismatch detected; "
            "falling back to partial state_dict load while keeping current pert embeddings."
        )

        try:
            # PyTorch 2.6 changed torch.load default to weights_only=True, which
            # breaks loading full Lightning checkpoints in this fallback path.
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict")
        if state_dict is None:
            raise RuntimeError(f"Checkpoint is missing 'state_dict': {checkpoint_path}") from exc

        model = ModelClass(**model_init_kwargs)
        current_state = model.state_dict()

        dropped_keys = []
        for key in list(state_dict.keys()):
            if key not in current_state:
                continue
            ckpt_tensor = state_dict[key]
            current_tensor = current_state[key]
            ckpt_shape = tuple(getattr(ckpt_tensor, "shape", []))
            current_shape = tuple(getattr(current_tensor, "shape", []))
            if ckpt_shape == current_shape:
                continue

            if key.endswith("pert_embeddings.weight"):
                dropped_keys.append((key, ckpt_shape, current_shape))
                state_dict.pop(key, None)
            else:
                raise RuntimeError(
                    "Found non-perturbation shape mismatch while loading checkpoint: "
                    f"{key}: ckpt={ckpt_shape}, current={current_shape}"
                ) from exc

        if not dropped_keys:
            raise

        incompatible = model.load_state_dict(state_dict, strict=False)
        missing_keys = set(incompatible.missing_keys)
        unexpected_keys = set(incompatible.unexpected_keys)
        allowed_missing = {key for key, _, _ in dropped_keys}
        disallowed_missing = sorted(missing_keys - allowed_missing)
        if disallowed_missing or unexpected_keys:
            raise RuntimeError(
                "Unexpected incompatibilities during fallback checkpoint load. "
                f"missing={disallowed_missing}, unexpected={sorted(unexpected_keys)}"
            )

        for key, ckpt_shape, current_shape in dropped_keys:
            logger.warning(
                "Skipped loading %s due to shape mismatch (ckpt=%s, current=%s).",
                key,
                ckpt_shape,
                current_shape,
            )
        return model


def _resolve_non_holdout_calib_dataset_paths(cfg_dict: dict, holdout_cell_line: str) -> dict[str, str]:
    toml_path = cfg_dict["data"]["kwargs"].get("toml_config_path")
    if not toml_path:
        return {}
    parsed = _parse_simple_cell_load_toml(toml_path)
    datasets = parsed.get("datasets", {})
    result: dict[str, str] = {}
    for dataset_name, dataset_path in datasets.items():
        if not str(dataset_name).endswith("_calib"):
            continue
        cell_line = str(dataset_name).rsplit("_", 1)[0]
        if cell_line == holdout_cell_line:
            continue
        result[cell_line] = str(dataset_path)
    return result


def _resolve_split_csv_for_cell_line(reference_split_csv_path: str, cell_line: str, holdout_cell_line: str) -> str:
    if cell_line == holdout_cell_line:
        return reference_split_csv_path
    split_dir = Path(reference_split_csv_path).resolve().parent
    candidate = split_dir / f"DE_DOC_{cell_line}_splitted.csv"
    if not candidate.exists():
        raise FileNotFoundError(
            f"Could not find split CSV for non-holdout cell line {cell_line!r}: {candidate}"
        )
    return str(candidate)


def _load_allowed_perts_for_groups(
    split_csv_path: str,
    split_column: str | None,
    groups: list[str],
    control_pert: str,
    exclude_perts: set[str] | None = None,
) -> set[str]:
    split_df = pd.read_csv(split_csv_path)
    if "pert" not in split_df.columns:
        raise ValueError(f"Split CSV must contain a 'pert' column for prediction filtering: {split_csv_path}")
    split_col = _resolve_split_col(split_df, split_column)
    split_series = split_df[split_col].astype(str).str.lower()
    allowed = set(split_df.loc[split_series.isin(groups), "pert"].astype(str).tolist())
    if exclude_perts:
        allowed -= {str(pert) for pert in exclude_perts}
    allowed.add(str(control_pert))
    return allowed


def _build_allowed_perts_by_cell_line(
    data_kwargs: dict,
    observed_cell_lines: set[str],
    holdout_cell_line: str,
) -> dict[str, set[str]]:
    test_groups, calib_groups = _resolve_branch2_groups(data_kwargs)
    if not test_groups and not calib_groups:
        return {}
    split_csv_path = data_kwargs.get("split_csv_path")
    split_column = data_kwargs.get("split_column")
    control_pert = str(data_kwargs.get("control_pert", "non-targeting"))
    dropped_by_cell = _load_dropped_test_perts(data_kwargs.get("dropped_test_perts_json"))
    if not split_csv_path:
        return {}
    allowed_by_cell_line: dict[str, set[str]] = {}
    for cell_line in observed_cell_lines:
        groups = test_groups if cell_line == holdout_cell_line else calib_groups
        if not groups:
            continue
        cell_line_split_csv = _resolve_split_csv_for_cell_line(split_csv_path, cell_line, holdout_cell_line)
        allowed_by_cell_line[cell_line] = _load_allowed_perts_for_groups(
            split_csv_path=cell_line_split_csv,
            split_column=split_column,
            groups=groups,
            control_pert=control_pert,
            exclude_perts=dropped_by_cell.get(str(cell_line).lower(), set()) if cell_line == holdout_cell_line else None,
        )
    return allowed_by_cell_line


def _filter_cross_prediction_by_split(adata, cfg_dict: dict):
    if adata is None:
        return None
    data_kwargs = cfg_dict["data"]["kwargs"]
    split_csv_path = data_kwargs.get("split_csv_path")
    holdout_cell_line = str(cfg_dict["model"]["kwargs"].get("cell_line", cfg_dict.get("name", "")))
    test_groups, calib_groups = _resolve_branch2_groups(data_kwargs)
    if not split_csv_path or (not test_groups and not calib_groups):
        return adata

    pert_key = data_kwargs.get("pert_col", "gene")
    if pert_key not in adata.obs.columns:
        return adata
    cell_type_key = data_kwargs.get("cell_type_key", "cell_type")
    # Exported adata may not always preserve cell type metadata. In that case, avoid
    # applying a global union filter that can break holdout/non-holdout isolation.
    if cell_type_key not in adata.obs.columns:
        logger.info(
            "Skipping adata-level cross split filter because obs column %r is unavailable.",
            cell_type_key,
        )
        return adata

    obs_cell_lines = set(adata.obs[cell_type_key].astype(str).tolist())
    allowed_by_cell_line = _build_allowed_perts_by_cell_line(
        data_kwargs=data_kwargs,
        observed_cell_lines=obs_cell_lines,
        holdout_cell_line=holdout_cell_line,
    )
    perts = adata.obs[pert_key].astype(str)
    cell_lines = adata.obs[cell_type_key].astype(str)
    mask = np.array(
        [
            pert in allowed_by_cell_line.get(cell_line, set())
            for pert, cell_line in zip(perts.tolist(), cell_lines.tolist())
        ],
        dtype=bool,
    )
    return adata[mask].copy()


def apply_cross_split_filter_to_loader(test_loader, cfg_dict: dict) -> None:
    """
    Filter cross-mode test-loader sentences to the requested split groups.
    """
    data_kwargs = cfg_dict["data"]["kwargs"]
    holdout_cell_line = str(cfg_dict["model"]["kwargs"].get("cell_line", cfg_dict.get("name", "")))
    split_csv_path = data_kwargs.get("split_csv_path")
    test_groups, calib_groups = _resolve_branch2_groups(data_kwargs)
    if not split_csv_path or (not test_groups and not calib_groups):
        return

    sampler = getattr(test_loader, "batch_sampler", None)
    if sampler is None or not hasattr(sampler, "sentences") or not hasattr(sampler, "_create_batches"):
        return

    original_sentences = [list(s) for s in sampler.sentences]
    sentence_metadata = []
    observed_cell_lines = set()
    for sentence in original_sentences:
        pert_name, cell_line_name, _ = _sentence_metadata(sampler.dataset, sentence)
        sentence_metadata.append((sentence, pert_name, cell_line_name))
        observed_cell_lines.add(cell_line_name)

    allowed_by_cell_line = _build_allowed_perts_by_cell_line(
        data_kwargs=data_kwargs,
        observed_cell_lines=observed_cell_lines,
        holdout_cell_line=holdout_cell_line,
    )
    kept_sentences = []
    for sentence, pert, cell_line_name in sentence_metadata:
        if pert in allowed_by_cell_line.get(cell_line_name, set()):
            kept_sentences.append(sentence)

    if not kept_sentences:
        raise ValueError(
            "No test sentences left after split filtering with "
            f"test_groups={test_groups}, calib_groups={calib_groups}, split_csv={split_csv_path}."
        )

    sampler.sentences = kept_sentences
    sampler.batches = sampler._create_batches()
    sampler.tot_num = int(sum(len(s) for s in sampler.sentences))
    logger.info(
        "Applied cross split filter on loader: sentences %d -> %d, batches=%d.",
        len(original_sentences),
        len(kept_sentences),
        len(sampler.batches),
    )


def export_predictions_manual_h5ad(
    model,
    test_loader,
    cfg: dict,
    dataset_path: str,
    prediction_dir: str,
    sample_multiplier: float | None = None,
    prediction_use_sampling: bool | None = None,
    show_progress: bool = False,
) -> None:
    """
    Run prediction manually and write h5ad outputs.

    Parameters
    ----------
    model
        Loaded Lightning perturbation model.
    test_loader
        Dataloader containing prediction sentences.
    cfg : dict
        Run config with data/model kwargs.
    dataset_path : str
        Dataset used to recover control cells and var metadata.
    prediction_dir : str
        Directory where h5ad files are written.
    """
    sample_multiplier = validate_prediction_sample_multiplier(
        sample_multiplier if sample_multiplier is not None else cfg["model"]["kwargs"].get("prediction_sample_multiplier", 1)
    )
    logger.info("Prediction sample multiplier: %s", sample_multiplier)
    if prediction_use_sampling is None:
        prediction_use_sampling = bool(cfg["model"]["kwargs"].get("prediction_use_sampling", True))
    logger.info("Prediction use sampling: %s", prediction_use_sampling)

    pert_col = cfg["data"]["kwargs"].get("pert_col", "gene")
    control_pert = cfg["data"]["kwargs"].get("control_pert", "non-targeting")
    mode_suffix = "cross" if cfg["model"]["kwargs"].get("mode", "within") == "cross" else "within"
    cell_type_key = cfg["data"]["kwargs"].get("cell_type_key", "cell_type")
    test_groups, calib_groups = _resolve_branch2_groups(cfg["data"]["kwargs"]) if mode_suffix == "cross" else ([], [])
    split_cross_by_calib_cell_line = mode_suffix == "cross" and (not test_groups) and bool(calib_groups)
    if mode_suffix == "cross":
        if parse_bool(cfg["data"]["kwargs"].get("use_split_gene_space", False)):
            logger.info(
                "Cross prediction export is using the existing runtime dataset gene space; "
                "split CSV gene space is only used when building the runtime datasets."
            )
        gene_space = None
    else:
        gene_space = resolve_export_gene_space(cfg)
    control_adata = None
    pred_var_frame = load_var_frame_for_export(dataset_path, gene_space=gene_space) if split_cross_by_calib_cell_line else None
    if not split_cross_by_calib_cell_line:
        control_adata = load_control_adata_for_export(
            dataset_path,
            pert_col,
            control_pert,
            cell_type_key=cell_type_key if mode_suffix == "cross" else None,
            cell_type_value=str(cfg["model"]["kwargs"].get("cell_line", cfg.get("name", ""))) if mode_suffix == "cross" else None,
            gene_space=gene_space,
        )

    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    if cfg["model"]["kwargs"].get("mode", "within") == "cross":
        apply_cross_split_filter_to_loader(test_loader, cfg)
    apply_prediction_sample_multiplier_to_loader(test_loader, sample_multiplier)

    device = next(model.parameters()).device
    model_class_name = cfg["model"]["name"]
    final_preds, final_reals = [], []
    all_pert_names, all_real_pert_names = [], []
    all_cell_types, all_real_cell_types = [], []
    write_true = math.isclose(sample_multiplier, 1.0)

    logger.info("Generating predictions on test set using manual loop...")
    iterator = tqdm(test_loader, desc="Predicting", unit="batch") if show_progress else test_loader
    with torch.no_grad():
        for batch_idx, batch in enumerate(iterator):
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            autocast_enabled = torch.cuda.is_available()
            autocast_device = "cuda" if autocast_enabled else "cpu"
            with torch.autocast(device_type=autocast_device, enabled=autocast_enabled):
                batch_preds = model.predict_step(
                    batch,
                    batch_idx,
                    padded=False,
                    prediction_use_sampling=prediction_use_sampling,
                )
                if batch_preds["preds"] is None:
                    if hasattr(model, "_predict_outputs"):
                        model._predict_outputs.clear()
                    continue
                batch_preds, _ = post_process_preds(batch_preds, model_class_name)

            pred_np = batch_preds["preds"].float().cpu().numpy()
            names = _as_list(batch_preds["pert_name"], pred_np.shape[0])
            cell_types = _normalize_row_values(batch_preds.get("cell_type"), pred_np.shape[0], default_value="unknown")
            final_preds.append(pred_np)
            all_pert_names.extend(names)
            all_cell_types.extend(cell_types)
            if write_true:
                final_reals.append(batch_preds["X"].float().cpu().numpy())
                all_real_pert_names.extend(names)
                all_real_cell_types.extend(cell_types)
            if hasattr(model, "_predict_outputs"):
                model._predict_outputs.clear()

    if not final_preds:
        logger.info("No prediction batches produced outputs; skipping h5ad export.")
        return

    final_preds = np.concatenate(final_preds, axis=0)
    final_reals = np.concatenate(final_reals, axis=0) if write_true else None
    logger.info(
        "Prediction rows after multiplier %s: %d; true rows: %d",
        sample_multiplier,
        final_preds.shape[0],
        final_reals.shape[0] if final_reals is not None else 0,
    )

    adata_pred = build_vae_scgen_style_adata(
        final_preds,
        all_pert_names,
        control_adata,
        pert_col,
        cell_types=all_cell_types if mode_suffix == "cross" else None,
        cell_type_key=cell_type_key if mode_suffix == "cross" else None,
        var_frame=pred_var_frame,
    )
    adata_real = None
    if write_true:
        adata_real = build_vae_scgen_style_adata(
            final_reals,
            all_real_pert_names,
            control_adata,
            pert_col,
            cell_types=all_real_cell_types if mode_suffix == "cross" else None,
            cell_type_key=cell_type_key if mode_suffix == "cross" else None,
            var_frame=pred_var_frame,
        )

    os.makedirs(prediction_dir, exist_ok=True)
    if mode_suffix == "cross":
        adata_pred = _filter_cross_prediction_by_split(adata_pred, cfg)
        if adata_real is not None:
            adata_real = _filter_cross_prediction_by_split(adata_real, cfg)

    cell_line = cfg["model"]["kwargs"].get("cell_line", cfg["name"])
    sample_suffix = format_prediction_sample_multiplier(sample_multiplier)
    pred_suffix = f"_{'sampling' if prediction_use_sampling else 'mean'}_{sample_suffix}"

    if mode_suffix == "cross":
        if split_cross_by_calib_cell_line:
            holdout_cell_line = str(cfg["model"]["kwargs"].get("cell_line", cfg.get("name", "")))
            if cell_type_key not in adata_pred.obs.columns:
                raise ValueError(
                    f"Cannot split cross prediction by cell line because obs column {cell_type_key!r} is missing."
                )
            calib_dataset_paths = _resolve_non_holdout_calib_dataset_paths(cfg, holdout_cell_line)
            if not calib_dataset_paths:
                raise ValueError(
                    "Expected non-holdout *_calib datasets for calibration-only cross prediction, but none were found."
                )

            saved_count = 0
            for non_holdout_cell_line in sorted(calib_dataset_paths):
                line_mask = adata_pred.obs[cell_type_key].astype(str) == str(non_holdout_cell_line)
                line_pred = adata_pred[line_mask].copy()
                if line_pred.n_obs == 0:
                    logger.info(
                        "Skipping empty calibration prediction for non-holdout cell line %s.",
                        non_holdout_cell_line,
                    )
                    continue

                line_control = load_control_adata_for_export(
                    dataset_path=calib_dataset_paths[non_holdout_cell_line],
                    pert_col=pert_col,
                    control_pert=control_pert,
                    cell_type_key=cell_type_key,
                    cell_type_value=non_holdout_cell_line,
                    gene_space=gene_space,
                )
                line_out_adata = anndata.concat(
                    [line_pred, line_control],
                    axis=0,
                    join="inner",
                    merge="same",
                    index_unique=None,
                )
                out = os.path.join(
                    prediction_dir,
                    f"{holdout_cell_line}_{non_holdout_cell_line}_calibration{pred_suffix}.h5ad",
                )
                line_out_adata.write_h5ad(out)
                logger.info("Saved split calibration prediction to %s", out)
                saved_count += 1

            if saved_count == 0:
                raise ValueError(
                    "Calibration-only cross prediction produced no non-holdout files after filtering."
                )
            return

        out = os.path.join(prediction_dir, f"{cell_line}_cross{pred_suffix}.h5ad")
        adata_pred.write_h5ad(out)
        logger.info("Saved adata_pred to %s", out)
    else:
        if adata_real is not None:
            true_out = os.path.join(prediction_dir, f"{cell_line}_within_true{pred_suffix}.h5ad")
            adata_real.write_h5ad(true_out)
            logger.info("Saved adata_real to %s", true_out)
        else:
            logger.info("Skipping within_true export for sample multiplier %s.", sample_multiplier)
        out = os.path.join(prediction_dir, f"{cell_line}_within{pred_suffix}.h5ad")
        adata_pred.write_h5ad(out)
        logger.info("Saved adata_pred to %s", out)


def main():
    """
    Load a trained run, restore the model, and export predictions.
    """
    args = parse_args()
    config_path = os.path.join(args.output_dir, "config.yaml")
    cfg = load_config(config_path)
    logger.info("Loaded config from %s", config_path)

    run_output_dir = args.output_dir
    cell_sentence_len = cfg["model"]["kwargs"].get("cell_sentence_len", 1)
    if cfg["data"]["kwargs"]["pert_col"] == "drugname_drugconc":
        cfg["data"]["kwargs"]["control_pert"] = "[('DMSO_TF', 0.0, 'uM')]"
    cfg["data"]["kwargs"].setdefault("train_valid_seed", cfg["training"].get("train_seed", 42))

    mode = cfg["model"]["kwargs"].get("mode", "within")
    if mode == "cross":
        if args.split_csv_path is not None:
            cfg["data"]["kwargs"]["split_csv_path"] = args.split_csv_path
        if args.split_column is not None:
            cfg["data"]["kwargs"]["split_column"] = args.split_column
        if args.use_split_gene_space is not None:
            cfg["data"]["kwargs"]["use_split_gene_space"] = parse_bool(args.use_split_gene_space)
        if args.dropped_test_perts_json is not None:
            cfg["data"]["kwargs"]["dropped_test_perts_json"] = args.dropped_test_perts_json
        # In cross prediction mode we always reuse the existing runtime/toml.
        # Groups are driven strictly by CLI input; omitted groups are disabled.
        cfg["data"]["kwargs"]["test_split_groups"] = args.test_split_groups
        cfg["data"]["kwargs"]["calib_split_groups"] = args.calib_split_groups
        prepare_cross_prediction_toml(cfg, output_dir=run_output_dir, toml_name=args.toml_name)
        logger.info(
            "Cross prediction: reusing existing runtime/toml; applying split-based filtering on test loader before prediction."
        )
    else:
        if args.split_csv_path is not None:
            cfg["data"]["kwargs"]["split_csv_path"] = args.split_csv_path
        if args.split_column is not None:
            cfg["data"]["kwargs"]["split_column"] = args.split_column
        if args.use_split_gene_space is not None:
            cfg["data"]["kwargs"]["use_split_gene_space"] = parse_bool(args.use_split_gene_space)
        if args.train_split_groups is not None:
            cfg["data"]["kwargs"]["train_split_groups"] = args.train_split_groups
        if args.test_split_groups is not None:
            cfg["data"]["kwargs"]["test_split_groups"] = args.test_split_groups
        if args.calib_split_groups is not None:
            cfg["data"]["kwargs"]["calib_split_groups"] = args.calib_split_groups
        if args.dropped_test_perts_json is not None:
            cfg["data"]["kwargs"]["dropped_test_perts_json"] = args.dropped_test_perts_json
        logger.info("Prediction: reusing existing runtime/toml for non-cross mode (no rebuild).")

    data_module = get_datamodule(
        name=cfg["data"]["name"],
        kwargs=cfg["data"]["kwargs"],
        batch_size=cfg["training"]["batch_size"],
        cell_sentence_len=cell_sentence_len,
    )
    data_module.setup()

    pl.seed_everything(cfg["training"]["train_seed"])
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    checkpoint_path = resolve_checkpoint_path(checkpoint_dir, args.checkpoint)
    logger.info("Loading model from %s", checkpoint_path)

    if cfg["model"]["name"].lower() in ["cpa", "scvi", "lowranklinear", "gears"] or cfg["model"]["name"].lower().startswith("scgpt"):
        cfg["model"]["kwargs"]["n_cell_types"] = len(data_module.cell_type_onehot_map)
        cfg["model"]["kwargs"]["n_perts"] = len(data_module.pert_onehot_map)
        cfg["model"]["kwargs"]["n_batches"] = len(data_module.batch_onehot_map)

    if cfg["model"]["name"].lower() == "scvi":
        from scVI_src.models.scvi import SCVIPerturbationModel

        ModelClass = SCVIPerturbationModel
    else:
        raise ValueError(f"Only scvi is supported in this trimmed baseline, got: {cfg['model']['name']}")

    var_dims = data_module.get_var_dims()
    model_kwargs = cfg["model"]["kwargs"]
    model_init_kwargs = {
        "input_dim": var_dims["input_dim"],
        "gene_dim": var_dims["gene_dim"],
        "hvg_dim": var_dims["hvg_dim"],
        "output_dim": var_dims["output_dim"],
        "pert_dim": var_dims["pert_dim"],
        "batch_dim": var_dims.get("batch_dim"),
        "pert_names": var_dims.get("pert_names"),
        **model_kwargs,
    }

    model = load_model_with_pert_embedding_fallback(
        ModelClass=ModelClass,
        checkpoint_path=checkpoint_path,
        model_init_kwargs=model_init_kwargs,
    )
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    logger.info("Model loaded successfully.")

    data_module.cell_sentence_len = cfg["model"]["kwargs"].get("cell_sentence_len", 1)
    test_loader = data_module.test_dataloader()
    print(f"DEBUG: data_module.batch_size: {data_module.batch_size}")
    if test_loader is None:
        logger.warning("No test dataloader found. Exiting.")
        sys.exit(0)

    dataset_path = resolve_dataset_h5ad_path(cfg["data"]["kwargs"]["dataset_dir"], cfg["data"]["kwargs"]["dataset_name"])
    sample_multiplier = (
        args.prediction_sample_multiplier
        if args.prediction_sample_multiplier is not None
        else cfg["model"]["kwargs"].get("prediction_sample_multiplier", 1)
    )
    prediction_use_sampling = (
        parse_bool(args.prediction_use_sampling)
        if args.prediction_use_sampling is not None
        else bool(cfg["model"]["kwargs"].get("prediction_use_sampling", True))
    )

    export_predictions_manual_h5ad(
        model=model,
        test_loader=test_loader,
        cfg=cfg,
        dataset_path=dataset_path,
        prediction_dir=os.path.join(args.output_dir, "predictions"),
        sample_multiplier=sample_multiplier,
        prediction_use_sampling=prediction_use_sampling,
        show_progress=True,
    )


if __name__ == "__main__":
    main()
