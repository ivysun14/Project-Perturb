import logging
import json
import os
import pickle
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import anndata as ad
import h5py
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _parse_split_groups(value, name: str) -> list[str]:
    if value is None:
        raise ValueError(f"{name} must be set explicitly.")
    elif isinstance(value, str):
        groups = [part.strip() for part in value.replace(",", " ").split()]
    else:
        groups = [str(part).strip() for part in value]

    groups = [group.lower() for group in groups if group]
    if not groups:
        raise ValueError(f"{name} must contain at least one split group.")
    return list(dict.fromkeys(groups))


def _parse_optional_split_groups(value, name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str) and not value.strip():
        return []
    if not isinstance(value, str):
        try:
            if len(value) == 0:
                return []
        except TypeError:
            pass
    return _parse_split_groups(value, name)


def _load_dropped_test_perts(path: Optional[str]) -> dict[str, set[str]]:
    if path is None or not str(path).strip():
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dropped test perts JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {str(k).lower(): {str(v) for v in values} for k, values in raw.items()}


def resolve_dataset_h5ad_path(dataset_dir: str, dataset_name: str) -> str:
    """
    Resolve the dataset file as <dataset_dir>/<dataset_name>.h5ad.
    """
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    dataset_path = os.path.join(dataset_dir, f"{dataset_name}.h5ad")
    if not os.path.exists(dataset_path):
        available = sorted(
            filename for filename in os.listdir(dataset_dir) if filename.endswith(".h5ad")
        )
        raise FileNotFoundError(
            f"Expected dataset file {dataset_path} not found. "
            f"Available .h5ad files: {available}"
        )
    return dataset_path


def _decode_h5_values(values) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype.kind == "S":
        return np.char.decode(arr, "utf-8")
    if arr.dtype.kind == "O":
        return np.array(
            [value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else str(value) for value in arr],
            dtype=object,
        )
    return arr.astype(str)


def _list_raw_h5ad_datasets(dataset_dir: str) -> Dict[str, str]:
    root = Path(dataset_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    datasets = {}
    for path in sorted(root.glob("*_raw.h5ad")):
        cell_line = path.stem.removesuffix("_raw")
        datasets[cell_line] = str(path)

    if not datasets:
        raise FileNotFoundError(f"No *_raw.h5ad datasets found under {dataset_dir}")
    return datasets


def _resolve_cross_source_dataset_dir(dataset_dir: str, split_csv_path: Optional[str]) -> str:
    candidates = [dataset_dir]
    dataset_path = Path(dataset_dir)
    if dataset_path.name == ".cross_cell_load_runtime":
        candidates.append(str(dataset_path.parent))

    if split_csv_path:
        split_path = Path(split_csv_path).resolve()
        project_root = split_path.parent.parent
        candidates.append(str(project_root / "h5ad_objects"))

    checked = []
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        checked.append(candidate)
        try:
            _list_raw_h5ad_datasets(candidate)
            return candidate
        except FileNotFoundError:
            continue

    raise FileNotFoundError(
        "Cross mode requires access to raw datasets (*_raw.h5ad), but none were found. "
        f"Tried: {checked}"
    )


def _read_var_names(source_path: str) -> list[str]:
    adata = ad.read_h5ad(source_path, backed="r")
    try:
        return [str(name) for name in adata.var_names.tolist()]
    finally:
        file_manager = getattr(adata, "file", None)
        if file_manager is not None:
            file_manager.close()


def _read_split_genes(csv_path: str, gene_col: str = "gene") -> list[str]:
    split_df = pd.read_csv(csv_path, usecols=[gene_col])
    genes = [str(gene).strip() for gene in split_df[gene_col].dropna().tolist()]
    genes = [gene for gene in genes if gene]
    if not genes:
        raise ValueError(f"No genes found in split CSV column {gene_col!r}: {csv_path}")
    return list(dict.fromkeys(genes))


def _subset_adata_gene_space(adata: ad.AnnData, genes: list[str]) -> ad.AnnData:
    genes = [str(gene) for gene in genes]
    var_names = [str(name) for name in adata.var_names.tolist()]
    var_name_set = set(var_names)
    if all(gene in var_name_set for gene in genes):
        subset = adata[:, genes].copy()
        subset.var_names = pd.Index(genes)
        return subset

    if "gene_name" not in adata.var.columns:
        missing = sorted(set(genes) - var_name_set)[:10]
        raise ValueError(
            "Requested split gene space does not match var_names, and adata.var['gene_name'] is missing. "
            f"Example missing genes: {missing}"
        )

    gene_to_var = {}
    for var_name, gene_name in zip(adata.var_names.tolist(), adata.var["gene_name"].astype(str).tolist()):
        gene_to_var.setdefault(str(gene_name), var_name)

    missing = [gene for gene in genes if gene not in gene_to_var]
    if missing:
        raise ValueError(
            f"{len(missing)} split genes were not found in adata.var['gene_name']; "
            f"examples={missing[:10]}"
        )

    selected_var_names = [gene_to_var[gene] for gene in genes]
    subset = adata[:, selected_var_names].copy()
    subset.var_names = pd.Index(genes)
    subset.var["gene_name"] = genes
    return subset


def _write_runtime_h5ad_subset(
    source_path: str,
    runtime_path: str,
    common_genes: list[str],
    cell_line: str,
    pert_col: Optional[str],
    batch_col: Optional[str],
    cell_type_key: Optional[str],
    control_pert: str,
    allowed_perts: Optional[Iterable[str]] = None,
    include_controls: bool = True,
) -> Dict[str, int]:
    adata = ad.read_h5ad(source_path)
    try:
        subset = _subset_adata_gene_space(adata, common_genes)

        if pert_col:
            if pert_col not in subset.obs.columns:
                raise KeyError(f"Missing required obs column {pert_col!r} in {source_path}")
            pert_series = subset.obs[pert_col].astype(str).str.strip()
            if allowed_perts is None:
                mask = np.ones(subset.n_obs, dtype=bool)
            else:
                mask = np.zeros(subset.n_obs, dtype=bool)
                allowed = {str(pert) for pert in allowed_perts}
                mask |= pert_series.isin(sorted(allowed)).to_numpy()
            if include_controls:
                mask |= (pert_series == str(control_pert)).to_numpy()
            subset = subset[mask].copy()

        for column in (pert_col, batch_col):
            if not column:
                continue
            if column not in subset.obs.columns:
                raise KeyError(f"Missing required obs column {column!r} in {source_path}")
            subset.obs[column] = pd.Categorical(subset.obs[column].astype(str).str.strip())

        if cell_type_key:
            subset.obs[cell_type_key] = pd.Categorical(
                np.full(subset.n_obs, cell_line, dtype=object)
            )

        subset.write_h5ad(runtime_path)

        ctrl_count = 0
        pert_count = subset.n_obs
        if pert_col and pert_col in subset.obs.columns:
            pert_series = subset.obs[pert_col].astype(str)
            ctrl_mask = pert_series == str(control_pert)
            ctrl_count = int(ctrl_mask.sum())
            pert_count = int((~ctrl_mask).sum())

        return {
            "cells": int(subset.n_obs),
            "genes": int(subset.n_vars),
            "controls": ctrl_count,
            "perturbed": pert_count,
        }
    finally:
        del adata


def _write_dataset_entry(handle, dataset_name: str, dataset_path: str) -> None:
    handle.write(f'{dataset_name} = "{os.path.abspath(dataset_path)}"\n')


def _resolve_cross_split_csv_paths(reference_split_csv: str, expected_cell_lines: Iterable[str]) -> Dict[str, str]:
    reference_path = Path(reference_split_csv).resolve()
    split_dir = reference_path.parent
    reference_name = reference_path.name
    expected = [str(cell_line) for cell_line in expected_cell_lines]

    # Infer a filename template by locating one known cell line token in the
    # provided reference CSV file name, then replacing that token for each cell line.
    token_matches = []
    for cell_line in expected:
        for match in re.finditer(re.escape(cell_line), reference_name, flags=re.IGNORECASE):
            token_matches.append((-(match.end() - match.start()), match.start(), match.end(), match.group(0)))

    if not token_matches:
        raise FileNotFoundError(
            "Could not infer cross split CSV filename template from reference CSV "
            f"{reference_split_csv}. Expected filename to contain one of {sorted(expected)}."
        )

    token_matches.sort()
    _, start, end, matched_token = token_matches[0]
    prefix = reference_name[:start]
    suffix = reference_name[end:]

    if matched_token.islower():
        render_cell_line = lambda name: name.lower()
    elif matched_token.isupper():
        render_cell_line = lambda name: name.upper()
    else:
        render_cell_line = lambda name: name

    result = {}
    for cell_line in expected:
        variants = [
            render_cell_line(cell_line),
            cell_line,
            cell_line.lower(),
            cell_line.upper(),
        ]
        candidate = None
        for variant in dict.fromkeys(variants):
            probe = split_dir / f"{prefix}{variant}{suffix}"
            if probe.exists():
                candidate = probe
                break

        if candidate is None:
            raise FileNotFoundError(
                "Expected split CSV for "
                f"{cell_line} matching template {prefix}<cell_line>{suffix} in {split_dir}, "
                f"derived from {reference_split_csv}."
            )
        result[cell_line] = str(candidate)
    return result


def _load_split_lists(
    csv_path: str,
    split_column: Optional[str],
    train_values,
    calibration_values,
    control_pert: str,
) -> Tuple[list[str], list[str]]:
    split_df = pd.read_csv(csv_path)
    candidate_split_columns = []
    for candidate in (split_column, "split", "split_calib"):
        if candidate and candidate not in candidate_split_columns:
            candidate_split_columns.append(candidate)
    split_col = next((col for col in candidate_split_columns if col in split_df.columns), None)
    if split_col is None:
        raise ValueError(
            "Split CSV must contain a split column. "
            f"Tried {candidate_split_columns}, available columns are {sorted(split_df.columns.tolist())}."
        )

    if "pert" not in split_df.columns:
        raise ValueError(f"Split CSV must contain a 'pert' column: {csv_path}")

    split_df = split_df.copy()
    split_df["pert"] = split_df["pert"].astype(str)
    split_series = split_df[split_col].astype(str).str.lower()
    train_values = _parse_split_groups(train_values, "train_split_groups")
    calibration_values = _parse_optional_split_groups(calibration_values, "calib_split_groups")

    pert_split_counts = split_df.groupby("pert")[split_col].nunique()
    bad_perts = pert_split_counts[pert_split_counts > 1]
    if not bad_perts.empty:
        raise ValueError(
            "Each perturbation must map to exactly one split. "
            f"Found conflicting split assignments for: {bad_perts.index.tolist()[:10]}"
        )

    calibration_perts = (
        sorted(
            {
                pert
                for pert in split_df.loc[split_series.isin(calibration_values), "pert"].dropna().tolist()
                if pert != control_pert
            }
        )
        if calibration_values
        else []
    )
    train_pool_perts = sorted(
        {
            pert
            for pert in split_df.loc[split_series.isin(train_values), "pert"].dropna().tolist()
            if pert != control_pert
        }
    )

    if not train_pool_perts:
        raise ValueError(
            f"No train perturbations found in {csv_path} using column {split_col} "
            f"and groups {train_values}."
        )

    return train_pool_perts, calibration_perts


def _load_prediction_perts(
    csv_path: str,
    split_column: Optional[str],
    prediction_values,
    control_pert: str,
    exclude_perts: Optional[Iterable[str]] = None,
) -> list[str]:
    split_df = pd.read_csv(csv_path)
    candidate_split_columns = []
    for candidate in (split_column, "split", "split_calib"):
        if candidate and candidate not in candidate_split_columns:
            candidate_split_columns.append(candidate)
    split_col = next((col for col in candidate_split_columns if col in split_df.columns), None)
    if split_col is None:
        raise ValueError(
            "Split CSV must contain a split column. "
            f"Tried {candidate_split_columns}, available columns are {sorted(split_df.columns.tolist())}."
        )

    if "pert" not in split_df.columns:
        raise ValueError(f"Split CSV must contain a 'pert' column: {csv_path}")

    split_df = split_df.copy()
    split_df["pert"] = split_df["pert"].astype(str)
    split_series = split_df[split_col].astype(str).str.lower()
    prediction_values = _parse_optional_split_groups(prediction_values, "test_split_groups")

    pert_split_counts = split_df.groupby("pert")[split_col].nunique()
    bad_perts = pert_split_counts[pert_split_counts > 1]
    if not bad_perts.empty:
        raise ValueError(
            "Each perturbation must map to exactly one split. "
            f"Found conflicting split assignments for: {bad_perts.index.tolist()[:10]}"
        )

    prediction_perts = sorted(
        {
            pert
            for pert in split_df.loc[split_series.isin(prediction_values), "pert"].dropna().tolist()
            if pert != control_pert
        }
    )
    if exclude_perts:
        excluded = {str(pert) for pert in exclude_perts}
        prediction_perts = [pert for pert in prediction_perts if pert not in excluded]
    if prediction_values and not prediction_perts:
        raise ValueError(
            f"No prediction perturbations found in {csv_path} using column {split_col} "
            f"and groups {prediction_values}."
        )
    return prediction_perts


def build_cross_cell_load_config(
    dataset_dir: str,
    cross_cell_line: str,
    reference_split_csv: str,
    output_dir: str,
    pert_col: str,
    batch_col: Optional[str],
    cell_type_key: Optional[str],
    control_pert: str,
    split_column: Optional[str] = None,
    train_split_groups=None,
    test_split_groups=None,
    calibration_split_groups=None,
    train_valid_ratio: float = 0.05,
    train_valid_seed: int = 42,
    use_split_gene_space: bool = False,
    split_gene_col: str = "gene",
    dropped_test_perts_json: Optional[str] = None,
) -> Dict[str, str]:
    raw_datasets = _list_raw_h5ad_datasets(dataset_dir)
    if cross_cell_line not in raw_datasets:
        raise ValueError(
            f"Cross cell line {cross_cell_line!r} not found under {dataset_dir}. "
            f"Available: {sorted(raw_datasets)}"
        )
    if len(raw_datasets) < 2:
        raise ValueError("Cross mode requires at least two cell line datasets.")

    ordered_cell_lines = sorted(raw_datasets)
    split_csv_paths = _resolve_cross_split_csv_paths(reference_split_csv, ordered_cell_lines)

    use_split_gene_space = _parse_bool(use_split_gene_space)
    if use_split_gene_space:
        common_genes = _read_split_genes(split_csv_paths[ordered_cell_lines[0]], split_gene_col)
        gene_counts = {ordered_cell_lines[0]: len(common_genes)}
        for cell_line in ordered_cell_lines[1:]:
            current_genes = _read_split_genes(split_csv_paths[cell_line], split_gene_col)
            gene_counts[cell_line] = len(current_genes)
            current_gene_set = set(current_genes)
            common_genes = [gene for gene in common_genes if gene in current_gene_set]
    else:
        common_genes = _read_var_names(raw_datasets[ordered_cell_lines[0]])
        gene_counts = {ordered_cell_lines[0]: len(common_genes)}
        for cell_line in ordered_cell_lines[1:]:
            current_genes = _read_var_names(raw_datasets[cell_line])
            gene_counts[cell_line] = len(current_genes)
            current_gene_set = set(current_genes)
            common_genes = [gene for gene in common_genes if gene in current_gene_set]

    if not common_genes:
        raise ValueError("No common genes found across cross-mode cell line datasets.")

    runtime_dir = os.path.join(output_dir, ".cross_cell_load_runtime")
    os.makedirs(runtime_dir, exist_ok=True)

    dataset_paths = {}
    dataset_summaries = {}
    train_dataset_names = []
    zeroshot_entries = []
    export_dataset_path = None
    prediction_requested = False
    split_summaries = {}
    if not (0.0 < train_valid_ratio < 1.0):
        raise ValueError(f"train_valid_ratio must be between 0 and 1, got {train_valid_ratio}.")

    train_split_groups = _parse_split_groups(train_split_groups, "train_split_groups")
    test_split_groups = _parse_optional_split_groups(test_split_groups, "test_split_groups")
    calibration_split_groups = _parse_optional_split_groups(calibration_split_groups, "calib_split_groups")
    dropped_by_cell = _load_dropped_test_perts(dropped_test_perts_json)
    unknown_prediction_groups = set(test_split_groups) - ({"test"} | set(calibration_split_groups))
    if unknown_prediction_groups:
        raise ValueError(
            "Cross mode prediction groups must be 'test' for the cross cell line "
            f"or one of calibration groups {calibration_split_groups}; got {sorted(unknown_prediction_groups)}."
        )
    predict_cross_test = "test" in test_split_groups
    predict_calibration = bool(set(test_split_groups) & set(calibration_split_groups))
    predict_non_cross_calibration = predict_calibration and not predict_cross_test
    prediction_requested = bool(test_split_groups)

    rng = np.random.default_rng(train_valid_seed)
    for cell_line in ordered_cell_lines:
        if cell_line == cross_cell_line:
            holdout_prediction_groups = []
            if "test" in test_split_groups:
                holdout_prediction_groups.append("test")
            holdout_prediction_groups.extend(
                [group for group in calibration_split_groups if group in test_split_groups]
            )
            holdout_allowed_perts = (
                _load_prediction_perts(
                    csv_path=split_csv_paths[cell_line],
                    split_column=split_column,
                    prediction_values=holdout_prediction_groups,
                    control_pert=control_pert,
                    exclude_perts=dropped_by_cell.get(cell_line.lower(), set()),
                )
                if holdout_prediction_groups
                else None
            )
            test_dataset_name = f"{cell_line}_test"
            test_runtime_path = os.path.join(runtime_dir, f"{test_dataset_name}.h5ad")
            if os.path.exists(test_runtime_path):
                os.remove(test_runtime_path)
            dataset_summaries[test_dataset_name] = _write_runtime_h5ad_subset(
                source_path=raw_datasets[cell_line],
                runtime_path=test_runtime_path,
                common_genes=common_genes,
                cell_line=cell_line,
                pert_col=pert_col,
                batch_col=batch_col,
                cell_type_key=cell_type_key,
                control_pert=control_pert,
                allowed_perts=holdout_allowed_perts,
                include_controls=True,
            )
            dataset_paths[test_dataset_name] = test_runtime_path
            if predict_cross_test:
                zeroshot_entries.append((test_dataset_name, cell_line, "test"))
                export_dataset_path = test_runtime_path
            continue

        train_pool_perts, calibration_perts = _load_split_lists(
            csv_path=split_csv_paths[cell_line],
            split_column=split_column,
            train_values=train_split_groups,
            calibration_values=calibration_split_groups,
            control_pert=control_pert,
        )
        shuffled_train_perts = list(train_pool_perts)
        rng.shuffle(shuffled_train_perts)

        n_val = int(round(len(shuffled_train_perts) * train_valid_ratio))
        n_val = max(1, min(n_val, len(shuffled_train_perts) - 1))
        val_perts = sorted(shuffled_train_perts[:n_val])
        final_train_perts = sorted(shuffled_train_perts[n_val:])

        if not final_train_perts:
            raise ValueError(
                f"Validation split consumed all train perturbations for {cell_line}; "
                "adjust train_valid_ratio."
            )

        split_summaries[cell_line] = {
            "train_pool": len(train_pool_perts),
            "derived_train": len(final_train_perts),
            "derived_val": len(val_perts),
            "calibration": len(calibration_perts),
            "train_groups": train_split_groups,
            "calibration_groups": calibration_split_groups,
        }

        train_dataset_name = f"{cell_line}_train"
        train_runtime_path = os.path.join(runtime_dir, f"{train_dataset_name}.h5ad")
        if os.path.exists(train_runtime_path):
            os.remove(train_runtime_path)
        dataset_summaries[train_dataset_name] = _write_runtime_h5ad_subset(
            source_path=raw_datasets[cell_line],
            runtime_path=train_runtime_path,
            common_genes=common_genes,
            cell_line=cell_line,
            pert_col=pert_col,
            batch_col=batch_col,
            cell_type_key=cell_type_key,
            control_pert=control_pert,
            allowed_perts=final_train_perts,
            include_controls=True,
        )
        dataset_paths[train_dataset_name] = train_runtime_path
        train_dataset_names.append(train_dataset_name)

        val_dataset_name = f"{cell_line}_val"
        val_runtime_path = os.path.join(runtime_dir, f"{val_dataset_name}.h5ad")
        if os.path.exists(val_runtime_path):
            os.remove(val_runtime_path)
        dataset_summaries[val_dataset_name] = _write_runtime_h5ad_subset(
            source_path=raw_datasets[cell_line],
            runtime_path=val_runtime_path,
            common_genes=common_genes,
            cell_line=cell_line,
            pert_col=pert_col,
            batch_col=batch_col,
            cell_type_key=cell_type_key,
            control_pert=control_pert,
            allowed_perts=val_perts,
            include_controls=True,
        )
        dataset_paths[val_dataset_name] = val_runtime_path
        zeroshot_entries.append((val_dataset_name, cell_line, "val"))

        if calibration_split_groups:
            calib_dataset_name = f"{cell_line}_calib"
            calib_runtime_path = os.path.join(runtime_dir, f"{calib_dataset_name}.h5ad")
            if os.path.exists(calib_runtime_path):
                os.remove(calib_runtime_path)
            dataset_summaries[calib_dataset_name] = _write_runtime_h5ad_subset(
                source_path=raw_datasets[cell_line],
                runtime_path=calib_runtime_path,
                common_genes=common_genes,
                cell_line=cell_line,
                pert_col=pert_col,
                batch_col=batch_col,
                cell_type_key=cell_type_key,
                control_pert=control_pert,
                allowed_perts=calibration_perts,
                include_controls=True,
            )
            dataset_paths[calib_dataset_name] = calib_runtime_path
        if predict_non_cross_calibration:
            zeroshot_entries.append((calib_dataset_name, cell_line, "test"))
            if export_dataset_path is None:
                export_dataset_path = calib_runtime_path

    generated_toml_path = os.path.join(output_dir, "cell_load_cross_split.toml")
    os.makedirs(os.path.dirname(generated_toml_path), exist_ok=True)

    with open(generated_toml_path, "w", encoding="utf-8") as handle:
        handle.write("# Auto-generated cross-cell-line Cell Load config.\n")
        handle.write("[datasets]\n")
        for dataset_name, dataset_path in dataset_paths.items():
            _write_dataset_entry(handle, dataset_name, dataset_path)
        handle.write("\n")

        handle.write("[training]\n")
        for dataset_name in train_dataset_names:
            handle.write(f'{dataset_name} = "train"\n')
        handle.write("\n")

        handle.write("[zeroshot]\n")
        for dataset_name, cell_line, split in zeroshot_entries:
            handle.write(f'"{dataset_name}.{cell_line}" = "{split}"\n')
        handle.write("\n")

    logger.info(
        "Cross mode common-gene summary: %d cell lines, %d common genes. Per-dataset genes before intersection: %s",
        len(ordered_cell_lines),
        len(common_genes),
        ", ".join(f"{cell_line}={gene_counts[cell_line]}" for cell_line in ordered_cell_lines),
    )
    for dataset_name, summary in dataset_summaries.items():
        logger.info(
            "Cross dataset %s: cells=%d, common_genes=%d, controls=%d, perturbed=%d, runtime=%s",
            dataset_name,
            summary["cells"],
            summary["genes"],
            summary["controls"],
            summary["perturbed"],
            dataset_paths[dataset_name],
        )

    logger.info(
        (
            "Cross split strategy: cross=%s test uses cross-cell-line groups %s; "
            "non-cross train perts are re-split into train/val with seed=%d ratio=%.4f; "
            "prediction groups=%s."
        ),
        cross_cell_line,
        ",".join(test_split_groups),
        train_valid_seed,
        train_valid_ratio,
        ",".join(test_split_groups),
    )
    for cell_line in ordered_cell_lines:
        if cell_line == cross_cell_line:
            continue
        split_summary = split_summaries[cell_line]
        logger.info(
            (
                "Cross split %s: non-calibration pool=%d, derived train=%d, derived val=%d, "
                "calibration=%d via %s train_groups=%s calibration_groups=%s"
            ),
            cell_line,
            split_summary["train_pool"],
            split_summary["derived_train"],
            split_summary["derived_val"],
            split_summary["calibration"],
            split_column or "auto",
            ",".join(split_summary["train_groups"]),
            ",".join(split_summary["calibration_groups"]),
        )

    if export_dataset_path is None and not prediction_requested and train_dataset_names:
        export_dataset_path = dataset_paths[train_dataset_names[0]]

    if export_dataset_path is None:
        raise ValueError(
            f"Failed to determine export dataset for cross prediction groups {test_split_groups}."
        )

    export_runtime = Path(export_dataset_path)
    return {
        "dataset_dir": str(export_runtime.parent),
        "dataset_name": export_runtime.stem,
        "toml_config_path": generated_toml_path,
    }


def ensure_cell_load_runtime_copy(
    dataset_dir: str,
    dataset_name: str,
    output_dir: str,
    cell_type_name: str,
    pert_col: Optional[str],
    batch_col: Optional[str],
    cell_type_key: Optional[str],
) -> Tuple[str, str]:
    """
    Create a run-local low-memory compatibility copy for cell_load.
    """
    source_path = resolve_dataset_h5ad_path(dataset_dir, dataset_name)
    runtime_dir = os.path.join(output_dir, ".cell_load_runtime")
    os.makedirs(runtime_dir, exist_ok=True)
    runtime_name = f"{dataset_name}__cellload"
    runtime_path = os.path.join(runtime_dir, f"{runtime_name}.h5ad")

    with h5py.File(source_path, "r") as src:
        obs_group = src["obs"]

        required_columns = [column for column in (pert_col, batch_col) if column]
        missing_required = [column for column in required_columns if column not in obs_group]
        if missing_required:
            raise KeyError(f"Missing required obs columns for cell_load: {missing_required}")

        needs_copy = False
        changed_columns = []

        if cell_type_key and cell_type_key not in obs_group:
            needs_copy = True
            changed_columns.append(cell_type_key)

        for column in (pert_col, batch_col, cell_type_key):
            if not column or column not in obs_group:
                continue
            if not isinstance(obs_group[column], h5py.Group) or "categories" not in obs_group[column]:
                needs_copy = True
                if column not in changed_columns:
                    changed_columns.append(column)

        if not needs_copy:
            logger.info("Source dataset already satisfies cell_load metadata requirements: %s", source_path)
            return dataset_dir, dataset_name

    if os.path.exists(runtime_path):
        os.remove(runtime_path)

    adata = ad.read_h5ad(source_path)
    try:
        for column in (pert_col, batch_col):
            if not column:
                continue
            values = adata.obs[column].astype(str).str.strip()
            adata.obs[column] = pd.Categorical(values)

        if cell_type_key:
            if cell_type_key in adata.obs.columns:
                cell_type_values = adata.obs[cell_type_key].astype(str).str.strip()
            else:
                cell_type_values = pd.Series(
                    np.full(adata.n_obs, cell_type_name, dtype=object),
                    index=adata.obs_names,
                )
            adata.obs[cell_type_key] = pd.Categorical(cell_type_values)

        adata.write_h5ad(runtime_path)
    finally:
        del adata

    logger.info(
        "Created runtime cell_load copy: %s. Touched obs columns: %s",
        runtime_path,
        ", ".join(changed_columns),
    )
    return runtime_dir, runtime_name


def _is_vector_like(value) -> bool:
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value)
        return arr.ndim == 1 and arr.size > 0
    return False


def normalize_embedding_mapping(obj, source_name: str) -> Dict[str, np.ndarray]:
    """
    Normalize supported pickle payloads into a {name: vector} dictionary.
    """
    if isinstance(obj, dict):
        if obj and all(isinstance(k, str) and _is_vector_like(v) for k, v in obj.items()):
            return {
                key: np.asarray(value, dtype=np.float32)
                for key, value in obj.items()
            }

        if "embeddings" in obj and isinstance(obj["embeddings"], dict):
            return normalize_embedding_mapping(obj["embeddings"], source_name)

        names = None
        vectors = None
        for key in ("gene_names", "genes", "pert_names", "names", "symbols", "keys"):
            if key in obj:
                names = obj[key]
                break
        for key in ("gene_emb_X", "pert_emb_X", "embeddings", "vectors", "X", "values"):
            if key in obj and not isinstance(obj[key], dict):
                vectors = obj[key]
                break
        if names is not None and vectors is not None:
            names = [str(name) for name in names]
            vectors = np.asarray(vectors, dtype=np.float32)
            if vectors.ndim != 2:
                raise ValueError(
                    f"Embedding matrix in {source_name} must be 2D, got shape {vectors.shape}."
                )
            if len(names) != vectors.shape[0]:
                raise ValueError(
                    f"Embedding names and matrix size mismatch in {source_name}: "
                    f"{len(names)} names vs {vectors.shape[0]} rows."
                )
            return {name: vectors[idx] for idx, name in enumerate(names)}

    try:
        import pandas as pd  # optional

        if isinstance(obj, pd.DataFrame):
            return {
                str(index): row.to_numpy(dtype=np.float32, copy=True)
                for index, row in obj.iterrows()
            }
        if isinstance(obj, pd.Series):
            return {
                str(index): np.asarray(value, dtype=np.float32)
                for index, value in obj.items()
            }
    except Exception:
        pass

    raise ValueError(
        f"Unsupported embedding payload in {source_name}. "
        "Expected a dict[str, vector], a dict with names + matrix, or a pandas object."
    )


def load_pickle_embedding_mapping(path: str) -> Dict[str, np.ndarray]:
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    return normalize_embedding_mapping(payload, source_name=path)


def merge_embedding_mappings(paths: Iterable[str]) -> Dict[str, np.ndarray]:
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]

    merged = {}
    embedding_dim = None

    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Embedding file not found: {path}")

        current = load_pickle_embedding_mapping(path)
        if not current:
            logger.warning("No embeddings found in %s", path)
            continue

        current_dim = len(next(iter(current.values())))
        if embedding_dim is None:
            embedding_dim = current_dim
        elif current_dim != embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {embedding_dim}, "
                f"but {path} has dimension {current_dim}."
            )

        overlap = set(merged).intersection(current)
        if overlap:
            logger.info(
                "Overriding %d duplicated perturbation embeddings from %s",
                len(overlap),
                path,
            )

        merged.update(current)

    return merged


def build_pert_embedding_matrix(
    pert_names: Iterable[str],
    embedding_paths: Iterable[str],
    control_pert: str,
    strict: bool = False,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Tuple[str, ...]]:
    pert_names = [str(name) for name in pert_names]
    if isinstance(embedding_paths, (str, os.PathLike)):
        embedding_paths = [embedding_paths]

    merged = merge_embedding_mappings(embedding_paths)
    if not merged:
        raise ValueError("No perturbation embeddings were loaded from the provided paths.")

    embedding_dim = len(next(iter(merged.values())))
    matrix = np.zeros((len(pert_names), embedding_dim), dtype=np.float32)
    filtered = {}
    missing = []

    for idx, pert_name in enumerate(pert_names):
        if pert_name == control_pert:
            continue

        vector = merged.get(pert_name)
        if vector is None:
            missing.append(pert_name)
            continue

        vector = np.asarray(vector, dtype=np.float32)
        if vector.ndim != 1 or vector.shape[0] != embedding_dim:
            raise ValueError(
                f"Embedding for perturbation {pert_name} has invalid shape {vector.shape}; "
                f"expected ({embedding_dim},)."
            )

        matrix[idx] = vector
        filtered[pert_name] = vector

    if strict and missing:
        raise ValueError(
            "Missing GenePT embeddings for dataset perturbations: "
            + ", ".join(sorted(missing)[:20])
            + (" ..." if len(missing) > 20 else "")
        )

    logger.info(
        "Loaded %d merged GenePT embeddings, kept %d/%d dataset perturbations, dropped %d extras, missing %d.",
        len(merged),
        len(filtered),
        len(pert_names) - int(control_pert in pert_names),
        max(len(merged) - len(filtered), 0),
        len(missing),
    )

    return matrix, filtered, tuple(sorted(missing))


def prepare_pert_embeddings_from_obs(
    obs,
    pert_col: str,
    embedding_paths: Iterable[str],
    control_pert: str = "non-targeting",
    strict: bool = False,
):
    """
    Extract unique perturbations from obs[pert_col] and build a dataset-aligned
    perturbation embedding matrix.
    """
    if pert_col not in obs:
        raise KeyError(f"Column {pert_col} not found in obs.")

    pert_names = sorted({str(value) for value in obs[pert_col].tolist()})
    matrix, filtered, missing = build_pert_embedding_matrix(
        pert_names=pert_names,
        embedding_paths=embedding_paths,
        control_pert=control_pert,
        strict=strict,
    )
    return pert_names, matrix, filtered, missing


def build_cell_load_toml_from_csv(
    csv_path: str,
    dataset_dir: str,
    dataset_name: str,
    cell_type_name: str,
    output_path: str,
    control_pert: str,
    pert_col: str,
    batch_col: Optional[str],
    cell_type_key: Optional[str],
    train_valid_ratio: float = 0.05,
    train_valid_seed: int = 42,
    split_column: Optional[str] = None,
    train_split_groups=None,
    test_split_groups=None,
    use_split_gene_space: bool = False,
    split_gene_col: str = "gene",
) -> str:
    """
    Convert the repo's perturbation split CSV into a Cell Load TOML config.

    Expected CSV columns:
      - pert
      - split / split_calib
    Split values are controlled by train_split_groups and test_split_groups.
    """
    split_df = pd.read_csv(csv_path)
    candidate_split_columns = []
    for candidate in (split_column, "split", "split_calib"):
        if candidate and candidate not in candidate_split_columns:
            candidate_split_columns.append(candidate)
    split_col = next((col for col in candidate_split_columns if col in split_df.columns), None)
    if split_col is None:
        raise ValueError(
            "Split CSV must contain a split column. "
            f"Tried {candidate_split_columns}, available columns are {sorted(split_df.columns.tolist())}."
        )

    required_cols = {"pert", split_col}
    missing_cols = required_cols - set(split_df.columns)
    if missing_cols:
        raise ValueError(
            f"Split CSV must contain columns {sorted(required_cols)}, missing {sorted(missing_cols)}."
        )

    split_df = split_df.copy()
    split_df["pert"] = split_df["pert"].astype(str)
    split_series = split_df[split_col].astype(str).str.lower()
    train_split_groups = _parse_split_groups(train_split_groups, "train_split_groups")
    test_split_groups = _parse_optional_split_groups(test_split_groups, "test_split_groups")

    split_per_pert = split_df.groupby("pert")[split_col].nunique()
    bad_perts = split_per_pert[split_per_pert > 1]
    if not bad_perts.empty:
        raise ValueError(
            "Each perturbation must map to exactly one split. "
            f"Found conflicting split assignments for: {bad_perts.index.tolist()[:10]}"
        )

    train_perts = (
        split_df.loc[split_series.isin(train_split_groups), "pert"]
        .astype(str)
        .dropna()
        .unique()
        .tolist()
    )
    test_perts = (
        split_df.loc[split_series.isin(test_split_groups), "pert"]
        .astype(str)
        .dropna()
        .unique()
        .tolist()
        if test_split_groups
        else []
    )

    if not train_perts:
        raise ValueError(
            f"No train perturbations found in {csv_path} using column {split_col} "
            f"and groups {train_split_groups}."
        )
    if test_split_groups and not test_perts:
        raise ValueError(
            f"No perturbations found for prediction groups {test_split_groups} "
            f"in {csv_path} using column {split_col}."
        )

    if test_perts and control_pert in test_perts:
        raise ValueError(
            f"Control perturbation {control_pert!r} must not be assigned to test in {csv_path}."
        )

    train_perts = [pert for pert in train_perts if pert != control_pert]
    if not train_perts:
        raise ValueError("No non-control train perturbations remain after excluding the control perturbation.")

    if not (0.0 < train_valid_ratio < 1.0):
        raise ValueError(f"train_valid_ratio must be between 0 and 1, got {train_valid_ratio}.")

    rng = np.random.default_rng(train_valid_seed)
    shuffled_train_perts = list(train_perts)
    rng.shuffle(shuffled_train_perts)

    n_val = int(round(len(shuffled_train_perts) * train_valid_ratio))
    n_val = max(1, min(n_val, len(shuffled_train_perts) - 1))
    val_perts = sorted(shuffled_train_perts[:n_val])
    final_train_perts = sorted(shuffled_train_perts[n_val:])

    if not final_train_perts:
        raise ValueError("Validation split consumed all train perturbations; adjust train_valid_ratio.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    source_path = resolve_dataset_h5ad_path(dataset_dir, dataset_name)
    runtime_dir = os.path.join(os.path.dirname(output_path), ".cell_load_runtime")
    os.makedirs(runtime_dir, exist_ok=True)

    if _parse_bool(use_split_gene_space):
        common_genes = _read_split_genes(csv_path, split_gene_col)
    else:
        common_genes = _read_var_names(source_path)

    train_dataset_name = f"{dataset_name}_train"
    train_runtime_path = os.path.join(runtime_dir, f"{train_dataset_name}.h5ad")
    if os.path.exists(train_runtime_path):
        os.remove(train_runtime_path)
    train_summary = _write_runtime_h5ad_subset(
        source_path=source_path,
        runtime_path=train_runtime_path,
        common_genes=common_genes,
        cell_line=cell_type_name,
        pert_col=pert_col,
        batch_col=batch_col,
        cell_type_key=cell_type_key,
        control_pert=control_pert,
        allowed_perts=final_train_perts,
        include_controls=True,
    )

    val_dataset_name = f"{dataset_name}_val"
    val_runtime_path = os.path.join(runtime_dir, f"{val_dataset_name}.h5ad")
    if os.path.exists(val_runtime_path):
        os.remove(val_runtime_path)
    val_summary = _write_runtime_h5ad_subset(
        source_path=source_path,
        runtime_path=val_runtime_path,
        common_genes=common_genes,
        cell_line=cell_type_name,
        pert_col=pert_col,
        batch_col=batch_col,
        cell_type_key=cell_type_key,
        control_pert=control_pert,
        allowed_perts=val_perts,
        include_controls=True,
    )

    test_dataset_name = None
    test_runtime_path = None
    test_summary = None
    if test_split_groups:
        test_dataset_name = f"{dataset_name}_test"
        test_runtime_path = os.path.join(runtime_dir, f"{test_dataset_name}.h5ad")
        if os.path.exists(test_runtime_path):
            os.remove(test_runtime_path)
        test_summary = _write_runtime_h5ad_subset(
            source_path=source_path,
            runtime_path=test_runtime_path,
            common_genes=common_genes,
            cell_line=cell_type_name,
            pert_col=pert_col,
            batch_col=batch_col,
            cell_type_key=cell_type_key,
            control_pert=control_pert,
            allowed_perts=test_perts,
            include_controls=True,
        )

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("# Auto-generated from split CSV for Cell Load.\n")
        handle.write("[datasets]\n")
        _write_dataset_entry(handle, train_dataset_name, train_runtime_path)
        _write_dataset_entry(handle, val_dataset_name, val_runtime_path)
        if test_dataset_name is not None:
            _write_dataset_entry(handle, test_dataset_name, test_runtime_path)
        handle.write("\n")
        handle.write("[training]\n")
        handle.write(f'{train_dataset_name} = "train"\n\n')
        handle.write("[zeroshot]\n")
        handle.write(f'"{val_dataset_name}.{cell_type_name}" = "val"\n')
        if test_dataset_name is not None:
            handle.write(f'"{test_dataset_name}.{cell_type_name}" = "test"\n')
        handle.write("\n")

    logger.info(
        (
            "Generated Cell Load TOML from %s using split column %s. "
            "CSV train=%d, derived train=%d, derived val=%d, prediction(%s)=%d, seed=%d, ratio=%.4f. "
            "Runtime summaries train(cells=%d, ctrls=%d, perts=%d) val(cells=%d, ctrls=%d, perts=%d) "
        "test(cells=%s, ctrls=%s, perts=%s). Output: %s"
        ),
        csv_path,
        split_col,
        len(train_perts),
        len(final_train_perts),
        len(val_perts),
        ",".join(test_split_groups),
        len(test_perts),
        train_valid_seed,
        train_valid_ratio,
        train_summary["cells"],
        train_summary["controls"],
        train_summary["perturbed"],
        val_summary["cells"],
        val_summary["controls"],
        val_summary["perturbed"],
        test_summary["cells"] if test_summary is not None else "skipped",
        test_summary["controls"] if test_summary is not None else "skipped",
        test_summary["perturbed"] if test_summary is not None else "skipped",
        output_path,
    )
    return output_path

def prepare_cell_load_kwargs(
    data_kwargs: Dict,
    output_dir: str,
    default_dataset_name: str = "jurkat",
    default_cell_type_name: Optional[str] = None,
    mode: str = "within",
    cross_cell_line: Optional[str] = None,
) -> Dict:
    """
    Normalize datamodule kwargs and optionally auto-generate the Cell Load TOML
    from a CSV split file.
    """
    kwargs = dict(data_kwargs)
    split_csv_path = kwargs.get("split_csv_path")
    dataset_dir = kwargs.get("dataset_dir")
    dataset_name = kwargs.get("dataset_name", default_dataset_name)
    cell_type_name = kwargs.get(
        "cell_type_name", default_cell_type_name or dataset_name
    )
    generated_toml_path = kwargs.get("generated_toml_path")
    split_column = kwargs.get("split_column")
    train_split_groups = kwargs.get("train_split_groups")
    test_split_groups = kwargs.get("test_split_groups")
    calib_split_groups = kwargs.get("calib_split_groups")
    train_valid_ratio = kwargs.get("train_valid_ratio", 0.05)
    train_valid_seed = kwargs.get("train_valid_seed")
    control_pert = kwargs.get("control_pert", "non-targeting")
    pert_col = kwargs.get("pert_col")
    batch_col = kwargs.get("batch_col")
    cell_type_key = kwargs.get("cell_type_key")
    use_split_gene_space = _parse_bool(kwargs.get("use_split_gene_space", False))
    split_gene_col = kwargs.get("split_gene_col", "gene")
    dropped_test_perts_json = kwargs.get("dropped_test_perts_json")

    if mode == "cross":
        if dataset_dir is None:
            raise ValueError("data.kwargs.dataset_dir must be provided for cross mode.")
        if split_csv_path is None:
            raise ValueError("data.kwargs.split_csv_path must be provided for cross mode.")
        if not cross_cell_line:
            raise ValueError("cross_cell_line must be provided for cross mode.")

        source_dataset_dir = _resolve_cross_source_dataset_dir(dataset_dir, split_csv_path)
        if source_dataset_dir != dataset_dir:
            logger.info(
                "Cross mode: using raw dataset_dir=%s instead of runtime dataset_dir=%s",
                source_dataset_dir,
                dataset_dir,
            )
        kwargs["source_dataset_dir"] = source_dataset_dir

        cross_paths = build_cross_cell_load_config(
            dataset_dir=source_dataset_dir,
            cross_cell_line=cross_cell_line,
            reference_split_csv=split_csv_path,
            output_dir=output_dir,
            pert_col=pert_col,
            batch_col=batch_col,
            cell_type_key=cell_type_key,
            control_pert=control_pert,
            split_column=split_column,
            train_split_groups=train_split_groups,
            test_split_groups=test_split_groups,
            calibration_split_groups=calib_split_groups,
            train_valid_ratio=float(train_valid_ratio),
            train_valid_seed=int(train_valid_seed or 42),
            use_split_gene_space=use_split_gene_space,
            split_gene_col=split_gene_col,
            dropped_test_perts_json=dropped_test_perts_json,
        )
        kwargs.update(cross_paths)
        kwargs["cell_type_name"] = cross_cell_line
        return kwargs

    if dataset_dir is not None:
        dataset_dir, dataset_name = ensure_cell_load_runtime_copy(
            dataset_dir=dataset_dir,
            dataset_name=dataset_name,
            output_dir=output_dir,
            cell_type_name=cell_type_name,
            pert_col=pert_col,
            batch_col=batch_col,
            cell_type_key=cell_type_key,
        )
        kwargs["dataset_dir"] = dataset_dir
        kwargs["dataset_name"] = dataset_name

    if split_csv_path is not None:
        if dataset_dir is None:
            raise ValueError("data.kwargs.dataset_dir must be provided when using split_csv_path.")
        if generated_toml_path is None:
            generated_toml_path = os.path.join(output_dir, "cell_load_split.toml")
        if train_valid_seed is None:
            train_valid_seed = 42
        kwargs["toml_config_path"] = build_cell_load_toml_from_csv(
            csv_path=split_csv_path,
            dataset_dir=dataset_dir,
            dataset_name=dataset_name,
            cell_type_name=cell_type_name,
            output_path=generated_toml_path,
            control_pert=control_pert,
            pert_col=pert_col,
            batch_col=batch_col,
            cell_type_key=cell_type_key,
            train_valid_ratio=float(train_valid_ratio),
            train_valid_seed=int(train_valid_seed),
            split_column=split_column,
            train_split_groups=train_split_groups,
            test_split_groups=test_split_groups,
            use_split_gene_space=use_split_gene_space,
            split_gene_col=split_gene_col,
        )

    return kwargs
