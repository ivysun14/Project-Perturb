import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
import scanpy as sc
from datetime import datetime
import pickle
import logging
import os
import json

def resolve_split_csv_path(split_csv_path_template, cell):
    """
    Resolve split CSV template with cell placeholders.

    Supported placeholders:
    - {cell}: original cell line name, e.g. "Jurkat"
    - {cell_lower}: lower-case cell line name, e.g. "jurkat"
    - {cell_upper}: upper-case cell line name, e.g. "JURKAT"
    """
    if split_csv_path_template is None:
        return None
    replacements = {
        "cell": str(cell),
        "cell_lower": str(cell).lower(),
        "cell_upper": str(cell).upper(),
    }
    try:
        return split_csv_path_template.format(**replacements)
    except (KeyError, ValueError):
        # If no valid placeholder is provided, treat it as a literal path.
        return split_csv_path_template


def resolve_split_csv_path_for_cell(split_csv_path_template, cell, expected_cells=None):
    """
    Resolve a split CSV path for one cell line.

    Literal paths are kept when they already point at the requested cell. For
    cross-cell-line runs, a literal reference path such as hepg2-de.csv can be
    used to infer sibling files such as k562-de.csv.
    """
    resolved = resolve_split_csv_path(split_csv_path_template, cell)
    if resolved is None:
        return None
    if expected_cells is None:
        return resolved

    path = os.path.abspath(resolved)
    if os.path.exists(path):
        name_lower = os.path.basename(path).lower()
        cell_lower = str(cell).lower()
        if cell_lower in name_lower:
            return path

    parent = os.path.dirname(path)
    filename = os.path.basename(path)
    expected = [str(c) for c in expected_cells]
    token_matches = []
    for candidate_cell in expected:
        start = filename.lower().find(candidate_cell.lower())
        if start >= 0:
            token_matches.append((-(len(candidate_cell)), start, start + len(candidate_cell), filename[start:start + len(candidate_cell)]))
    if not token_matches:
        return path

    _, start, end, token = sorted(token_matches)[0]
    if token.islower():
        rendered_cell = str(cell).lower()
    elif token.isupper():
        rendered_cell = str(cell).upper()
    else:
        rendered_cell = str(cell)
    inferred = os.path.join(parent, f"{filename[:start]}{rendered_cell}{filename[end:]}")
    return inferred


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _read_split_genes(csv_path, gene_col="gene"):
    split_df = pd.read_csv(csv_path, usecols=[gene_col])
    genes = [str(g).strip() for g in split_df[gene_col].dropna().tolist()]
    genes = [g for g in genes if g]
    if not genes:
        raise ValueError(f"No genes found in split CSV column {gene_col!r}: {csv_path}")
    return list(dict.fromkeys(genes))


def _subset_gene_space(adata, genes):
    if genes is None:
        return adata.copy()

    genes = [str(g) for g in genes]
    var_names = [str(v) for v in adata.var_names.tolist()]
    var_name_set = set(var_names)
    if all(g in var_name_set for g in genes):
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

    missing = [g for g in genes if g not in gene_to_var]
    if missing:
        raise ValueError(
            f"{len(missing)} split genes were not found in adata.var['gene_name']; "
            f"examples={missing[:10]}"
        )

    selected_var_names = [gene_to_var[g] for g in genes]
    subset = adata[:, selected_var_names].copy()
    subset.var_names = pd.Index(genes)
    subset.var["gene_name"] = genes
    return subset


def _load_dropped_test_perts(path):
    if path is None or not str(path).strip():
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dropped test perts JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {str(k).lower(): {str(v) for v in values} for k, values in raw.items()}


def balancer(adata, cell_line_key="cell_line", target_col="gene"):
    """
    Balance cell population across different cell lines via upsampling to the maximum class size. 

    Steps:
        1. Count number of cells per class.
        2. Identify maximum class size.
        3. For each class:
            Randomly sample (with replacement) to match max class size.
        4. Concatenate all resampled data into a new AnnData object.
    """
    pert_list = np.unique(adata.obs[target_col])
    all_data_x = []
    all_data_label = []
    all_data_condition = []
    for p in pert_list:
        adata_p = adata[adata.obs[target_col] == p]
        class_names = np.unique(adata_p.obs[cell_line_key])
        class_pop = {}
        for cls in class_names:
            class_pop[cls] = adata_p.copy()[adata_p.obs[cell_line_key] == cls].shape[0]
        max_number = np.max(list(class_pop.values()))
        for cls in class_names:
            temp = adata_p.copy()[adata_p.obs[cell_line_key] == cls]
            index = np.random.choice(range(len(temp)), max_number)
            if sparse.issparse(temp.X):
                temp_x = temp.X.A[index]
            else:
                temp_x = temp.X[index]
            all_data_x.append(temp_x)
            temp_ct = np.repeat(cls, max_number)
            all_data_label.append(temp_ct)
            temp_cc = np.repeat(np.unique(temp.obs[target_col]), max_number)
            all_data_condition.append(temp_cc)
    balanced_data = ad.AnnData(np.concatenate(all_data_x))
    balanced_data.obs[cell_line_key] = np.concatenate(all_data_label)
    balanced_data.obs[target_col] = np.concatenate(all_data_condition)
    return balanced_data

def data_read_split(data_path_dict,
                    cell_hold_out,
                    cell_line_key,
                    target_col,
                    control_name,
                    split_csv_path=None,
                    split_csv_dir=None,
                    split_csv_template="DE_DOC_{cell}_splitted.csv",
                    split_column="split_calib",
                    split_set_non_holdout="train",
                    split_set_holdout="test",
                    split_set_calibration="calibration",
                    use_split_gene_space=False,
                    split_gene_col="gene",
                    dropped_test_perts_json=None):
    """
    Read multiple AnnData files, split into holdout and non-holdout cell lines, and optionally downsample using a CSV filter.
    Prepare data for cross-cell-line settings.

    Returns:
        adata_non_holdout : 
                - All non-held-out cell lines
                - Both control and perturbed cells
        adata_holdout_ctrl : 
            Control cells from held-out cell line
        adata_holdout_pert : 
            Perturbed cells from held-out cell line
        adata_calibration : 
            Calibration cells from non-held-out cell line
    """
    use_split_gene_space = _parse_bool(use_split_gene_space)
    dropped_by_cell = _load_dropped_test_perts(dropped_test_perts_json)
    expected_cells = list(data_path_dict.keys())

    non_holdout_groups = {s.strip().lower() for s in str(split_set_non_holdout).split(",") if s.strip()}
    holdout_groups = {s.strip().lower() for s in str(split_set_holdout).split(",") if s.strip()}
    calibration_groups = {s.strip().lower() for s in str(split_set_calibration).split(",") if s.strip()}
    if not non_holdout_groups:
        non_holdout_groups = {split_set_non_holdout}
    if not holdout_groups:
        holdout_groups = {split_set_holdout}
    if not calibration_groups:
        calibration_groups = {split_set_calibration}

    def _split_csv_for_cell(cell):
        if split_csv_path is not None:
            return resolve_split_csv_path_for_cell(split_csv_path, cell, expected_cells=expected_cells)
        if split_csv_dir is not None:
            csv_template_path = os.path.join(split_csv_dir, split_csv_template)
            return resolve_split_csv_path(csv_template_path, cell)
        return None

    def _allowed_perts_for_groups(cell, groups, drop_dropped_test_perts=False):
        """
        Return the allowed perturbations for one cell line under the specified split groups.
        If split CSV source is None, return None (meaning no filtering by CSV).
        """
        csv_path = _split_csv_for_cell(cell)
        if csv_path is None:
            return None
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Split CSV not found: {csv_path}")
        split_df = pd.read_csv(csv_path)
        if "pert" not in split_df.columns:
            raise ValueError(f"'pert' column missing in split CSV: {csv_path}")
        if split_column not in split_df.columns:
            raise ValueError(f"Split column {split_column!r} missing in split CSV: {csv_path}")
        split_series = split_df[split_column].astype(str).str.lower()
        allowed = set(split_df.loc[split_series.isin(groups), "pert"].astype(str).tolist())
        if drop_dropped_test_perts:
            allowed -= dropped_by_cell.get(str(cell).lower(), set())
        return allowed

    def _subset_with_control_and_allowed(adata, allowed_perts):
        """
        Keep:
        - all control cells
        - perturbation cells whose target_col is in allowed_perts
        If allowed_perts is None, return adata unchanged.
        """
        if allowed_perts is None:
            return adata.copy()
        target_series = adata.obs[target_col].astype(str)
        is_control = target_series == control_name
        is_allowed = target_series.isin(allowed_perts)
        return adata[is_control | is_allowed].copy()

    def _subset_allowed_only(adata, allowed_perts):
        """
        Keep only perturbation cells whose target_col is in allowed_perts.
        If allowed_perts is None, return an empty AnnData object.
        """
        if allowed_perts is None:
            return adata[[]].copy()
        target_series = adata.obs[target_col].astype(str)
        is_allowed = target_series.isin(allowed_perts)
        return adata[is_allowed].copy()

    # 1) Read raw data for all cell lines
    adata_map = {}
    for cell, path in data_path_dict.items():
        adata = ad.read_h5ad(path)
        adata.obs[cell_line_key] = cell
        adata_map[cell] = adata
    if cell_hold_out not in adata_map:
        raise ValueError(f"cell_hold_out={cell_hold_out!r} not found in data_path_dict keys.")

    # 2) Common-gene intersection across all cell lines
    common_genes = None
    if use_split_gene_space:
        for cell in adata_map:
            csv_path = _split_csv_for_cell(cell)
            if csv_path is None:
                raise ValueError("split_csv_path or split_csv_dir must be provided when use_split_gene_space=true.")
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Split CSV not found: {csv_path}")
            genes = set(_read_split_genes(csv_path, split_gene_col))
            common_genes = genes if common_genes is None else common_genes & genes
    else:
        for adata in adata_map.values():
            genes = set(map(str, adata.var_names))
            common_genes = genes if common_genes is None else common_genes & genes
    if not common_genes:
        raise ValueError("No common genes found across provided cell lines.")
    common_genes = sorted(common_genes)

    # 3) Process each cell line
    adata_non_holdout_list = []
    adata_calibration_list = []
    adata_holdout_ctrl = None
    adata_holdout_pert = None

    for cell, adata in adata_map.items():
        # Keep common genes and preprocess
        adata = _subset_gene_space(adata, common_genes)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        if "gene_name" not in adata.var.columns:
            adata.var["gene_name"] = adata.var_names.astype(str)

        if cell != cell_hold_out:
            # Non-held-out: build train set
            allowed_train = _allowed_perts_for_groups(cell, non_holdout_groups)
            adata_train = _subset_with_control_and_allowed(adata, allowed_train)
            adata_non_holdout_list.append(adata_train)
            # Non-held-out: build calibration set
            allowed_calib = _allowed_perts_for_groups(cell, calibration_groups)
            adata_calib = _subset_allowed_only(adata, allowed_calib)
            adata_calibration_list.append(adata_calib)
        else:
            # Held-out: split into control and test perturbations
            allowed_test = _allowed_perts_for_groups(cell, holdout_groups, drop_dropped_test_perts=True)
            adata_holdout = _subset_with_control_and_allowed(adata, allowed_test)
            target_series = adata_holdout.obs[target_col].astype(str)
            ctrl_mask = target_series == control_name
            adata_holdout_ctrl = adata_holdout[ctrl_mask].copy()
            adata_holdout_pert = adata_holdout[~ctrl_mask].copy()

    # 4) Concatenate non-held-out train and calibration sets
    if len(adata_non_holdout_list) == 0:
        raise ValueError("No non-held-out cell lines found for adata_non_holdout.")
    if len(adata_calibration_list) == 0:
        raise ValueError("No non-held-out cell lines found for adata_calibration.")

    adata_non_holdout = ad.concat(
        adata_non_holdout_list, join="inner", label="batch", merge="first"
    )
    adata_calibration = ad.concat(
        adata_calibration_list, join="inner", label="batch", merge="first"
    )

    return adata_non_holdout, adata_holdout_ctrl, adata_holdout_pert, adata_calibration

def get_embeddings(path, path_miss, pert_set):
    with open(path, "rb") as f:
        embeddings = pickle.load(f)
    with open(path_miss, "rb") as f:
        embeddings_miss = pickle.load(f)
    embeddings.update(embeddings_miss)
    embeddings = {
        k: np.asarray(v, dtype=np.float32)
        for k, v in embeddings.items()
        if k in pert_set
    }
    return embeddings

def similarity_emb(emb_train, emb_test, n):
    """
    Compute cosine similarity between test and train perturbation embeddings,
    and return the top-n nearest neighbors with normalized similarity weights.

    Returns:
        pd.DataFrame with columns:
            test_pert,
            nn_1 ... nn_n,
            weight_1 ... weight_n
    """
    train_names = list(emb_train.keys())
    train_matrix = np.stack([emb_train[k] for k in train_names])
    results = []
    for test_name, test_emb in emb_test.items():
        similarities = train_matrix @ test_emb  # (N_train,)
        top_idx = np.argsort(-similarities)[:n]
        # weights for the perturbation transfer
        weights = similarities[top_idx]
        wsum = weights.sum()
        if wsum == 0:
            weights = np.repeat(1.0 / len(weights), len(weights))
        else:
            weights = weights / wsum

        nearest_perts = [train_names[i] for i in top_idx]
        results.append([test_name] + nearest_perts + list(weights))
    columns = ["test_pert"] + [f"nn_{i+1}" for i in range(n)] + [f"weight_{i+1}" for i in range(n)]
    df_result = pd.DataFrame(results, columns=columns)
    return df_result

def train_valid_split(adata, valid_ratio, random_state, target_col, cell_line_key, min_count_per_group = 50):
    """
    Stratified split of AnnData into train/valid by (cell_line, condition).

    Important args:
        min_count_per_group: groups with too few cells will be handled safely
            (we keep them in train by default to avoid empty-group issues).
    """
    rng = np.random.default_rng(random_state)

    cl = adata.obs[cell_line_key].astype(str).values
    cond = adata.obs[target_col].astype(str).values
    group = np.char.add(np.char.add(cl, "||"), cond)   # e.g. "K562||ctrl"

    n = adata.n_obs
    all_idx = np.arange(n)

    valid_idx_list = []
    train_idx_list = []

    # split within each group
    for g in np.unique(group):
        idx_g = all_idx[group == g]
        m = len(idx_g)

        # size too small: put into train (safe default)
        if m < min_count_per_group:
            train_idx_list.append(idx_g)
            continue

        # desired valid count (at least 1 and at most m-1)
        k = int(np.round(m * valid_ratio))
        k = max(1, min(k, m - 1))

        perm = rng.permutation(idx_g)
        valid_idx_list.append(perm[:k])
        train_idx_list.append(perm[k:])

    train_idx = np.concatenate(train_idx_list) if len(train_idx_list) else np.array([], dtype=int)
    valid_idx = np.concatenate(valid_idx_list) if len(valid_idx_list) else np.array([], dtype=int)
    adata_train = adata[train_idx]
    adata_valid = adata[valid_idx]
    return adata_train, adata_valid

def train_test_split(
    adata,
    csv_path,
    target_col,
    control_name,
    split_set_train="train",
    split_set_test="test",
    split_column="split_calib",
):
    """
    Split AnnData into train and test sets based on perturbation-level split CSV.
    """

    split_df = pd.read_csv(csv_path)
    train_values = [s.strip().lower() for s in str(split_set_train).split(",") if s.strip()]
    test_values = [s.strip().lower() for s in str(split_set_test).split(",") if s.strip()]
    if not train_values:
        train_values = ["train"]
    if not test_values:
        test_values = ["test"]
    if split_column not in split_df.columns:
        raise ValueError(f"Split column {split_column!r} missing in split CSV: {csv_path}")
    split_series = split_df[split_column].astype(str).str.lower()
    train_perts = set(split_df.loc[split_series.isin(train_values), "pert"])
    test_perts  = set(split_df.loc[split_series.isin(test_values), "pert"])

    pert_series = adata.obs[target_col]
    is_control = pert_series == control_name
    is_train   = pert_series.isin(train_perts)
    is_test    = pert_series.isin(test_perts)

    adata_train = adata[is_train | is_control]
    adata_test  = adata[is_test]
    return adata_train, adata_test

def log_setting(log_dir, log_name):
    """
    Set up logging to file and console with timestamped log files.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{log_name}")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        filemode="a",
        format="%(message)s"
    )
    log = logging.getLogger(__file__)
    return log
