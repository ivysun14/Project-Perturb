import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from datetime import datetime
import pickle
import logging
import os

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
    class_names = np.unique(adata.obs[cell_line_key])
    class_pop = {}
    for cls in class_names:
        class_pop[cls] = adata.copy()[adata.obs[cell_line_key] == cls].shape[0]
    max_number = np.max(list(class_pop.values()))
    all_data_x = []
    all_data_label = []
    all_data_condition = []
    for cls in class_names:
        temp = adata.copy()[adata.obs[cell_line_key] == cls]
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
    balanced_data = anndata.AnnData(np.concatenate(all_data_x))
    balanced_data.obs[cell_line_key] = np.concatenate(all_data_label)
    balanced_data.obs[target_col] = np.concatenate(all_data_condition)
    class_names = np.unique(balanced_data.obs[cell_line_key])
    class_pop = {}
    for cls in class_names:
        class_pop[cls] = len(balanced_data[balanced_data.obs[cell_line_key] == cls])
    return balanced_data

def data_read_split(data_path_dict,
                    cell_hold_out,
                    cell_line_key,
                    target_col,
                    control_name,
                    downsample=False,
                    csv_path=None):
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
    """
    adata_list = []

    for cell, path in data_path_dict.items():
        adata = ad.read_h5ad(path)
        adata.obs["cell_line"] = cell
        if downsample:
            adata = PerturbQA_filter(csv_path, adata) 
        if cell != cell_hold_out:
            adata_list.append(adata)
        else:
            ctrl_mask = adata.obs[target_col] == control_name
            adata_holdout_ctrl = adata[ctrl_mask]
            adata_holdout_pert = adata[~ctrl_mask]

    adata_non_holdout = ad.concat(adata_list, join="inner", label="batch")
    print(f"Non-holdout cell line: {adata_non_holdout.shape[0]} cells")
    print(f"Holdout cell line - control: {adata_holdout_ctrl.shape[0]} cells")
    print(f"Holdout cell line - perturbed: {adata_holdout_pert.shape[0]} cells")

    return adata_non_holdout, adata_holdout_ctrl, adata_holdout_pert

def get_embeddings(path, pert_set):
    with open(path, "rb") as f:
        embeddings = pickle.load(f)
    embeddings = {
        k: np.asarray(v, dtype=np.float32)
        for k, v in embeddings.items()
        if k in pert_set
    }
    return embeddings

def similarity_emb(emb_train, emb_test, n, tau=0.8):
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
    train_norm = train_matrix / np.linalg.norm(train_matrix, axis=1, keepdims=True)
    results = []
    for test_name, test_emb in emb_test.items():
        test_emb = test_emb / np.linalg.norm(test_emb)
        similarities = train_norm @ test_emb  # (N_train,)
        top_idx = np.argsort(-similarities)[:n]
        # weights for the perturbation transfer
        weights = similarities[top_idx]
        weights = weights / weights.sum()

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

def train_test_split(adata, csv_path, target_col, control_name):
    """
    Split AnnData into train and test sets based on perturbation-level split CSV.
    """

    split_df = pd.read_csv(csv_path)
    train_perts = set(split_df[split_df["split"] == "train"]["pert"])
    test_perts  = set(split_df[split_df["split"] == "test"]["pert"])
    print(f"Train perturbations: {len(train_perts)}")
    print(f"Test perturbations: {len(test_perts)}")

    pert_series = adata.obs[target_col]
    is_control = pert_series == control_name
    is_train   = pert_series.isin(train_perts)
    is_test    = pert_series.isin(test_perts)

    adata_train = adata[is_train | is_control]
    adata_test  = adata[is_test]

    print("\nFinal split:")
    print(f"Train cells: {adata_train.n_obs}")
    print(f"Test cells:  {adata_test.n_obs}")

    return adata_train, adata_test

def PerturbQA_filter(csv_path, adata, pert_col="pert", gene_col="gene", adata_pert_col="gene", control_key="non-targeting"):
    """
    Filter an AnnData object based on perturbation-gene pairs from a CSV file, following PerturbQA-style filtering.
    """
    print(f"Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    selected_perts = set(sorted(df[pert_col].unique()))
    selected_genes = set(sorted(df[gene_col].unique()))
    selected_perts.add(control_key)

    print(f"Selected perturbations: {len(selected_perts)}")
    print(f"Selected genes: {len(selected_genes)}")

    print("\nBefore Filtering:")
    print(f"Cells: {adata.n_obs}")
    print(f"Genes: {adata.n_vars}")
    # Filter perturbations (obs)
    adata = adata[adata.obs[adata_pert_col].isin(selected_perts)]

    # Filter genes (var)
    mask = adata.var["gene_name"].isin(selected_genes)
    adata = adata[:, mask]

    print("\nAfter Filtering:")
    print(f"Cells: {adata.n_obs}")
    print(f"Genes: {adata.n_vars}")
    return adata

def log_setting(log_dir):
    """
    Set up logging to file and console with timestamped log files.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),   
            logging.StreamHandler()                   
        ],
        force=True 
    )
    log = logging.getLogger(__file__)
    return log