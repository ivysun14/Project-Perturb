import scipy.sparse as sp
import numpy as np
import os
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root, resolved from this file's location
import sys
import argparse
import pandas as pd
import scanpy as sc
sys.path.insert(0, _REPO)
from majority_vote.analysis_utility import resolve_repeats

os.chdir(_REPO)
parser = argparse.ArgumentParser()
parser.add_argument('--pred_ds', type=str, default=None, help='Single held-out cell line to process. If omitted, runs all 4.')
args = parser.parse_args()

ds_list = ['k562', 'rpe1', 'hepg2', 'jurkat']
ds_filename = {
    'k562':   '/projects/ychen756/project_perturb_data/h5ad_objects/K562_raw.h5ad',
    'rpe1':   '/projects/ychen756/project_perturb_data/h5ad_objects/RPE1_raw.h5ad',
    'hepg2':  '/projects/ychen756/project_perturb_data/h5ad_objects/HepG2_raw.h5ad',
    'jurkat': '/projects/ychen756/project_perturb_data/h5ad_objects/Jurkat_raw.h5ad',
}
ds_csv = {
    'k562':   '/projects/ychen756/project_perturb_data/csv/DE_DOC_K562_splitted.csv',
    'rpe1':   '/projects/ychen756/project_perturb_data/csv/DE_DOC_RPE1_splitted.csv',
    'hepg2':  '/projects/ychen756/project_perturb_data/csv/DE_DOC_HepG2_splitted.csv',
    'jurkat': '/projects/ychen756/project_perturb_data/csv/DE_DOC_Jurkat_splitted.csv',
}
out_dir = 'STATE/training_data'
os.makedirs(out_dir, exist_ok=True)


def scread_uniq_genes(ds):
    raw_ad = sc.read(f'{ds_filename[ds]}')
    var_df = raw_ad.var.copy()
    var_df['gene_id'] = var_df.index
    var_df['gene_name'] = var_df['gene_name'].astype(str)
    raw_ad.var = var_df
    raw_ad = resolve_repeats(raw_ad, gene_row='gene_name')
    raw_ad.var = raw_ad.var.set_index('gene_name')
    # required by STATE's cell_type_key
    raw_ad.obs['cell_line'] = ds
    return raw_ad


# load split information
split_df = {ds: pd.read_csv(ds_csv[ds]) for ds in ds_list}

# load all raw h5ads once
raw_ad_dict = {ds: scread_uniq_genes(ds) for ds in ds_list}

# common gene universe across all 4 datasets
common_genes = sorted(set.intersection(*[set(raw_ad_dict[ds].var_names) for ds in ds_list]))
print(f"Common genes across all datasets: {len(common_genes)}")

pred_ds_list = [args.pred_ds] if args.pred_ds else ds_list
for pred_ds in pred_ds_list:
    print(f"\nProcessing target={pred_ds} ..........................")
    train_ds = [i for i in ds_list if i != pred_ds]

    pred_split = split_df[pred_ds]
    train_perts = pred_split[pred_split['split_calib'] == 'train']['pert'].unique().tolist()
    test_perts  = pred_split[pred_split['split_calib'] == 'test']['pert'].unique().tolist()

    # gene universe = common gene set across all 4 datasets (same as TranScouter)
    gene_universe = common_genes

    # target/held-out cell line: train + test perts + ctrl (calibration excluded)
    pred_mask = raw_ad_dict[pred_ds].obs['gene'].isin(train_perts + test_perts + ['non-targeting'])
    raw_ad_pred = raw_ad_dict[pred_ds][pred_mask][:, gene_universe].copy()
    raw_ad_pred.var = raw_ad_pred.var[['gene_id']]
    print(f"  Target {pred_ds}: {len(train_perts)} train + {len(test_perts)} test perts, {raw_ad_pred.n_obs} cells")

    # Task 1: per-pert cell counts for pred_ds test perts
    test_cell_counts = (
        raw_ad_dict[pred_ds][raw_ad_dict[pred_ds].obs['gene'].isin(test_perts)]
        .obs['gene'].value_counts()
        .rename_axis('pert').reset_index(name='n_cells')
    )
    test_cell_counts.to_csv(f'{out_dir}/pred_task1_cell_counts_{pred_ds}.csv', index=False)
    print(f"  Task 1 cell counts: {len(test_cell_counts)} perts, {test_cell_counts['n_cells'].sum()} cells")

    # Task 2: per-pert cell counts for each training cell line's calibration perts
    calib_rows = []
    for td in train_ds:
        td_calib_perts = split_df[td][split_df[td]['split_calib'] == 'calibration']['pert'].unique()
        counts = (
            raw_ad_dict[td][raw_ad_dict[td].obs['gene'].isin(td_calib_perts)]
            .obs['gene'].value_counts()
            .rename_axis('pert').reset_index(name='n_cells')
        )
        counts.insert(0, 'cell_line', td)
        calib_rows.append(counts)
    calib_cell_counts = pd.concat(calib_rows, ignore_index=True)
    calib_cell_counts.to_csv(f'{out_dir}/pred_task2_cell_counts_{pred_ds}.csv', index=False)
    print(f"  Task 2 cell counts: {len(calib_cell_counts)} pert-cellline pairs, {calib_cell_counts['n_cells'].sum()} cells")

    # training cell lines: train + calib + test perts (equivalent to all perts)
    raw_ad_train_dict = {}
    for td in train_ds:
        td_split = split_df[td]
        td_perts = td_split[td_split['split_calib'].isin(['train', 'test', 'calibration'])]['pert'].unique().tolist()
        sub_ad = raw_ad_dict[td][raw_ad_dict[td].obs['gene'].isin(td_perts + ['non-targeting'])].copy()
        print(f"  Training {td}: {len(td_perts)} perts (train+test+calib), {sub_ad.n_obs} cells")

        X = sub_ad.X.toarray() if sp.issparse(sub_ad.X) else sub_ad.X
        td_X_df = pd.DataFrame(X, index=sub_ad.obs_names, columns=sub_ad.var_names)
        td_X_df = td_X_df.reindex(columns=gene_universe, fill_value=0.)
        td_ad = sc.AnnData(
            X=td_X_df.values,
            obs=sub_ad.obs.copy(),
            var=raw_ad_pred.var[['gene_id']],
        )
        raw_ad_train_dict[td] = td_ad

    # combine: target train cells + all training cell line cells
    combined_ad = sc.concat(
        [raw_ad_pred] + list(raw_ad_train_dict.values()), axis=0, join='inner'
    )
    combined_ad.var = raw_ad_pred.var[['gene_id']].copy()
    combined_ad.obs_names_make_unique()
    combined_ad.obs.index.name = "cell_id"
    combined_ad.obs["cell_barcode"] = combined_ad.obs_names

    sc.pp.normalize_total(combined_ad, target_sum=1e4, key_added="size_factors")
    sc.pp.log1p(combined_ad)

    print(f"  Combined AnnData: {combined_ad.shape}")
    if not sp.issparse(combined_ad.X):  # save as sparse matrix to save memory
        combined_ad.X = sp.csr_matrix(combined_ad.X)
    combined_ad.X = combined_ad.X.astype(np.float32)
    combined_ad.write_h5ad(f'{out_dir}/combined_for_{pred_ds}_full.h5ad')
    print(f"  Saved to {out_dir}/combined_for_{pred_ds}_full.h5ad")
