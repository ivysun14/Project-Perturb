import os
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root, resolved from this file's location
import sys
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

sys.path.insert(0, _REPO)
from preprocessing.data_utility import resolve_repeats

os.chdir(_REPO)

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

parser = argparse.ArgumentParser()
parser.add_argument('--pred_ds', type=str, default=None, choices=ds_list,
                    help='Held-out cell line. If omitted, runs all 4.')
args = parser.parse_args()

out_dir = 'TranScouter/training_data'
os.makedirs(out_dir, exist_ok=True)


def scread_uniq_genes(ds):
    raw_ad = sc.read(ds_filename[ds])
    var_df = raw_ad.var.copy()
    var_df['gene_id'] = var_df.index
    var_df['gene_name'] = var_df['gene_name'].astype(str)
    raw_ad.var = var_df
    raw_ad = resolve_repeats(raw_ad, gene_row='gene_name')
    raw_ad.var = raw_ad.var.set_index('gene_name')
    raw_ad.obs['cell_line'] = ds
    raw_ad.obs['gene'] = raw_ad.obs['gene'].cat.rename_categories({'non-targeting': 'ctrl'})
    return raw_ad


# Load all 4 datasets once
print("Loading all 4 datasets...")
raw_ad_dict = {ds: scread_uniq_genes(ds) for ds in ds_list}
split_df    = {ds: pd.read_csv(ds_csv[ds]) for ds in ds_list}

# Common gene set across all 4 cell lines
common_genes = sorted(set.intersection(*[set(raw_ad_dict[ds].var_names) for ds in ds_list]))
print(f"Common genes across all 4 datasets: {len(common_genes)}")

pred_ds_list = [args.pred_ds] if args.pred_ds else ds_list

for pred_ds in pred_ds_list:
    print(f"\nHeld-out = {pred_ds} ..........................")
    train_ds = [ds for ds in ds_list if ds != pred_ds]

    sub_ads = []
    for td in train_ds:
        # all perts (train + test + calib) for this training cell line; calib held out at training time
        td_perts = split_df[td]['pert'].unique().tolist()
        sub_ad = raw_ad_dict[td][
            raw_ad_dict[td].obs['gene'].isin(td_perts + ['ctrl'])
        ][:, common_genes].copy()
        sub_ad.obs = sub_ad.obs[['gene', 'gene_id', 'cell_line']].copy()
        sub_ad.var = sub_ad.var[['gene_id']].copy()
        sub_ad.var.index.name = None
        sub_ad.var['gene_name'] = sub_ad.var.index.astype(str)
        print(f"  {td}: {len(td_perts)} perts, {sub_ad.n_obs} cells")
        sub_ads.append(sub_ad)

    # free the full raw h5ads; sub_ads retain the needed slices
    del raw_ad_dict
    combined_ad = sc.concat(sub_ads, axis=0, join='inner')
    combined_ad.obs_names_make_unique()
    combined_ad.obs = combined_ad.obs[['gene', 'gene_id', 'cell_line']].copy()
    combined_ad.var = sub_ads[0].var[['gene_id']].copy()
    combined_ad.var.index.name = None
    combined_ad.var['gene_name'] = combined_ad.var.index.astype(str)

    sc.pp.normalize_total(combined_ad, target_sum=1e4, key_added='size_factors')
    sc.pp.log1p(combined_ad)

    if not sp.issparse(combined_ad.X):
        combined_ad.X = sp.csr_matrix(combined_ad.X)
    combined_ad.X = combined_ad.X.astype(np.float32)

    out_path = f'{out_dir}/combined_for_{pred_ds}.h5ad'
    combined_ad.write_h5ad(out_path)
    print(f"  Saved: {out_path}  shape={combined_ad.shape}")
