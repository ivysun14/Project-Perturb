import os
import sys
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

sys.path.insert(0, '/home/jsun140/scratchyiqunchen/jsun140/Project-Perturb')
from majority_vote.analysis_utility import resolve_repeats

os.chdir('/home/jsun140/scratchyiqunchen/jsun140/Project-Perturb')

ds_list = ['k562', 'rpe1', 'hepg2', 'jurkat']
ds_filename = {
    'k562':   '/home/jsun140/scratchyiqunchen/project_perturb_data/h5ad_objects/K562_raw.h5ad',
    'rpe1':   '/home/jsun140/scratchyiqunchen/project_perturb_data/h5ad_objects/RPE1_raw.h5ad',
    'hepg2':  '/home/jsun140/scratchyiqunchen/project_perturb_data/h5ad_objects/HepG2_raw.h5ad',
    'jurkat': '/home/jsun140/scratchyiqunchen/project_perturb_data/h5ad_objects/Jurkat_raw.h5ad',
}
QA_DIR      = '/home/jsun140/scratchyiqunchen/project_perturb_data/perturbqa_datasets'
GENE_UNIVERSE = '/home/jsun140/scratchyiqunchen/project_perturb_data/common_gene_universe_QA.csv'

parser = argparse.ArgumentParser()
parser.add_argument('--pred_ds', type=str, required=True, choices=ds_list,
                    help='Held-out cell line.')
args = parser.parse_args()

pred_ds   = args.pred_ds
train_ds  = [ds for ds in ds_list if ds != pred_ds]
qa_genes  = pd.read_csv(GENE_UNIVERSE)['gene_name'].tolist()
out_dir   = 'TranScouter/perturbqa_expr/training_data'
os.makedirs(out_dir, exist_ok=True)

print(f"Held-out: {pred_ds}  |  Training on: {train_ds}")
print(f"PerturbQA gene universe: {len(qa_genes)} genes")


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


sub_ads = []
for td in train_ds:
    # all perts from PerturbQA file for this training cell line (train + test splits)
    td_perts = pd.read_csv(f'{QA_DIR}/{td}-de.csv')['pert'].unique().tolist()
    raw_ad = scread_uniq_genes(td)
    sub_ad = raw_ad[raw_ad.obs['gene'].isin(td_perts + ['ctrl'])][:, qa_genes].copy()
    sub_ad.obs = sub_ad.obs[['gene', 'gene_id', 'cell_line']].copy()
    sub_ad.var = sub_ad.var[['gene_id']].copy()
    sub_ad.var.index.name = None
    sub_ad.var['gene_name'] = sub_ad.var.index.astype(str)
    print(f"  {td}: {len(td_perts)} perts, {sub_ad.n_obs} cells, {sub_ad.n_vars} genes")
    sub_ads.append(sub_ad)
    del raw_ad

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
print(f"Saved: {out_path}  shape={combined_ad.shape}")
