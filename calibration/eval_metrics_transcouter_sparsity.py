import os
import sys
import json
import argparse
os.chdir('/home/jsun140/scratchyiqunchen/jsun140/Project-Perturb')
_root = os.path.abspath('.')
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, 'majority_vote'))
import pandas as pd
import scanpy as sc
from perturb_metrics import compute_all_metrics_from_dfs

ds_list = ['k562', 'rpe1', 'hepg2', 'jurkat']
ds_csv = {
    'k562':   '/home/jsun140/scratchyiqunchen/project_perturb_data/csv/DE_DOC_K562_splitted.csv',
    'rpe1':   '/home/jsun140/scratchyiqunchen/project_perturb_data/csv/DE_DOC_RPE1_splitted.csv',
    'hepg2':  '/home/jsun140/scratchyiqunchen/project_perturb_data/csv/DE_DOC_HepG2_splitted.csv',
    'jurkat': '/home/jsun140/scratchyiqunchen/project_perturb_data/csv/DE_DOC_Jurkat_splitted.csv',
}

parser = argparse.ArgumentParser()
parser.add_argument('--task_id', type=int, required=True, choices=range(len(ds_list)))
args = parser.parse_args()
ds = ds_list[args.task_id]

calib_pred_dir  = 'calibration/sparsity_calibrated/TranScouter'
de_dir          = 'calibration/sparsity_de/TranScouter'
cross_pred_dir  = f'Scouter/Scouter/cross_cond/prediction/{ds}'
out_dir         = 'calibration/sparsity_metrics/TranScouter'
os.makedirs(out_dir, exist_ok=True)

gene_universe = set(pd.read_csv(
    '/home/jsun140/scratchyiqunchen/project_perturb_data/common_gene_universe.csv',
    index_col=0)['gene_name'].tolist())

with open('/home/jsun140/scratchyiqunchen/project_perturb_data/dropped_test_perts.json') as f:
    dropped_perts = set(json.load(f)[ds])

print(f'\n=== {ds} / testcalib (sparsity-calibrated, TranScouter) ===')
print(f'Gene universe: {len(gene_universe)} genes | Dropped perts: {len(dropped_perts)}')

pred_ad = sc.read_h5ad(f'{calib_pred_dir}/{ds}_testcalib_pred.h5ad')
pred_ad = pred_ad[~pred_ad.obs['gene'].isin(dropped_perts)].copy()

real_ad = sc.read_h5ad(f'{cross_pred_dir}/{ds}_testcalib_real.h5ad')
real_ad = real_ad[~real_ad.obs['gene'].isin(dropped_perts)].copy()

pred_de_df = pd.read_csv(f'{de_dir}/{ds}_testcalib_pred_DE.csv')
pred_de_df = pred_de_df[pred_de_df['gene'].isin(gene_universe) & ~pred_de_df['pert'].isin(dropped_perts)]

truth_df   = pd.read_csv(ds_csv[ds])
real_de_df = truth_df[truth_df['split_calib'].isin(['calibration', 'test'])]
real_de_df = real_de_df[real_de_df['gene'].isin(gene_universe) & ~real_de_df['pert'].isin(dropped_perts)]

print(f'pred_ad:    {pred_ad.n_obs} cells')
print(f'real_ad:    {real_ad.n_obs} cells')
print(f'pred_de_df: {len(pred_de_df)} rows, {pred_de_df["pert"].nunique()} perts')
print(f'real_de_df: {len(real_de_df)} rows, {real_de_df["pert"].nunique()} perts')

metrics = compute_all_metrics_from_dfs(
    pred_de_df = pred_de_df,
    real_de_df = real_de_df,
    pred_ad    = pred_ad,
    real_ad    = real_ad,
)

results_df = pd.DataFrame([{'dataset': ds, 'split': 'testcalib', **metrics}])
out_path   = os.path.join(out_dir, f'{ds}_metrics.csv')
results_df.to_csv(out_path, index=False)
print(f'\nSaved to {out_path}')
print(results_df.to_string())
