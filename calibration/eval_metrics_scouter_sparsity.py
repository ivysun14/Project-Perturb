import os
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root, resolved from this file's location
import sys
import argparse
os.chdir(_REPO)
_root = os.path.abspath('.')
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, 'majority_vote'))
import pandas as pd
import scanpy as sc
import anndata as ad
from metrics.perturb_metrics import compute_all_metrics_from_dfs

ds_list = ['k562', 'rpe1', 'hepg2', 'jurkat']
ds_csv = {
    'k562':   '/projects/ychen756/project_perturb_data/csv/DE_DOC_K562_splitted.csv',
    'rpe1':   '/projects/ychen756/project_perturb_data/csv/DE_DOC_RPE1_splitted.csv',
    'hepg2':  '/projects/ychen756/project_perturb_data/csv/DE_DOC_HepG2_splitted.csv',
    'jurkat': '/projects/ychen756/project_perturb_data/csv/DE_DOC_Jurkat_splitted.csv',
}

parser = argparse.ArgumentParser()
parser.add_argument('--task_id', type=int, required=True, choices=range(len(ds_list)))
args = parser.parse_args()
ds = ds_list[args.task_id]

within_pred_dir = f'Scouter/Scouter/within_cond/prediction/{ds}'
calib_pred_dir  = 'calibration/sparsity_calibrated/Scouter'
de_dir          = 'calibration/sparsity_de/Scouter'
out_dir         = 'calibration/sparsity_metrics/Scouter'
os.makedirs(out_dir, exist_ok=True)

print(f'\n=== {ds} / testcalib (sparsity-calibrated) ===')

# Predicted: sparsity-calibrated combined testcalib object
pred_ad = sc.read_h5ad(f'{calib_pred_dir}/{ds}_testcalib_pred.h5ad')

# Real: combine calib_real + test_real (strip ctrl from test to avoid duplication)
calib_real_ad = sc.read_h5ad(f'{within_pred_dir}/{ds}_calib_real.h5ad')
test_real_ad  = sc.read_h5ad(f'{within_pred_dir}/{ds}_test_real.h5ad')
test_real_perts = test_real_ad[test_real_ad.obs['gene'] != 'ctrl']
real_ad = ad.concat([calib_real_ad, test_real_perts], join='inner')
real_ad.obs_names_make_unique()

# Predicted DE from sparsity-calibrated object
pred_de_df = pd.read_csv(f'{de_dir}/{ds}_testcalib_pred_DE.csv')

# Real DE: truth filtered to both calib and test splits
truth_df   = pd.read_csv(ds_csv[ds])
real_de_df = truth_df[truth_df['split_calib'].isin(['calibration', 'test'])]

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
