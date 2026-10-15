import os
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root, resolved from this file's location
import sys
import argparse
os.chdir(_REPO)
_root = os.path.abspath('.')
sys.path.insert(0, _root)

import anndata as ad
import scanpy as sc
from calibration.calibrate_sparsity import calibrate_zeros

ds_list = ['k562', 'rpe1', 'hepg2', 'jurkat']

parser = argparse.ArgumentParser()
parser.add_argument('--task_id', type=int, required=True, choices=range(len(ds_list)))
args = parser.parse_args()
ds = ds_list[args.task_id]

pred_dir = f'Scouter/Scouter/within_cond/prediction/{ds}'
out_dir  = 'calibration/sparsity_calibrated/Scouter'
os.makedirs(out_dir, exist_ok=True)

print(f'\n=== {ds} ===')
calib_ad = sc.read_h5ad(f'{pred_dir}/{ds}_calib_pred.h5ad')
test_ad  = sc.read_h5ad(f'{pred_dir}/{ds}_test_pred.h5ad')

# Strip ctrl from test to avoid duplicating control cells (same logic as scouter_metric_full.py)
test_perts = test_ad[test_ad.obs['gene'] != 'ctrl']
print(f'Calib cells: {calib_ad.n_obs}, Test pert cells (ctrl removed): {test_perts.n_obs}')

combined = ad.concat([calib_ad, test_perts], join='inner')
combined.var = calib_ad.var.loc[combined.var_names]  # restore gene_id, gene_name dropped by concat
combined.obs_names_make_unique()
print(f'Combined cells: {combined.n_obs}')

combined = calibrate_zeros(
    combined,
    target_col='gene',
    control_name='ctrl',
    ratio=1.0,
)

out_path = os.path.join(out_dir, f'{ds}_testcalib_pred.h5ad')
print(f'Saving to {out_path}')
combined.write_h5ad(out_path)
print('Done.')
