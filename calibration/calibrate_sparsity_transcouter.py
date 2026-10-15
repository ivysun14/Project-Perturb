import os
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root, resolved from this file's location
import sys
import argparse
import numpy as np
import scipy.sparse as sp
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
pred_ds   = ds_list[args.task_id]
train_dss = [ds for ds in ds_list if ds != pred_ds]

pred_dir = f'Scouter/Scouter/cross_cond/prediction/{pred_ds}'
out_dir  = 'calibration/sparsity_calibrated/TranScouter'
os.makedirs(out_dir, exist_ok=True)

print(f'\n=== {pred_ds} | training ctrl pool: {train_dss} ===')

# 1. Pool ctrl cells from 3 training-line calib_real h5ads
ctrl_ads = []
for train_ds in train_dss:
    ds_ad = sc.read_h5ad(f'{pred_dir}/{pred_ds}_{train_ds}_calib_real.h5ad')
    ctrl_only = ds_ad[ds_ad.obs['gene'] == 'ctrl'].copy()
    print(f'{train_ds}: {ctrl_only.n_obs} ctrl cells')
    ctrl_ads.append(ctrl_only)
pooled_ctrl_ad = ad.concat(ctrl_ads, join='inner')
pooled_ctrl_ad.var = ctrl_ads[0].var.loc[pooled_ctrl_ad.var_names]
pooled_ctrl_ad.obs_names_make_unique()
print(f'Pooled ctrl: {pooled_ctrl_ad.n_obs} cells x {pooled_ctrl_ad.n_vars} genes')

# 2. Build combined object: pooled ctrl + pred pert cells
pred_ad   = sc.read_h5ad(f'{pred_dir}/{pred_ds}_testcalib_pred.h5ad')
pred_perts = pred_ad[pred_ad.obs['gene'] != 'ctrl'].copy()
print(f'Prediction object: {pred_ad.n_obs} cells ({pred_perts.n_obs} pert, {pred_ad.n_obs - pred_perts.n_obs} ctrl)')
combined = ad.concat([pooled_ctrl_ad, pred_perts], join='inner')
combined.var = pooled_ctrl_ad.var.loc[combined.var_names]
combined.obs_names_make_unique()

# 3. Calibrate using pooled ctrl zero ratios
combined = calibrate_zeros(combined, target_col='gene', control_name='ctrl', ratio=1.0)

# 4. Write calibrated pert expression back into pred_ad
n_ctrl    = pooled_ctrl_ad.n_obs
calib_X   = combined.X[n_ctrl:]          # calibrated pert rows, same order as pred_perts
X_out     = pred_ad.X.toarray() if sp.issparse(pred_ad.X) else np.array(pred_ad.X)
pert_mask = pred_ad.obs['gene'].values != 'ctrl'
X_out[pert_mask] = calib_X
pred_ad.X = X_out
pred_ad.var['gene_name'] = pred_ad.var_names

# 5. Save calibrated object
out_path = os.path.join(out_dir, f'{pred_ds}_testcalib_pred.h5ad')
pred_ad.write_h5ad(out_path)
print(f'Saved to {out_path}')
