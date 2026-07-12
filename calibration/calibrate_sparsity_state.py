import os
import sys
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
os.chdir('/home/jsun140/scratchyiqunchen/jsun140/Project-Perturb')
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

out_dir = 'calibration/sparsity_calibrated/STATE'
os.makedirs(out_dir, exist_ok=True)

print(f'\n=== {pred_ds} | training ctrl pool: {train_dss} ===')

# Gene universe: maps integer position → gene name (STATE var order)
gene_universe = pd.read_csv(
    '/home/jsun140/scratchyiqunchen/project_perturb_data/common_gene_universe.csv',
    index_col=0,
)['gene_name'].tolist()

# 1. Pool ctrl cells from STATE eval_calib.ckpt/adata_real.h5ad
# This object contains control cells from all 3 training cell lines
calib_real = sc.read_h5ad(f'STATE/results/{pred_ds}/eval_calib.ckpt/adata_real.h5ad')
calib_real.var_names = gene_universe

# Verify all 3 training cell lines are present
present = calib_real.obs['cell_line'].unique().tolist()
print(f'Cell lines in eval_calib real: {present}')
assert all(ds in present for ds in train_dss), f'Missing training cell lines: {set(train_dss) - set(present)}'
pooled_ctrl_ad = calib_real[calib_real.obs['gene'] == 'non-targeting'].copy()
pooled_ctrl_ad.obs_names_make_unique()
print(f'Pooled ctrl: {pooled_ctrl_ad.n_obs} cells x {pooled_ctrl_ad.n_vars} genes')
print(f'  Per cell line: {pooled_ctrl_ad.obs.cell_line.value_counts().to_dict()}')

# 2. Load STATE pred on 40% calib+test set of held-out cell line and rename var from integers to gene names
pred_ad = sc.read_h5ad(f'STATE/results/{pred_ds}/eval_task1_merged.ckpt/adata_pred.h5ad')
pred_ad.var_names = gene_universe
pred_ad.var['gene_name'] = pred_ad.var_names
print(f'STATE pred object: {pred_ad.n_obs} cells x {pred_ad.n_vars} genes')

# 3. Build combined object: pooled ctrl + pred pert cells
pred_perts = pred_ad[pred_ad.obs['gene'] != 'non-targeting'].copy()
print(f'Pert cells: {pred_perts.n_obs}  |  Ctrl cells: {pred_ad.n_obs - pred_perts.n_obs}')
combined = ad.concat([pooled_ctrl_ad, pred_perts], join='inner')
combined.var['gene_name'] = combined.var_names
combined.obs_names_make_unique()

# 4. Calibrate using pooled ctrl zero ratios
combined = calibrate_zeros(combined, target_col='gene', control_name='non-targeting', ratio=1.0)

# 5. Write calibrated pert rows back into pred_ad
n_ctrl    = pooled_ctrl_ad.n_obs
calib_X   = combined.X[n_ctrl:]
X_out     = pred_ad.X.toarray() if sp.issparse(pred_ad.X) else np.array(pred_ad.X)
pert_mask = pred_ad.obs['gene'].values != 'non-targeting'
X_out[pert_mask] = calib_X
pred_ad.X = X_out

# 6. Save
out_path = os.path.join(out_dir, f'{pred_ds}_testcalib_pred.h5ad')
pred_ad.write_h5ad(out_path)
print(f'Saved to {out_path}')
