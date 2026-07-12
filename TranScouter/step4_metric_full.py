import os
import sys
import argparse
os.chdir('/home/jsun140/scratchyiqunchen/jsun140/Project-Perturb')
_root = os.path.abspath('.')
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, 'majority_vote'))
import pandas as pd
import scanpy as sc
from perturb_metrics import compute_all_metrics_from_dfs

ds_list = ['k562', 'rpe1', 'hepg2', 'jurkat']

parser = argparse.ArgumentParser()
parser.add_argument('--task_id', type=int, required=True, choices=range(len(ds_list)))
args = parser.parse_args()
pred_ds = ds_list[args.task_id]

ds_csv = {
    'k562':   '/home/jsun140/scratchyiqunchen/project_perturb_data/csv/DE_DOC_K562_splitted.csv',
    'rpe1':   '/home/jsun140/scratchyiqunchen/project_perturb_data/csv/DE_DOC_RPE1_splitted.csv',
    'hepg2':  '/home/jsun140/scratchyiqunchen/project_perturb_data/csv/DE_DOC_HepG2_splitted.csv',
    'jurkat': '/home/jsun140/scratchyiqunchen/project_perturb_data/csv/DE_DOC_Jurkat_splitted.csv',
}

pred_dir = f'TranScouter/prediction/{pred_ds}'
de_dir   = f'TranScouter/de_label/{pred_ds}'

print(f'\n{pred_ds} / testcalib')
truth_df   = pd.read_csv(ds_csv[pred_ds])
real_de_df = truth_df[truth_df['split_calib'].isin(['test', 'calibration'])]

pred_de_df = pd.read_csv(f'{de_dir}/{pred_ds}-testcalib.csv')
pred_ad    = sc.read_h5ad(f'{pred_dir}/{pred_ds}_testcalib_pred.h5ad')
real_ad    = sc.read_h5ad(f'{pred_dir}/{pred_ds}_testcalib_real.h5ad')

metrics = compute_all_metrics_from_dfs(
    pred_de_df = pred_de_df,
    real_de_df = real_de_df,
    pred_ad    = pred_ad,
    real_ad    = real_ad,
)

results_df = pd.DataFrame([{'pred_ds': pred_ds, **metrics}])
out_path = f'TranScouter/transcouter_metric_{pred_ds}.csv'
results_df.to_csv(out_path, index=False)
print(f'\nSaved to {out_path}')
print(results_df.to_string())
