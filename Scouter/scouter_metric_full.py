import os
import sys
import argparse
os.chdir('/home/jsun140/scratchyiqunchen/jsun140/Project-Perturb')
_root = os.path.abspath('.')
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, 'majority_vote'))
import pandas as pd
import scanpy as sc
import anndata as ad
from perturb_metrics import compute_all_metrics_from_dfs

ds_list = ['k562', 'rpe1', 'hepg2', 'jurkat']

parser = argparse.ArgumentParser()
parser.add_argument('--task_id', type=int, required=True, choices=range(len(ds_list)))
args = parser.parse_args()
ds_list = [ds_list[args.task_id]]
ds_csv = {
    'k562':   '/home/jsun140/scratchyiqunchen/project_perturb_data/csv/DE_DOC_K562_splitted.csv',
    'rpe1':   '/home/jsun140/scratchyiqunchen/project_perturb_data/csv/DE_DOC_RPE1_splitted.csv',
    'hepg2':  '/home/jsun140/scratchyiqunchen/project_perturb_data/csv/DE_DOC_HepG2_splitted.csv',
    'jurkat': '/home/jsun140/scratchyiqunchen/project_perturb_data/csv/DE_DOC_Jurkat_splitted.csv',
}

truth_df = {ds: pd.read_csv(ds_csv[ds]) for ds in ds_list}

results = {}
for ds in ds_list:
    pred_dir = f'Scouter/prediction/{ds}'
    de_dir   = f'Scouter/de_label/{ds}'

    pred_ad = {s: sc.read_h5ad(f'{pred_dir}/{ds}_{s}_pred.h5ad') for s in ['calib', 'test']}
    real_ad = {s: sc.read_h5ad(f'{pred_dir}/{ds}_{s}_real.h5ad') for s in ['calib', 'test']}
    pred_de = {s: pd.read_csv(f'{de_dir}/{ds}-{s}.csv')          for s in ['calib', 'test']}
    real_de = {
        'calib': truth_df[ds][truth_df[ds]['split_calib'] == 'calibration'],
        'test':  truth_df[ds][truth_df[ds]['split_calib'] == 'test'],
    }

    test_pred_perts = pred_ad['test'][pred_ad['test'].obs['gene'] != 'ctrl']
    test_real_perts = real_ad['test'][real_ad['test'].obs['gene'] != 'ctrl']
    combined_pred_ad = ad.concat([pred_ad['calib'], test_pred_perts], join='inner')
    combined_pred_ad.obs_names_make_unique()
    combined_real_ad = ad.concat([real_ad['calib'], test_real_perts], join='inner')
    combined_real_ad.obs_names_make_unique()
    pred_de['combined'] = pd.concat([pred_de['calib'], pred_de['test']], ignore_index=True)
    real_de['combined'] = pd.concat([real_de['calib'], real_de['test']], ignore_index=True)
    pred_ad['combined'] = combined_pred_ad
    real_ad['combined'] = combined_real_ad

    for split in ['calib', 'test', 'combined']:
        print(f'\n{ds} / {split}')
        results[(ds, split)] = compute_all_metrics_from_dfs(
            pred_de_df = pred_de[split],
            real_de_df = real_de[split],
            pred_ad    = pred_ad[split],
            real_ad    = real_ad[split],
        )

ds = ds_list[0]
results_df = pd.DataFrame(results).T
results_df.index.names = ['dataset', 'split']
os.makedirs('Scouter/metrics', exist_ok=True)
out_path = f'Scouter/metrics/scouter_metric_{ds}.csv'
results_df.to_csv(out_path)
print(f'\nSaved to {out_path}')
print(results_df.to_string())
