import os
os.chdir('/users/jsun2/Project-Perturb')
import sys
sys.path.insert(0, 'majority_vote')
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import pearsonr
from analysis_utility import compute_3class_metrics

ds_list = ["rpe1", "jurkat", "hepg2", "k562"]
ds_csv = {
    'rpe1':   'data/DE_DOC_rpe1_splitted.csv',
    'jurkat': 'data/DE_DOC_jurkat_splitted.csv',
    'hepg2':  'data/DE_DOC_hepg2_splitted.csv',
    'k562':   'data/DE_DOC_K562_Replogle_splitted.csv',
}

# Calculate per-gene, per-perturbation mean
def mean_expr(ad, pert):
    X = ad[ad.obs['gene'] == pert].X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    return np.asarray(X.mean(axis=0)).squeeze()

results = {}
for ds in ds_list:
    print(f'Processing {ds}..........................')
    pred_dir = f'Scouter/Scouter/within_cond/prediction/scouter_full/{ds}'
    de_dir   = f'Scouter/Scouter/within_cond/de_label/scouter_full/{ds}'
    truth_df = pd.read_csv(ds_csv[ds])

    for split in ['test', 'calib']:
        print(f'  split={split}')
        
        if split == 'calib':
            split_col = 'calibration'
        else:
            split_col = 'test'

        # 3-class DE metrics
        de_pred = pd.read_csv(f'{de_dir}/{ds}-{split}.csv')
        de_real = truth_df[truth_df['split_calib'] == split_col][['pert', 'gene', 'label']]
        merged = de_real.merge(
            de_pred[['pert', 'gene', 'label']], on=['pert', 'gene'],
            how='inner', suffixes=('_real', '_pred')
        )
        metrics = compute_3class_metrics(merged['label_real'].values, merged['label_pred'].values)

        # Expression-level metrics: correlation and MSE
        # compute per-gene per-perturbation mean profiles, then concatenate into one 1D array
        pred_ad = sc.read_h5ad(f'{pred_dir}/{ds}_{split}_pred.h5ad')
        real_ad = sc.read_h5ad(f'{pred_dir}/{ds}_{split}_real.h5ad')
        perts = [p for p in pred_ad.obs['gene'].unique() if p != 'ctrl']

        pred_means = np.vstack([mean_expr(pred_ad, p) for p in perts])  # (n_perts, n_genes)
        real_means = np.vstack([mean_expr(real_ad, p) for p in perts])  # (n_perts, n_genes)

        corr, _ = pearsonr(pred_means.ravel(), real_means.ravel())      # pearson correlation
        mse = np.mean((pred_means - real_means) ** 2)                   # MSE

        metrics['pearson'] = corr
        metrics['mse']     = mse
        results[(ds, split)] = metrics

results_df = pd.DataFrame(results).T
results_df.index.names = ['dataset', 'split']
out_path = 'Scouter/Scouter/within_cond/scouter_metric_full.csv'
results_df.to_csv(out_path)
print(results_df.to_string())
print(f'\nSaved to {out_path}')
