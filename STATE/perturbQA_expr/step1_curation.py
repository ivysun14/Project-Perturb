"""
step1_curation.py
=================
Adapted from STATE/step1_curation_full.py for the perturbQA subset.
- Training: Train + Test pert-gene pairs from 3 training cell lines + Train from held-out cell line
- Testing: Test pert-gene pairs of the held-out cell line
- Test perts of the held-out cell line not seen in the overall training set are dropped.
"""

import json
import scipy.sparse as sp
import numpy as np
import os
import sys
import argparse
import pandas as pd
import scanpy as sc

os.chdir('/home/jsun140/scratchyiqunchen/jsun140/Project-Perturb')
sys.path.insert(0, '/home/jsun140/scratchyiqunchen/jsun140/Project-Perturb')
from majority_vote.analysis_utility import resolve_repeats

parser = argparse.ArgumentParser()
parser.add_argument('--pred_ds', type=str, default=None, help='Single held-out cell line. If omitted, runs all 4.')
args = parser.parse_args()

DS_LIST = ['k562', 'rpe1', 'hepg2', 'jurkat']
DS_H5AD = {
    'k562':   '/home/jsun140/scratchyiqunchen/project_perturb_data/h5ad_objects/K562_raw.h5ad',
    'rpe1':   '/home/jsun140/scratchyiqunchen/project_perturb_data/h5ad_objects/RPE1_raw.h5ad',
    'hepg2':  '/home/jsun140/scratchyiqunchen/project_perturb_data/h5ad_objects/HepG2_raw.h5ad',
    'jurkat': '/home/jsun140/scratchyiqunchen/project_perturb_data/h5ad_objects/Jurkat_raw.h5ad',
}
QA_DIR = '/home/jsun140/scratchyiqunchen/project_perturb_data/perturbqa_datasets'

TRAIN_DATA_DIR = 'STATE/perturbQA_expr/training_data'
COVERAGE_DIR = 'STATE/perturbQA_expr/pert_coverage'
os.makedirs(TRAIN_DATA_DIR, exist_ok=True)
os.makedirs(COVERAGE_DIR,   exist_ok=True)


def scread_uniq_genes(ds):
    raw_ad = sc.read(DS_H5AD[ds])
    var_df = raw_ad.var.copy()
    var_df['gene_id']   = var_df.index
    var_df['gene_name'] = var_df['gene_name'].astype(str)
    raw_ad.var = var_df
    raw_ad = resolve_repeats(raw_ad, gene_row='gene_name')
    raw_ad.var = raw_ad.var.set_index('gene_name')
    raw_ad.obs['cell_line'] = ds
    return raw_ad


# gene universe = 1337 common gene space shared across all cross-condition methods
GENE_UNIVERSE_CSV = '/home/jsun140/scratchyiqunchen/project_perturb_data/common_gene_universe_QA.csv'
common_genes = pd.read_csv(GENE_UNIVERSE_CSV)['gene_name'].tolist()
print(f"PerturbQA common gene universe: {len(common_genes)} genes")

# load split info and raw h5ads once
split_df    = {ds: pd.read_csv(f'{QA_DIR}/{ds}-de.csv') for ds in DS_LIST}
raw_ad_dict = {ds: scread_uniq_genes(ds) for ds in DS_LIST}

# Load or initialise dropped_test_perts.json
dropped_json_path = f'{COVERAGE_DIR}/dropped_test_perts.json'
if os.path.exists(dropped_json_path):
    with open(dropped_json_path) as f:
        dropped_test_map = json.load(f)
else:
    dropped_test_map = {ds: [] for ds in DS_LIST}

pred_ds_list = [args.pred_ds] if args.pred_ds else DS_LIST

for pred_ds in pred_ds_list:
    
    print(f"\nProcessing target={pred_ds} ..........................")
    
    train_ds_list = [ds for ds in DS_LIST if ds != pred_ds]
    pred_split = split_df[pred_ds]
    train_perts = pred_split[pred_split['split'] == 'train']['pert'].unique().tolist()
    test_perts  = pred_split[pred_split['split'] == 'test' ]['pert'].unique().tolist()
    print(f"  PerturbQA perts: {len(train_perts)} train, {len(test_perts)} test")

    # Dropped test perts: test perts not seen in train+test of any training cell line
    seen = set()
    for td in train_ds_list:
        td_df = split_df[td]
        seen |= set(td_df['pert'].unique())  # all perturbQA perts of training cell lines
    dropped_test = sorted(set(test_perts) - seen)
    test_perts_filtered = [p for p in test_perts if p not in dropped_test]
    total_test_pairs = len(pred_split[pred_split['split'] == 'test'])
    dropped_test_pairs = len(pred_split[(pred_split['split'] == 'test') & pred_split['pert'].isin(dropped_test)])
    print(f"  Dropped test perts (unseen in training cell lines): {len(dropped_test)}")
    print(f"  Dropped test pert-gene pairs: {dropped_test_pairs}/{total_test_pairs} "
          f"({dropped_test_pairs/max(total_test_pairs,1):.1%})")

    dropped_test_map[pred_ds] = dropped_test
    with open(dropped_json_path, 'w') as f:
        json.dump(dropped_test_map, f, indent=2)

    # Task1 cell counts: test perts of held-out cell line
    test_cell_counts = (
        raw_ad_dict[pred_ds][raw_ad_dict[pred_ds].obs['gene'].isin(test_perts_filtered)]
        .obs['gene'].value_counts()
        .rename_axis('pert').reset_index(name='n_cells')
    )
    counts_path = f'{COVERAGE_DIR}/test_cell_counts_{pred_ds}.csv'
    test_cell_counts.to_csv(counts_path, index=False)
    print(f"  Task1 cell counts: {len(test_cell_counts)} perts, "
          f"{test_cell_counts['n_cells'].sum()} cells → {counts_path}")

    # Build held-out cell line h5ad: train + filtered test perts + ctrl
    pred_mask = raw_ad_dict[pred_ds].obs['gene'].isin(train_perts + test_perts_filtered + ['non-targeting'])
    raw_ad_pred = raw_ad_dict[pred_ds][pred_mask][:, common_genes].copy()
    raw_ad_pred.var = raw_ad_pred.var[['gene_id']]
    print(f"  Target {pred_ds}: {raw_ad_pred.n_obs} cells "
          f"({len(train_perts)} train + {len(test_perts_filtered)} test perts "
          f"[{len(dropped_test)} dropped] + ctrl)")

    # Build training cell line h5ads: all perturbQA perts + ctrl
    raw_ad_train_dict = {}
    for td in train_ds_list:
        td_perts = split_df[td]['pert'].unique().tolist()  # all train+test perts
        sub_ad = raw_ad_dict[td][raw_ad_dict[td].obs['gene'].isin(td_perts + ['non-targeting'])].copy()
        print(f"  Training {td}: {len(td_perts)} perts (train+test), {sub_ad.n_obs} cells")

        X = sub_ad.X.toarray() if sp.issparse(sub_ad.X) else sub_ad.X
        td_X_df = pd.DataFrame(X, index=sub_ad.obs_names, columns=sub_ad.var_names)
        # drop genes not in common set, and fill 0 for any gene in common_genes missing from td_X_df
        td_X_df = td_X_df.reindex(columns=common_genes, fill_value=0.)
        td_ad = sc.AnnData(
            X=td_X_df.values,
            obs=sub_ad.obs.copy(),
            var=raw_ad_pred.var[['gene_id']],
        )
        raw_ad_train_dict[td] = td_ad

    # Combine and normalize
    combined_ad = sc.concat([raw_ad_pred] + list(raw_ad_train_dict.values()), axis=0, join='inner')
    combined_ad.var = raw_ad_pred.var[['gene_id']].copy()
    combined_ad.obs_names_make_unique()
    combined_ad.obs.index.name  = 'cell_id'
    combined_ad.obs['cell_barcode'] = combined_ad.obs_names
    sc.pp.normalize_total(combined_ad, target_sum=1e4, key_added='size_factors')
    sc.pp.log1p(combined_ad)

    print(f"  Combined AnnData: {combined_ad.shape}")
    if not sp.issparse(combined_ad.X):
        combined_ad.X = sp.csr_matrix(combined_ad.X)
    combined_ad.X = combined_ad.X.astype(np.float32)
    out_h5ad = f'{TRAIN_DATA_DIR}/combined_for_{pred_ds}_perturbqa.h5ad'
    combined_ad.write_h5ad(out_h5ad)
    print(f"  Saved -> {out_h5ad}")

    ## ============ Combined AnnData structure ============ ##
    sep = '─' * 70
    print(f'\n{sep}')
    print(f'  Combined AnnData structure  (pred_ds={pred_ds})')
    print(sep)
    print(combined_ad)
    print(f'\n  X matrix')
    print(f'    shape : {combined_ad.X.shape}')
    print(f'    dtype : {combined_ad.X.dtype}')
    if sp.issparse(combined_ad.X):
        nnz   = combined_ad.X.nnz
        total = combined_ad.X.shape[0] * combined_ad.X.shape[1]
        print(f'    format: {combined_ad.X.format.upper()}  |  '
              f'non-zero: {nnz:,} / {total:,}  ({nnz/total:.1%} density)')
    print(f'\n  .obs  ({combined_ad.n_obs} cells × {len(combined_ad.obs.columns)} columns)')
    print(combined_ad.obs.dtypes.rename('dtype').to_frame().to_string())
    print(f'\n  .obs sample (first 3 rows):')
    print(combined_ad.obs.head(3).to_string())
    print(f'\n  .var  ({combined_ad.n_vars} genes × {len(combined_ad.var.columns)} columns)')
    print(f'    index name : {combined_ad.var.index.name!r}')
    print(combined_ad.var.dtypes.rename('dtype').to_frame().to_string())
    print(f'\n  .var sample (first 3 rows):')
    print(combined_ad.var.head(3).to_string())
    print(sep)
