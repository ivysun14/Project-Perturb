"""
step4_analysis.py
=================
Adapted from STATE/step4_analysis_full.py for the perturbQA subset.

Ground truth: DE_DOC_{ds}_splitted.csv filtered to exact QA test (pert, gene)
pairs, dropped perts excluded, restricted to 1337 common gene universe.
Predicted DE from STATE/perturbQA_expr/results/{pred_ds}/de_pred.csv.
Expression h5ads from eval_best.ckpt/.
"""

import os
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root, resolved from this file's location
import sys
import json
import argparse

os.chdir(_REPO)
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.join(os.path.abspath('.'), 'majority_vote'))

import pandas as pd
import scanpy as sc
from metrics.perturb_metrics import compute_all_metrics_from_dfs

DS_LIST = ['k562', 'rpe1', 'hepg2', 'jurkat']
DS_CSV  = {
    'k562':   '/projects/ychen756/project_perturb_data/csv/DE_DOC_K562_splitted.csv',
    'rpe1':   '/projects/ychen756/project_perturb_data/csv/DE_DOC_RPE1_splitted.csv',
    'hepg2':  '/projects/ychen756/project_perturb_data/csv/DE_DOC_HepG2_splitted.csv',
    'jurkat': '/projects/ychen756/project_perturb_data/csv/DE_DOC_Jurkat_splitted.csv',
}
QA_DIR             = '/projects/ychen756/project_perturb_data/perturbqa_datasets'
DROPPED_PERTS_JSON = '/projects/ychen756/project_perturb_data/dropped_test_perts_QA.json'
GENE_UNIVERSE_CSV  = '/projects/ychen756/project_perturb_data/common_gene_universe_QA.csv'

parser = argparse.ArgumentParser()
parser.add_argument('--task_id', type=int, required=True, choices=range(len(DS_LIST)))
args = parser.parse_args()
pred_ds = DS_LIST[args.task_id]

pred_dir = f'STATE/perturbQA_expr/results/{pred_ds}/eval_best.ckpt'
de_dir   = f'STATE/perturbQA_expr/results/{pred_ds}'

print(f'\n{pred_ds} / test split')

with open(DROPPED_PERTS_JSON) as f:
    dropped_perts = set(json.load(f)[pred_ds])
common_genes = set(pd.read_csv(GENE_UNIVERSE_CSV).iloc[:, 0])
print(f"  Dropping {len(dropped_perts)} perts; common gene universe: {len(common_genes)} genes")

# Ground truth: exact QA test (pert, gene) pairs, dropped perts excluded, restricted to common genes
qa_de   = pd.read_csv(f'{QA_DIR}/{pred_ds}-de.csv')
qa_test = qa_de[qa_de['split'] == 'test'].copy()
qa_test = qa_test[~qa_test['pert'].isin(dropped_perts)]
qa_test = qa_test[qa_test['gene'].isin(common_genes)]
qa_test['_key'] = qa_test['pert'] + '|||' + qa_test['gene']
test_key_set = set(qa_test['_key'])
print(f"  Eval pairs (no dropped perts, common genes): {len(test_key_set)}, "
      f"perts: {qa_test['pert'].nunique()}, genes: {qa_test['gene'].nunique()}")

full_truth_df = pd.read_csv(DS_CSV[pred_ds])
full_truth_df['_key'] = full_truth_df['pert'] + '|||' + full_truth_df['gene']
real_de_df = full_truth_df[full_truth_df['_key'].isin(test_key_set)].drop(columns='_key').copy()
print(f"  DE_DOC rows matched: {len(real_de_df)}")

pred_de_df = pd.read_csv(f'{de_dir}/de_pred.csv')
pred_de_df = pred_de_df[~pred_de_df['pert'].isin(dropped_perts)]

pred_ad = sc.read_h5ad(f'{pred_dir}/adata_pred.h5ad')
real_ad = sc.read_h5ad(f'{pred_dir}/adata_real.h5ad')

# STATE uses 'non-targeting'; perturb_metrics expects 'ctrl'
pred_ad.obs['gene'] = pred_ad.obs['gene'].replace('non-targeting', 'ctrl')
real_ad.obs['gene'] = real_ad.obs['gene'].replace('non-targeting', 'ctrl')

pred_ad = pred_ad[~pred_ad.obs['gene'].isin(dropped_perts)].copy()
real_ad = real_ad[~real_ad.obs['gene'].isin(dropped_perts)].copy()
print(f"  pred_ad: {pred_ad.n_obs} cells, real_ad: {real_ad.n_obs} cells")

metrics = compute_all_metrics_from_dfs(
    pred_de_df=pred_de_df,
    real_de_df=real_de_df,
    pred_ad=pred_ad,
    real_ad=real_ad,
)

results_df = pd.DataFrame([{'pred_ds': pred_ds, **metrics}])
out_path   = f'{de_dir}/STATE_perturbqa_metric_{pred_ds}.csv'
results_df.to_csv(out_path, index=False)
print(f'\nSaved to {out_path}')
print(results_df.to_string())
