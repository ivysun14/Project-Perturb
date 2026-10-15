import os
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root, resolved from this file's location
import sys
import json
import argparse
os.chdir(_REPO)
_root = os.path.abspath('.')
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, 'majority_vote'))
import pandas as pd
import scanpy as sc
from metrics.perturb_metrics import compute_all_metrics_from_dfs

ds_list = ['k562', 'rpe1', 'hepg2', 'jurkat']
QA_DIR  = '/projects/ychen756/project_perturb_data/perturbqa_datasets'
ds_csv  = {
    'k562':   '/projects/ychen756/project_perturb_data/csv/DE_DOC_K562_splitted.csv',
    'rpe1':   '/projects/ychen756/project_perturb_data/csv/DE_DOC_RPE1_splitted.csv',
    'hepg2':  '/projects/ychen756/project_perturb_data/csv/DE_DOC_HepG2_splitted.csv',
    'jurkat': '/projects/ychen756/project_perturb_data/csv/DE_DOC_Jurkat_splitted.csv',
}

parser = argparse.ArgumentParser()
parser.add_argument('--task_id', type=int, required=True, choices=range(len(ds_list)))
args = parser.parse_args()
pred_ds = ds_list[args.task_id]

with open('/projects/ychen756/project_perturb_data/dropped_test_perts_QA.json') as f:
    dropped_perts = set(json.load(f)[pred_ds])
print(f'\n{pred_ds}: dropping {len(dropped_perts)} perts')

# exact QA test (pert, gene) pairs: drop dropped_perts, restrict to common gene universe
common_genes = set(
    pd.read_csv('/projects/ychen756/project_perturb_data/common_gene_universe_QA.csv').iloc[:, 0]
)
qa_de    = pd.read_csv(f'{QA_DIR}/{pred_ds}-de.csv')
qa_test  = qa_de[qa_de['split'] == 'test'].copy()
qa_test  = qa_test[~qa_test['pert'].isin(dropped_perts)]
qa_test  = qa_test[qa_test['gene'].isin(common_genes)]
qa_test['_key'] = qa_test['pert'] + '|||' + qa_test['gene']
test_key_set = set(qa_test['_key'])
print(f"  Exact eval pairs (no dropped perts, common genes only): {len(test_key_set)}, "
      f"perts: {qa_test['pert'].nunique()}, genes: {qa_test['gene'].nunique()}")

# ground truth DE labels from DE_DOC CSV, restricted to exact eval pairs
full_de    = pd.read_csv(ds_csv[pred_ds])
full_de['_key'] = full_de['pert'] + '|||' + full_de['gene']
real_de_df = full_de[full_de['_key'].isin(test_key_set)].drop(columns='_key').copy()

pred_dir   = f'TranScouter/perturbqa_expr/prediction/{pred_ds}'
de_dir     = f'TranScouter/perturbqa_expr/de_label/{pred_ds}'
pred_de_df = pd.read_csv(f'{de_dir}/{pred_ds}-test.csv')
pred_de_df = pred_de_df[~pred_de_df['pert'].isin(dropped_perts)]

pred_ad = sc.read_h5ad(f'{pred_dir}/{pred_ds}_test_pred.h5ad')
real_ad = sc.read_h5ad(f'{pred_dir}/{pred_ds}_test_real.h5ad')
pred_ad = pred_ad[~pred_ad.obs['gene'].isin(dropped_perts)].copy()
real_ad = real_ad[~real_ad.obs['gene'].isin(dropped_perts)].copy()

print(f'  pred_ad: {pred_ad.n_obs} cells, real_ad: {real_ad.n_obs} cells')
print(f'  pred_de: {pred_de_df["pert"].nunique()} perts, real_de: {real_de_df["pert"].nunique()} perts')

metrics = compute_all_metrics_from_dfs(
    pred_de_df=pred_de_df,
    real_de_df=real_de_df,
    pred_ad=pred_ad,
    real_ad=real_ad,
)

os.makedirs('TranScouter/perturbqa_expr/metrics', exist_ok=True)
results_df = pd.DataFrame([{'pred_ds': pred_ds, **metrics}])
out_path   = f'TranScouter/perturbqa_expr/metrics/transcouter_metric_drop_perts_{pred_ds}.csv'
results_df.to_csv(out_path, index=False)
print(f'\nSaved to {out_path}')
print(results_df.to_string())
