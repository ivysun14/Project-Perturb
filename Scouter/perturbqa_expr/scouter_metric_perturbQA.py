import os
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root, resolved from this file's location
import sys
import argparse
os.chdir(_REPO)
_root = os.path.abspath('.')
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, 'majority_vote'))
import pandas as pd
import scanpy as sc
from metrics.perturb_metrics import compute_all_metrics_from_dfs

ds_list = ['k562', 'rpe1', 'hepg2', 'jurkat']
QA_DIR = '/projects/ychen756/project_perturb_data/perturbqa_datasets'
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
print(f'\nProcessing {ds} ..........................')

# test perts and exact (pert, gene) pairs defined by PerturbQA split
qa_de = pd.read_csv(f'{QA_DIR}/{ds}-de.csv')
qa_test = qa_de[qa_de['split'] == 'test'].copy()
test_pert  = set(qa_test['pert'].unique())
test_genes = set(qa_test['gene'].unique())
qa_test['_key'] = qa_test['pert'] + '|||' + qa_test['gene']
test_key_set = set(qa_test['_key'])
print(f"Test perts: {len(test_pert)}, Test genes: {len(test_genes)}, Exact test pairs: {len(test_key_set)}")

# ground truth DE labels from DE_DOC CSV, restricted to exact PerturbQA test (pert, gene) pairs
full_de = pd.read_csv(ds_csv[ds])
full_de['_key'] = full_de['pert'] + '|||' + full_de['gene']
real_de_df = full_de[full_de['_key'].isin(test_key_set)].drop(columns='_key').copy()
print(f"Ground truth rows: {len(real_de_df)}")

# predicted DE from eval script
de_dir     = f'Scouter/perturbqa_expr/de_label/{ds}'
pred_de_df = pd.read_csv(f'{de_dir}/{ds}-test.csv')

# expression h5ad
pred_dir = f'Scouter/perturbqa_expr/prediction/{ds}'
pred_ad  = sc.read_h5ad(f'{pred_dir}/{ds}_test_pred.h5ad')
real_ad  = sc.read_h5ad(f'{pred_dir}/{ds}_test_real.h5ad')

print(f'pred_ad: {pred_ad.shape}, real_ad: {real_ad.shape}')
print(f'pred_de_df: {len(pred_de_df)} rows, real_de_df: {len(real_de_df)} rows')

results = compute_all_metrics_from_dfs(
    pred_de_df=pred_de_df,
    real_de_df=real_de_df,
    pred_ad=pred_ad,
    real_ad=real_ad,
)

os.makedirs('Scouter/perturbqa_expr/metrics', exist_ok=True)
results_df = pd.DataFrame([results], index=[ds])
results_df.index.name = 'dataset'
out_path = f'Scouter/perturbqa_expr/metrics/scouter_metric_{ds}.csv'
results_df.to_csv(out_path)
print(f'\nSaved to {out_path}')
print(results_df.to_string())
