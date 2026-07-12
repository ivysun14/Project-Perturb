import os
import sys
os.chdir('/home/jsun140/scratchyiqunchen/jsun140/Project-Perturb')
sys.path.insert(0, '/home/jsun140/scratchyiqunchen/jsun140/Project-Perturb/majority_vote')
import scanpy as sc
from DE_DOC_stats import compute_de_statistics
import argparse

ds_list = ['k562', 'rpe1', 'hepg2', 'jurkat']

parser = argparse.ArgumentParser()
parser.add_argument('i', type=int, choices=range(len(ds_list)))
args = parser.parse_args()

pred_ds = ds_list[args.i]
print(f'Processing {pred_ds} / test ..........................')

pred_dir = f'TranScouter/perturbqa_expr/prediction/{pred_ds}'
ad_for_de = sc.read_h5ad(f'{pred_dir}/{pred_ds}_test_pred.h5ad')
print(ad_for_de)

save_folder = f'TranScouter/perturbqa_expr/de_label/{pred_ds}'
os.makedirs(save_folder, exist_ok=True)

df = compute_de_statistics(
    adata=ad_for_de,
    control_key='ctrl',
    target_col='gene',
    gene_row='gene_name',
    log2fc_threshold=0.0,
    alpha=0.01,
)
df = df.assign(pred_ds=pred_ds, split_name='test')
df.to_csv(f'{save_folder}/{pred_ds}-test.csv', index=False)
print(f'Saved to {save_folder}/{pred_ds}-test.csv ({len(df)} rows)')
