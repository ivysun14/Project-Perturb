import os
import sys
os.chdir('/home/jsun140/scratchyiqunchen/jsun140/Project-Perturb')
sys.path.insert(0, '/home/jsun140/scratchyiqunchen/jsun140/Project-Perturb/majority_vote')
import scanpy as sc
from DE_DOC_stats import compute_de_statistics
import argparse

ds_list = ['k562', 'rpe1', 'hepg2', 'jurkat']

# 16 jobs: for each pred_ds → test split + 3 calib splits (one per training cell line)
params = []
for pred_ds in ds_list:
    train_ds = [ds for ds in ds_list if ds != pred_ds]
    params.append((pred_ds, 'test'))
    for td in train_ds:
        params.append((pred_ds, f'{td}_calib'))

parser = argparse.ArgumentParser()
parser.add_argument('i', type=int, choices=range(len(params)))
args = parser.parse_args()

pred_ds, split_name = params[args.i]
print(f'Processing {pred_ds} / {split_name} ..........................')

pred_dir = f'TranScouter/prediction/{pred_ds}'
ad_for_de = sc.read_h5ad(f'{pred_dir}/{pred_ds}_{split_name}_pred.h5ad')
print(ad_for_de)

save_folder = f'TranScouter/de_label/{pred_ds}'
os.makedirs(save_folder, exist_ok=True)

df = compute_de_statistics(
    adata=ad_for_de,
    control_key='ctrl',
    target_col='gene',
    gene_row='gene_name',
    log2fc_threshold=0.0,
    alpha=0.01,
)
df = df.assign(pred_ds=pred_ds, split_name=split_name)
df.to_csv(f'{save_folder}/{pred_ds}-{split_name}.csv', index=False)
print(f'Saved to {save_folder}/{pred_ds}-{split_name}.csv')
