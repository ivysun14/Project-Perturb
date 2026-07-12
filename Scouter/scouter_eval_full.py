import os
import sys
os.chdir('/home/jsun140/scratchyiqunchen/jsun140/Project-Perturb')
sys.path.insert(0, '/home/jsun140/scratchyiqunchen/jsun140/Project-Perturb/majority_vote')
import scanpy as sc
from DE_DOC_stats import compute_de_statistics
import argparse
import itertools

ds_list = ["rpe1", "jurkat", "hepg2", "k562"]
split_type_list = ['test', 'calib']
params = list(itertools.product(ds_list, split_type_list))

parser = argparse.ArgumentParser()
parser.add_argument("i", type=int, choices=range(len(params)))
args = parser.parse_args()

ds, split_type = params[args.i]
print(f'Processing {ds}_{split_type} ..........................')

# where to find predicted expressions
pred_dir = f'Scouter/prediction/{ds}'
ad_for_de = sc.read_h5ad(f'{pred_dir}/{ds}_{split_type}_pred.h5ad')
print(ad_for_de)
print(ad_for_de.var)
print(ad_for_de.obs)

# where to save discretized DE labels
save_folder = f'Scouter/de_label/{ds}'
os.makedirs(save_folder, exist_ok=True)

df = compute_de_statistics(adata=ad_for_de,
                            control_key='ctrl',
                            target_col='gene',
                            gene_row='gene_name',
                            log2fc_threshold=0.0,
                            alpha=0.01)
df = df.assign(Dataset=ds, split_type=split_type)
df.to_csv(f'{save_folder}/{ds}-{split_type}.csv', index=False)
