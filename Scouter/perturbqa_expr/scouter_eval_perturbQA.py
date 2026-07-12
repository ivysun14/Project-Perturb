import os
import sys
os.chdir('/home/jsun140/scratchyiqunchen/jsun140/Project-Perturb')
sys.path.insert(0, '/home/jsun140/scratchyiqunchen/jsun140/Project-Perturb/majority_vote')
import scanpy as sc
from DE_DOC_stats import compute_de_statistics
import argparse

ds_list = ["rpe1", "jurkat", "hepg2", "k562"]

parser = argparse.ArgumentParser()
parser.add_argument("i", type=int, choices=range(len(ds_list)))
args = parser.parse_args()

ds = ds_list[args.i]
print(f'Processing {ds} ..........................')

pred_dir = f'Scouter/perturbqa_expr/prediction/{ds}'
ad_for_de = sc.read_h5ad(f'{pred_dir}/{ds}_test_pred.h5ad')
print(ad_for_de)
print(ad_for_de.var)
print(ad_for_de.obs)

save_folder = f'Scouter/perturbqa_expr/de_label/{ds}'
os.makedirs(save_folder, exist_ok=True)

df = compute_de_statistics(adata=ad_for_de,
                            control_key='ctrl',
                            target_col='gene',
                            gene_row='gene_name',
                            log2fc_threshold=0.0,
                            alpha=0.01)
df = df.assign(Dataset=ds, split_type='test')
df.to_csv(f'{save_folder}/{ds}-test.csv', index=False)
print(f"Saved DE labels to {save_folder}/{ds}-test.csv ({len(df)} rows)")
