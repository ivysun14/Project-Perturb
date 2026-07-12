import os
import sys
import argparse
import pandas as pd
import scanpy as sc

os.chdir('/home/jsun140/scratchyiqunchen/jsun140/Project-Perturb')
sys.path.insert(0, 'majority_vote')
from DE_DOC_stats import compute_de_statistics

parser = argparse.ArgumentParser()
parser.add_argument('--pred_ds', required=True, choices=['k562', 'rpe1', 'hepg2', 'jurkat'])
args = parser.parse_args()

pred_ds = args.pred_ds
out_dir = f'STATE/results/{pred_ds}'

gene_universe = pd.read_csv('STATE/training_data/common_gene_universe.csv', index_col=0)


def load_pred(path):
    adata = sc.read_h5ad(path)
    # var_names are integer positional indexes; map to gene names via common_gene_universe.csv
    adata.var['gene_name'] = gene_universe['gene_name'].values
    return adata


## Task 1: test perts of held-out cell line
print(f"\n{'='*60}")
print(f"Task 1 — test perts of held-out cell line ({pred_ds.upper()})")
print('='*60)

adata_t1 = load_pred(f'{out_dir}/eval_best.ckpt/adata_pred.h5ad')
print(f"Loaded: {adata_t1.n_obs} cells x {adata_t1.n_vars} genes")
df_t1 = compute_de_statistics(
    adata=adata_t1,
    control_key='non-targeting',
    target_col='gene',
    gene_row='gene_name',
    alpha=0.01,
    log2fc_threshold=0.0,
)
df_t1.insert(0, 'cell_line', pred_ds)
out_t1 = f'{out_dir}/de_task1_pred.csv'
df_t1.to_csv(out_t1, index=False)
print(f"Saved {len(df_t1)} rows → {out_t1}")


# Task 2: calib perts of 3 training cell lines
print(f"\n{'='*60}")
print(f"Task 2 — calib perts of training cell lines (held-out: {pred_ds.upper()})")
print('='*60)

adata_t2 = load_pred(f'{out_dir}/eval_calib.ckpt/adata_pred.h5ad')
print(f"Loaded: {adata_t2.n_obs} cells x {adata_t2.n_vars} genes")
train_ds = [ds for ds in ['k562', 'rpe1', 'hepg2', 'jurkat'] if ds != pred_ds]
dfs_t2 = []
for td in train_ds:
    # Subset to this cell line's perturbed + control cells
    sub = adata_t2[adata_t2.obs['cell_line'] == td].copy()
    n_pert = (sub.obs['gene'] != 'non-targeting').sum()
    n_ctrl = (sub.obs['gene'] == 'non-targeting').sum()
    print(f"\n  [{td}] {n_pert} perturbed cells, {n_ctrl} control cells")

    df = compute_de_statistics(
        adata=sub,
        control_key='non-targeting',
        target_col='gene',
        gene_row='gene_name',
        alpha=0.01,
        log2fc_threshold=0.0,
    )
    df.insert(0, 'cell_line', td)
    dfs_t2.append(df)
df_t2 = pd.concat(dfs_t2, ignore_index=True)
out_t2 = f'{out_dir}/de_task2_pred.csv'
df_t2.to_csv(out_t2, index=False)
print(f"\nSaved {len(df_t2)} rows → {out_t2}")