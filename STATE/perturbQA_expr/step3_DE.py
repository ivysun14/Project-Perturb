"""
step3_DE.py
===========
Perform DE classification on STATE-predicted AnnData.
"""

import os
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root, resolved from this file's location
import sys
import argparse
import pandas as pd
import scanpy as sc

os.chdir(_REPO)
sys.path.insert(0, 'majority_vote')
from DE_DOC_stats import compute_de_statistics

parser = argparse.ArgumentParser()
parser.add_argument('--pred_ds', required=True, choices=['k562', 'rpe1', 'hepg2', 'jurkat'])
args = parser.parse_args()

pred_ds = args.pred_ds
out_dir = f'STATE/perturbQA_expr/results/{pred_ds}'
gene_universe = pd.read_csv('/projects/ychen756/project_perturb_data/common_gene_universe_QA.csv')


def load_pred(path):
    adata = sc.read_h5ad(path)
    adata.var['gene_name'] = gene_universe['gene_name'].values
    return adata


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
out_t1 = f'{out_dir}/de_pred.csv'
df_t1.to_csv(out_t1, index=False)
print(f"Saved {len(df_t1)} rows -> {out_t1}")
