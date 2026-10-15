import subprocess
import os
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root, resolved from this file's location
import sys
import json
import argparse
import toml
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import anndata as ad

os.chdir(_REPO)
sys.path.insert(0, 'majority_vote')
sys.path.insert(0, '.')
from analysis_utility import resolve_repeats
from DE_DOC_stats import compute_de_statistics

parser = argparse.ArgumentParser()
parser.add_argument('--pred_ds', required=True, choices=['k562', 'rpe1', 'hepg2', 'jurkat'])
args = parser.parse_args()
pred_ds = args.pred_ds
ds_list  = ['k562', 'rpe1', 'hepg2', 'jurkat']
train_ds = [ds for ds in ds_list if ds != pred_ds]
out_dir  = f'STATE/results/{pred_ds}'

ds_csv = {
    'k562':   '/projects/ychen756/project_perturb_data/csv/DE_DOC_K562_splitted.csv',
    'rpe1':   '/projects/ychen756/project_perturb_data/csv/DE_DOC_RPE1_splitted.csv',
    'hepg2':  '/projects/ychen756/project_perturb_data/csv/DE_DOC_HepG2_splitted.csv',
    'jurkat': '/projects/ychen756/project_perturb_data/csv/DE_DOC_Jurkat_splitted.csv',
}
ds_h5ad = {
    'k562':   '/projects/ychen756/project_perturb_data/h5ad_objects/K562_raw.h5ad',
    'rpe1':   '/projects/ychen756/project_perturb_data/h5ad_objects/RPE1_raw.h5ad',
    'hepg2':  '/projects/ychen756/project_perturb_data/h5ad_objects/HepG2_raw.h5ad',
    'jurkat': '/projects/ychen756/project_perturb_data/h5ad_objects/Jurkat_raw.h5ad',
}

# common gene universe
gene_universe_df = pd.read_csv('STATE/training_data/common_gene_universe.csv', index_col=0)
gene_universe    = gene_universe_df['gene_name'].tolist()   # alphabetically sorted, 6641 genes

# calib perts of pred_ds + dropped filter
split_df   = pd.read_csv(ds_csv[pred_ds])
calib_pert = split_df[split_df['split_calib'] == 'calibration']['pert'].unique().tolist()
print(f"Calib perts of {pred_ds}: {len(calib_pert)}")

# seen set = train+test of 3 training cell lines (same logic as dropped_test_perts)
seen = set()
for td in train_ds:
    td_df = pd.read_csv(ds_csv[td])
    # |= : in-place set union operator, add all elements from the right-hand set into seen
    seen |= set(td_df[td_df['split_calib'].isin(['train', 'test'])]['pert'].unique())

dropped_calib_held_out = set(calib_pert) - seen
calib_pert_filtered    = [p for p in calib_pert if p not in dropped_calib_held_out]
print(f"  Dropped (unseen): {len(dropped_calib_held_out)}  →  kept: {len(calib_pert_filtered)}")

# append dropped calib perts to dropped_test_perts.json
dropped_json_path = 'STATE/pert_coverage/dropped_test_perts.json'
with open(dropped_json_path) as f:
    dropped_json = json.load(f)
existing = set(dropped_json[pred_ds])
new_entries = sorted(dropped_calib_held_out - existing)
dropped_json[pred_ds] = sorted(existing | dropped_calib_held_out)
with open(dropped_json_path, 'w') as f:
    json.dump(dropped_json, f, indent=2)
print(f"  Added {len(new_entries)} new entries to dropped_test_perts.json['{pred_ds}']: {new_entries}")

# build h5ad for pred_ds calib perts + controls
print(f"\nBuilding h5ad for {pred_ds} calib perts ...")
raw_ad = sc.read(ds_h5ad[pred_ds])
var_df = raw_ad.var.copy()
var_df['gene_id']   = var_df.index
var_df['gene_name'] = var_df['gene_name'].astype(str)
raw_ad.var = var_df
raw_ad = resolve_repeats(raw_ad, gene_row='gene_name')
raw_ad.var = raw_ad.var.set_index('gene_name')
raw_ad.obs['cell_line'] = pred_ds

mask = raw_ad.obs['gene'].isin(calib_pert_filtered + ['non-targeting'])
sub  = raw_ad[mask][:, gene_universe].copy()
sub.var = sub.var[['gene_id']]

sc.pp.normalize_total(sub, target_sum=1e4, key_added='size_factors')
sc.pp.log1p(sub)
if not sp.issparse(sub.X):
    sub.X = sp.csr_matrix(sub.X)
sub.X = sub.X.astype(np.float32)
sub.obs['cell_barcode'] = sub.obs_names

# save calib-only h5ad (used for cell counts; not passed to STATE directly)
calib_only_h5ad = f'STATE/training_data/task3_calib_{pred_ds}.h5ad'
sub.write_h5ad(calib_only_h5ad)
print(f"  Saved calib-only {sub.n_obs} cells → {calib_only_h5ad}")

# append calib pert cell counts to pred_task1_cell_counts_{pred_ds}.csv
counts_csv = f'STATE/training_data/pred_task1_cell_counts_{pred_ds}.csv'
calib_counts = (
    sub.obs[sub.obs['gene'].isin(calib_pert_filtered)]
    .groupby('gene', observed=True).size()
    .reset_index(name='n_cells')
    .rename(columns={'gene': 'pert'})
)
existing_counts = pd.read_csv(counts_csv)
existing_perts  = set(existing_counts['pert'])
new_counts = calib_counts[~calib_counts['pert'].isin(existing_perts)]
pd.concat([existing_counts, new_counts], ignore_index=True).to_csv(counts_csv, index=False)
print(f"  Appended {len(new_counts)} calib pert counts → {counts_csv}")

# combine training h5ad + calib cells so pert_encoder shape matches the checkpoint (2396 perts)
# calib_pert_filtered are all seen in training cell lines, so no new perts are introduced
print(f"\nConcatenating training h5ad with calib cells for prediction ...")
training_h5ad_path = f'STATE/training_data/combined_for_{pred_ds}_full.h5ad'
adata_train = ad.read_h5ad(training_h5ad_path)
task3_combined = ad.concat([adata_train, sub], join='inner')
task3_combined.obs_names_make_unique()
print(f"  Training cells: {adata_train.n_obs}  +  calib cells: {sub.n_obs}  =  {task3_combined.n_obs} total")
print(f"  Unique perts in combined: {task3_combined.obs['gene'].nunique()}")

task3_h5ad = f'STATE/training_data/task3_combined_{pred_ds}.h5ad'
task3_combined.write_h5ad(task3_h5ad)
print(f"  Saved combined h5ad → {task3_h5ad}")

# TOML for task3 prediction
toml_task3_file = f'STATE/training_data/{pred_ds}_task3.toml'
config_task3 = {
    'datasets': {'combined': task3_h5ad},
    'training': {'combined': 'train'},
    'zeroshot': {},
    'fewshot':  {f'combined.{pred_ds}': {'test': calib_pert_filtered}},
}
with open(toml_task3_file, 'w') as f:
    toml.dump(config_task3, f)

# Symlink task3.ckpt → best.ckpt so output goes to eval_task3.ckpt/
ckpt_dir   = f'{out_dir}/checkpoints'
task3_ckpt = os.path.join(ckpt_dir, 'task3.ckpt')
if os.path.islink(task3_ckpt) or os.path.exists(task3_ckpt):
    os.remove(task3_ckpt)
os.symlink('best.ckpt', task3_ckpt)

# make prediction
print('\nRunning state tx predict for task3 ...')
env = os.environ.copy()
env['WANDB_MODE'] = 'disabled'
subprocess.run(
    [
        'state', 'tx', 'predict',
        '--output-dir', out_dir,
        '--toml',       toml_task3_file,
        '--checkpoint', 'task3.ckpt',
        '--profile',    'anndata',
        '--predict-only',
    ],
    check=True, env=env,
)

# DE on task3 predictions
print('\nRunning DE for task3 ...')
adata_t3 = ad.read_h5ad(f'{out_dir}/eval_task3.ckpt/adata_pred.h5ad')
adata_t3.var['gene_name'] = gene_universe_df['gene_name'].values

df_t3 = compute_de_statistics(
    adata=adata_t3,
    control_key='non-targeting',
    target_col='gene',
    gene_row='gene_name',
    alpha=0.01,
    log2fc_threshold=0.0,
)
df_t3.insert(0, 'cell_line', pred_ds)
df_t3.to_csv(f'{out_dir}/de_task3_pred.csv', index=False)
print(f"  Saved DE → {out_dir}/de_task3_pred.csv")


print('\nMerging task3 into task1 ...')

# Merge anndata
# Strip controls from task3 objects — task1 already has non-targeting cells; keeping both would duplicate them
adata_t1_pred = ad.read_h5ad(f'{out_dir}/eval_best.ckpt/adata_pred.h5ad')
adata_t1_real = ad.read_h5ad(f'{out_dir}/eval_best.ckpt/adata_real.h5ad')
adata_t3_real = ad.read_h5ad(f'{out_dir}/eval_task3.ckpt/adata_real.h5ad')

adata_t3_pert      = adata_t3[adata_t3.obs['gene'] != 'non-targeting'].copy()
adata_t3_real_pert = adata_t3_real[adata_t3_real.obs['gene'] != 'non-targeting'].copy()
print(f"  task1 pred cells: {adata_t1_pred.n_obs}  |  task3 pert-only cells: {adata_t3_pert.n_obs}")
print(f"  task1 real cells: {adata_t1_real.n_obs}  |  task3 real pert-only cells: {adata_t3_real_pert.n_obs}")

merged_pred = ad.concat([adata_t1_pred, adata_t3_pert], join='inner')
merged_real = ad.concat([adata_t1_real, adata_t3_real_pert], join='inner')
merged_pred.obs_names_make_unique()
merged_real.obs_names_make_unique()

os.makedirs(f'{out_dir}/eval_task1_merged.ckpt', exist_ok=True)
merged_pred.write_h5ad(f'{out_dir}/eval_task1_merged.ckpt/adata_pred.h5ad')
merged_real.write_h5ad(f'{out_dir}/eval_task1_merged.ckpt/adata_real.h5ad')
print(f"  Merged pred: {merged_pred.n_obs} cells → eval_task1_merged.ckpt/adata_pred.h5ad")
print(f"  Merged real: {merged_real.n_obs} cells → eval_task1_merged.ckpt/adata_real.h5ad")

# Merge DE
# task1 test perts and task3 calib perts are non-overlapping by split design; verify before concat
de_t1 = pd.read_csv(f'{out_dir}/de_task1_pred.csv')
overlap = set(de_t1['pert']) & set(df_t3['pert'])
assert len(overlap) == 0, f"Unexpected pert overlap between task1 and task3 DE: {overlap}"
de_merged = pd.concat([de_t1, df_t3], ignore_index=True)
de_merged.to_csv(f'{out_dir}/de_task1_merged_pred.csv', index=False)
print(f"  Merged DE: {len(de_t1)} task1 + {len(df_t3)} task3 = {len(de_merged)} rows → de_task1_merged_pred.csv")
