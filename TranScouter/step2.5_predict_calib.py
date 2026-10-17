"""
step2.5_predict_calib.py
========================
Use the saved TranScouter checkpoint to make predictions on the 20% calibration
perts of the held-out test cell line, then merge with the existing test predictions.

Steps
-----
1. Restore TranScouter from checkpoint (same data/model setup as step2_train_full.py,
   skip training).
2. Predict on calib perts of pred_ds → {pred_ds}_calib_pred.h5ad / _real.h5ad
3. Run DE on calib predictions → de_label/{pred_ds}/{pred_ds}-calib.csv
4. Merge calib + test h5ads (ctrl stripped from calib to avoid duplication)
   → {pred_ds}_testcalib_pred.h5ad / _testcalib_real.h5ad
5. Merge calib + test DE CSVs → de_label/{pred_ds}/{pred_ds}-testcalib.csv

Usage
-----
    python TranScouter/step2.5_predict_calib.py --pred_ds hepg2
"""

import os
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root, resolved from this file's location
import sys
import gc
import argparse
import random
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scipy.sparse as sp
import pickle
import torch
from sklearn.model_selection import train_test_split

sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, 'ModifiedScouter_TranScouter/TranScouter'))
sys.path.insert(0, os.path.join(_REPO, 'majority_vote'))
from preprocessing.data_utility import resolve_repeats
from preprocessing.DE_DOC_stats import compute_de_statistics
from transcouter import TranScouter, ScouterData

ds_list = ['k562', 'rpe1', 'hepg2', 'jurkat']
ds_filename = {
    'k562':   '/projects/ychen756/project_perturb_data/h5ad_objects/K562_raw.h5ad',
    'rpe1':   '/projects/ychen756/project_perturb_data/h5ad_objects/RPE1_raw.h5ad',
    'hepg2':  '/projects/ychen756/project_perturb_data/h5ad_objects/HepG2_raw.h5ad',
    'jurkat': '/projects/ychen756/project_perturb_data/h5ad_objects/Jurkat_raw.h5ad',
}
ds_csv = {
    'k562':   '/projects/ychen756/project_perturb_data/csv/DE_DOC_K562_splitted.csv',
    'rpe1':   '/projects/ychen756/project_perturb_data/csv/DE_DOC_RPE1_splitted.csv',
    'hepg2':  '/projects/ychen756/project_perturb_data/csv/DE_DOC_HepG2_splitted.csv',
    'jurkat': '/projects/ychen756/project_perturb_data/csv/DE_DOC_Jurkat_splitted.csv',
}

os.chdir(_REPO)
parser = argparse.ArgumentParser()
parser.add_argument('--pred_ds', required=True, choices=ds_list)
args = parser.parse_args()

pred_ds  = args.pred_ds
train_ds = [ds for ds in ds_list if ds != pred_ds]
results_dir = f'TranScouter/prediction/{pred_ds}'
de_dir      = f'TranScouter/de_label/{pred_ds}'
os.makedirs(de_dir, exist_ok=True)

print(f"Held-out: {pred_ds}  |  Training cell lines: {train_ds}")


def set_seeds(seed=24):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def scread_uniq_genes(ds):
    raw_ad = sc.read(ds_filename[ds])
    var_df = raw_ad.var.copy()
    var_df['gene_id'] = var_df.index
    var_df['gene_name'] = var_df['gene_name'].astype(str)
    raw_ad.var = var_df
    raw_ad = resolve_repeats(raw_ad, gene_row='gene_name')
    raw_ad.var = raw_ad.var.set_index('gene_name')
    raw_ad.obs['cell_line'] = ds
    raw_ad.obs['gene'] = raw_ad.obs['gene'].cat.rename_categories({'non-targeting': 'ctrl'})
    return raw_ad

def to_array(X):
    return X.toarray() if sp.issparse(X) else np.array(X)


###################### Load Data (same as training) ############################
set_seeds()
combined_ad  = sc.read_h5ad(f'TranScouter/training_data/combined_for_{pred_ds}.h5ad')
common_genes = combined_ad.var_names.tolist()
print(f"Combined AnnData loaded: {combined_ad.shape}")

embd_path_1 = '/projects/ychen756/project_perturb_data/embedding/GenePT_gene_protein_embedding_model_3_text.pickle.'
embd_path_2 = '/projects/ychen756/project_perturb_data/embedding/additional_genePT_embeddings.pkl'
with open(embd_path_1, 'rb') as f:
    embd_1 = pd.DataFrame(pickle.load(f)).T
with open(embd_path_2, 'rb') as f:
    embd_2 = pd.DataFrame(pickle.load(f)).T
ctrl_row = pd.DataFrame([np.zeros(embd_1.shape[1])], columns=embd_1.columns, index=['ctrl'])
embd = pd.concat([ctrl_row, embd_1, embd_2], axis=0)

pertdata = ScouterData(
    adata=combined_ad,
    embd=embd,
    key_pert='gene',
    key_cov='cell_line',
    key_var_gnames='gene_name',
    ctrl_value='ctrl',
)
pertdata.setup_ad('embd_index', slim=False)
pertdata.key_cov_pert       = 'cov_pert'
pertdata.key_uns_DegNames   = None
pertdata.key_uns_DegIdx     = None
pertdata.key_uns_NonzeroIdx = None
pertdata.adata.obs['cov_pert'] = pertdata.adata.obs[['cell_line', 'gene']].agg('_'.join, axis=1)
del combined_ad, embd


###################### Data Split (same as training) ###########################
split_df   = {ds: pd.read_csv(ds_csv[ds]) for ds in ds_list}
calib_pert = {td: split_df[td][split_df[td]['split_calib'] == 'calibration']['pert'].unique().tolist()
              for td in train_ds}

# calib perts of the held-out cell line
calib_pert_pred_ds = split_df[pred_ds][split_df[pred_ds]['split_calib'] == 'calibration']['pert'].unique().tolist()
print(f"Calib perts of {pred_ds}: {len(calib_pert_pred_ds)}")

present_cov_perts = set(pertdata.adata.obs.loc[pertdata.adata.obs['gene'] != 'ctrl', 'cov_pert'].unique())
test_conds      = [f"{td}_{p}" for td in train_ds for p in calib_pert[td] if f"{td}_{p}" in present_cov_perts]
train_val_conds = [cp for cp in present_cov_perts if cp not in set(test_conds)]
train_conds, val_conds = train_test_split(train_val_conds, test_size=0.1, random_state=42)

pertdata.split_Train_Val_Test(train_conds=train_conds, val_conds=val_conds, test_conds=test_conds)


###################### Restore Model from Checkpoint ##########################
scouter_model = TranScouter(pertdata)
scouter_model.data_init(key_stratify=['cell_line'])
scouter_model.model_init(
    hidden_enc_embd=(512, 256),
    hidden_enc_ctrl=(2048, 1024),
    bottle_dim=512,
    hidden_dec=(1024, 2048),
    use_batch_norm=True,
    use_layer_norm=False,
    dropout_rate=0.0,
    use_sampling=False,
    condition='control',
)

ckpt_path = f'{results_dir}/{pred_ds}_transcouter_checkpoint.pt'
print(f"\nLoading checkpoint from {ckpt_path} ...")
ckpt = torch.load(ckpt_path, map_location='cpu')
scouter_model.network.load_state_dict(ckpt['model_state_dict'])
scouter_model.network.eval()
print("Checkpoint loaded.")


###################### Load Pred-DS Raw Data ##################################
print(f"\nLoading {pred_ds} raw data for ctrl and calib cells ...")
_raw = scread_uniq_genes(pred_ds)
pred_ds_raw = _raw[:, common_genes].copy()
del _raw; gc.collect()
pred_ds_raw.obs = pred_ds_raw.obs[['gene', 'gene_id', 'cell_line']].copy()
pred_ds_raw.var = pred_ds_raw.var[['gene_id']].copy()
pred_ds_raw.var.index.name = None
pred_ds_raw.var['gene_name'] = pred_ds_raw.var.index.astype(str)
sc.pp.normalize_total(pred_ds_raw, target_sum=1e4, key_added='size_factors')
sc.pp.log1p(pred_ds_raw)

ctrl_arr_pred = to_array(pred_ds_raw[pred_ds_raw.obs['gene'] == 'ctrl'].X)
if len(ctrl_arr_pred) == 0:
    raise ValueError(f"No ctrl cells found in {pred_ds}.")
print(f"  Ctrl cells: {len(ctrl_arr_pred)}")


###################### Predict on Calib Perts of Pred-DS ######################
# Filter to perts reachable through the embedding (same check as for test perts)
calib_pert_available = [p for p in calib_pert_pred_ds if p in scouter_model.embd_idx_dict]
dropped = set(calib_pert_pred_ds) - set(calib_pert_available)
if dropped:
    print(f"  WARNING: {len(dropped)} calib perts not in embedding, dropped: {dropped}")
print(f"Predicting {len(calib_pert_available)} calib perts for {pred_ds} ...")

raw_counts = pred_ds_raw.obs['gene'].value_counts()
pred_dict = {}
for pert in calib_pert_available:
    n = int(raw_counts.get(pert, 300))
    ctrl_idx = np.random.choice(len(ctrl_arr_pred), size=n, replace=True)
    pred_df  = scouter_model.pred(
        pert_names=[pert] * n,
        ctrl_expr=ctrl_arr_pred[ctrl_idx],
        covariate=[pred_ds] * n,
    )
    pred_dict[pert] = pred_df.values

# Pred h5ad: ctrl + predicted perts
pred_arrays = [ctrl_arr_pred]
pert_labels = ['ctrl'] * ctrl_arr_pred.shape[0]
for pert, arr in pred_dict.items():
    pred_arrays.append(arr)
    pert_labels.extend([pert] * arr.shape[0])
obs_df     = pd.DataFrame({'gene': pert_labels, 'cell_line': [pred_ds] * len(pert_labels)})
calib_pred_ad = sc.AnnData(X=np.vstack(pred_arrays), obs=obs_df, var=scouter_model.all_adata.var)
calib_pred_ad.write(f'{results_dir}/{pred_ds}_calib_pred.h5ad')
print(f"  Saved calib pred: {calib_pred_ad.shape}")

# Real h5ad: slice from pred_ds_raw
real_mask  = pred_ds_raw.obs['gene'].isin(list(pred_dict) + ['ctrl'])
calib_real_ad = pred_ds_raw[real_mask].copy()
calib_real_ad.obs = calib_real_ad.obs[['gene', 'cell_line']].copy()
calib_real_ad.var = scouter_model.all_adata.var
calib_real_ad.write(f'{results_dir}/{pred_ds}_calib_real.h5ad')
print(f"  Saved calib real: {calib_real_ad.shape}")


###################### DE on Calib Predictions ################################
print(f"\nRunning DE on calib predictions ...")
df_calib = compute_de_statistics(
    adata=calib_pred_ad,
    control_key='ctrl',
    target_col='gene',
    gene_row='gene_name',
    alpha=0.01,
    log2fc_threshold=0.0,
)
df_calib = df_calib.assign(pred_ds=pred_ds, split_name='calib')
df_calib.to_csv(f'{de_dir}/{pred_ds}-calib.csv', index=False)
print(f"  Saved DE → {de_dir}/{pred_ds}-calib.csv  ({len(df_calib)} rows)")


###################### Merge Calib + Test #####################################
print(f"\nMerging calib + test ...")

# Load existing test objects
test_pred_ad = sc.read_h5ad(f'{results_dir}/{pred_ds}_test_pred.h5ad')
test_real_ad = sc.read_h5ad(f'{results_dir}/{pred_ds}_test_real.h5ad')

# Strip ctrl from calib objects — test already contains ctrl cells
calib_pred_pert = calib_pred_ad[calib_pred_ad.obs['gene'] != 'ctrl'].copy()
calib_real_pert = calib_real_ad[calib_real_ad.obs['gene'] != 'ctrl'].copy()
print(f"  test pred: {test_pred_ad.n_obs}  |  calib pert-only pred: {calib_pred_pert.n_obs}")
print(f"  test real: {test_real_ad.n_obs}  |  calib pert-only real: {calib_real_pert.n_obs}")

# Verify no pert overlap between test and calib splits
test_perts  = set(test_pred_ad.obs.loc[test_pred_ad.obs['gene'] != 'ctrl', 'gene'])
calib_perts = set(calib_pred_pert.obs['gene'])
overlap = test_perts & calib_perts
if overlap:
    print(f"  WARNING: {len(overlap)} perts appear in both test and calib — check split integrity: {overlap}")
else:
    print(f"  No pert overlap between test and calib splits. ✓")

merged_pred = ad.concat([test_pred_ad, calib_pred_pert], join='inner')
merged_pred.var = test_pred_ad.var.loc[merged_pred.var_names]
merged_real = ad.concat([test_real_ad, calib_real_pert], join='inner')
merged_real.var = test_real_ad.var.loc[merged_real.var_names]
merged_pred.obs_names_make_unique()
merged_real.obs_names_make_unique()

merged_pred.write_h5ad(f'{results_dir}/{pred_ds}_testcalib_pred.h5ad')
merged_real.write_h5ad(f'{results_dir}/{pred_ds}_testcalib_real.h5ad')
print(f"  Merged pred: {merged_pred.n_obs} cells → {pred_ds}_testcalib_pred.h5ad")
print(f"  Merged real: {merged_real.n_obs} cells → {pred_ds}_testcalib_real.h5ad")

# Merge DE CSVs
df_test = pd.read_csv(f'{de_dir}/{pred_ds}-test.csv')
overlap_de = set(df_test['pert']) & set(df_calib['pert'])
if overlap_de:
    print(f"  WARNING: {len(overlap_de)} perts appear in both test and calib DE — unexpected.")
de_merged = pd.concat([df_test, df_calib], ignore_index=True)
de_merged.to_csv(f'{de_dir}/{pred_ds}-testcalib.csv', index=False)
print(f"  Merged DE: {len(df_test)} test + {len(df_calib)} calib = {len(de_merged)} rows → {pred_ds}-testcalib.csv")

print(f"\nDone. All outputs written to {results_dir}/ and {de_dir}/")
