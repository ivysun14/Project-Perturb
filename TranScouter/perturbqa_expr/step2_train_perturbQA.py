import os
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root, resolved from this file's location
import sys
import gc
import argparse
import random
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import pickle
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, 'ModifiedScouter_TranScouter/TranScouter'))
from majority_vote.analysis_utility import resolve_repeats
from transcouter import TranScouter, ScouterData

ds_list = ['k562', 'rpe1', 'hepg2', 'jurkat']
ds_filename = {
    'k562':   '/projects/ychen756/project_perturb_data/h5ad_objects/K562_raw.h5ad',
    'rpe1':   '/projects/ychen756/project_perturb_data/h5ad_objects/RPE1_raw.h5ad',
    'hepg2':  '/projects/ychen756/project_perturb_data/h5ad_objects/HepG2_raw.h5ad',
    'jurkat': '/projects/ychen756/project_perturb_data/h5ad_objects/Jurkat_raw.h5ad',
}
QA_DIR      = '/projects/ychen756/project_perturb_data/perturbqa_datasets'
GENE_UNIVERSE = '/projects/ychen756/project_perturb_data/common_gene_universe_QA.csv'

os.chdir(_REPO)
parser = argparse.ArgumentParser()
parser.add_argument('--pred_ds', type=str, required=True, choices=ds_list,
                    help='Held-out cell line for cross-condition prediction.')
parser.add_argument('--test_run', action='store_true',
                    help='Subset to small pert sets and cap epochs to 3 for a quick sanity check.')
args = parser.parse_args()

pred_ds  = args.pred_ds
train_ds = [ds for ds in ds_list if ds != pred_ds]
print(f"Held-out: {pred_ds}  |  Training on: {train_ds}")
if args.test_run:
    print("TEST RUN: subsetting perturbations and capping epochs to 3.")


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

def predict_and_save(pert_list, ctrl_arr, cell_line, raw_counts, real_source, split_name):
    pred_dict = {}
    for pert in pert_list:
        n = int(raw_counts.get(pert, 300))
        ctrl_idx = np.random.choice(len(ctrl_arr), size=n, replace=True)
        pred_df  = scouter_model.pred(
            pert_names=[pert] * n,
            ctrl_expr=ctrl_arr[ctrl_idx],
            covariate=[cell_line] * n,
        )
        pred_dict[pert] = pred_df.values

    dropped = set(pert_list) - set(pred_dict)
    if dropped:
        print(f"  WARNING: {len(dropped)} {split_name} perts dropped: {dropped}")
    else:
        print(f"  All {len(pert_list)} {split_name} perts predicted.")

    pred_arrays = [ctrl_arr]
    pert_labels = ['ctrl'] * ctrl_arr.shape[0]
    for pert, arr in pred_dict.items():
        pred_arrays.append(arr)
        pert_labels.extend([pert] * arr.shape[0])
    obs_df     = pd.DataFrame({'gene': pert_labels, 'cell_line': [cell_line] * len(pert_labels)})
    pred_adata = sc.AnnData(X=np.vstack(pred_arrays), obs=obs_df, var=scouter_model.all_adata.var)
    pred_adata.write(f'{results_dir}/{pred_ds}_{split_name}_pred.h5ad')

    real_mask  = (real_source.obs['cell_line'] == cell_line) & \
                 (real_source.obs['gene'].isin(list(pred_dict) + ['ctrl']))
    real_adata = real_source[real_mask].copy()
    real_adata.obs = real_adata.obs[['gene', 'cell_line']].copy()
    real_adata.var = scouter_model.all_adata.var
    real_adata.write(f'{results_dir}/{pred_ds}_{split_name}_real.h5ad')
    print(f"  {split_name}: pred={pred_adata.shape}, real={real_adata.shape}")


###################### Load Data ##########################
set_seeds()
qa_genes    = pd.read_csv(GENE_UNIVERSE)['gene_name'].tolist()
combined_ad = sc.read_h5ad(f'TranScouter/perturbqa_expr/training_data/combined_for_{pred_ds}.h5ad')
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


###################### Data Split ##########################
# test perts of held-out cell line (PerturbQA test split)
qa_de_pred = pd.read_csv(f'{QA_DIR}/{pred_ds}-de.csv')
test_pert   = qa_de_pred[qa_de_pred['split'] == 'test']['pert'].unique().tolist()

# all cov_perts from training cell lines (both train and test PerturbQA splits — no calib holdout)
present_cov_perts = sorted(
    pertdata.adata.obs.loc[pertdata.adata.obs['gene'] != 'ctrl', 'cov_pert'].unique()
)

if args.test_run:
    subset_tv = []
    for td in train_ds:
        subset_tv += [cp for cp in present_cov_perts if cp.startswith(f"{td}_")][:5]
    present_cov_perts = subset_tv
    test_pert = test_pert[:5]

train_conds, holdout = train_test_split(present_cov_perts, test_size=0.15, random_state=42)
val_conds, test_conds = train_test_split(holdout, test_size=0.33, random_state=42)
print(f"Split — train: {len(train_conds)}, val: {len(val_conds)}, dummy test: {len(test_conds)}, held-out test perts: {len(test_pert)}")

# test_conds is a dummy slice of the holdout — data_init needs a non-empty test_adata
# to build BalancedDataset; test_dataset is never used (we do our own prediction)
pertdata.split_Train_Val_Test(train_conds=train_conds, val_conds=val_conds, test_conds=test_conds)

raw_cell_counts = {td: pertdata.adata[pertdata.adata.obs['cell_line'] == td].obs['gene'].value_counts()
                   for td in train_ds}


###################### Model Training ##########################
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
print(scouter_model.network)
print(f"Total parameters: {sum(p.numel() for p in scouter_model.network.parameters()):,}")

scouter_model.train(
    batch_size=512,
    w_norm=0.0,
    w_direction=0.25,
    w_kld=0.0,
    lr=0.0001,
    n_epochs=3 if args.test_run else 50,
    if_nonzero=False,
)


###################### Save Checkpoint and Loss Curve ##########################
results_dir = f'TranScouter/perturbqa_expr/prediction/{pred_ds}'
os.makedirs(results_dir, exist_ok=True)

torch.save({
    'model_state_dict': scouter_model.network.state_dict(),
    'n_genes':          scouter_model.n_genes,
    'loss_history':     scouter_model.loss_history,
    'best_val_loss':    scouter_model.best_val_loss,
    'pred_ds':          pred_ds,
}, f'{results_dir}/{pred_ds}_transcouter_checkpoint.pt')

history = scouter_model.loss_history
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(history['train_loss'], label='Train')
if history['val_loss']:
    ax.plot(history['val_loss'], label='Val')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
suffix = '_testrun' if args.test_run else ''
ax.set_title(f'{pred_ds} TranScouter PerturbQA loss curve{suffix}')
ax.legend(); fig.tight_layout()
fig.savefig(f'{results_dir}/{pred_ds}_loss_curve{suffix}.png', dpi=150)
plt.close(fig)
print(f"Checkpoint + loss curve saved to {results_dir}/")


###################### Prediction ##########################
# test perts of held-out cell line (zero-shot cross-condition prediction)
print(f"\nPredicting test perts of held-out cell line ({pred_ds})")
_raw = scread_uniq_genes(pred_ds)
pred_ds_raw = _raw[:, qa_genes].copy()
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

test_pert_available = [p for p in test_pert if p in scouter_model.embd_idx_dict]
if len(test_pert_available) < len(test_pert):
    dropped = set(test_pert) - set(test_pert_available)
    print(f"  WARNING: {len(dropped)} test perts not in embedding, dropped: {dropped}")

predict_and_save(
    pert_list   = test_pert_available,
    ctrl_arr    = ctrl_arr_pred,
    cell_line   = pred_ds,
    raw_counts  = pred_ds_raw.obs['gene'].value_counts(),
    real_source = pred_ds_raw,
    split_name  = 'test',
)
