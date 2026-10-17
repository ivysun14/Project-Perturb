import os
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root, resolved from this file's location
import sys
import pandas as pd
import scanpy as sc
import argparse
import pickle
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, _REPO)
from preprocessing.data_utility import resolve_repeats
from ModifiedScouter_TranScouter.scouter import Scouter, ScouterData


ds_list = ['k562', 'rpe1', 'hepg2', 'jurkat']
ds_filename = {
    'k562':   '/projects/ychen756/project_perturb_data/h5ad_objects/K562_raw.h5ad',
    'rpe1':   '/projects/ychen756/project_perturb_data/h5ad_objects/RPE1_raw.h5ad',
    'hepg2':  '/projects/ychen756/project_perturb_data/h5ad_objects/HepG2_raw.h5ad',
    'jurkat': '/projects/ychen756/project_perturb_data/h5ad_objects/Jurkat_raw.h5ad',
}
QA_DIR = '/projects/ychen756/project_perturb_data/perturbqa_datasets'

os.chdir(_REPO)
parser = argparse.ArgumentParser()
parser.add_argument("i", type=int, choices=range(len(ds_list)))
parser.add_argument("--test_run", action="store_true",
                    help="Subset to small perturbation sets (10 train / 5 test) and cap epochs to 3 for sanity check.")
args = parser.parse_args()


def scread_uniq_genes(ds):
    raw_ad = sc.read(f'{ds_filename[ds]}')
    var_df = raw_ad.var.copy()
    var_df['gene_id'] = var_df.index
    var_df['gene_name'] = var_df['gene_name'].astype(str)
    raw_ad.var = var_df
    raw_ad = resolve_repeats(raw_ad, gene_row='gene_name')
    raw_ad.var = raw_ad.var.set_index('gene_name')
    return raw_ad


ds = ds_list[args.i]
print(f"Processing {ds}..........................")

# train/test perturbation splits defined by PerturbQA split column
de_df = pd.read_csv(f'{QA_DIR}/{ds}-de.csv')
train_pert = de_df[de_df["split"] == "train"].pert.unique().tolist()
test_pert  = de_df[de_df["split"] == "test"].pert.unique().tolist()
# union of all genes observed in any PerturbQA pair for this cell line
qa_genes   = set(de_df["gene"].unique())
print(f"Train perts: {len(train_pert)}, Test perts: {len(test_pert)}, PerturbQA genes: {len(qa_genes)}")

if args.test_run:
    print("TEST RUN: subsetting to 10 train / 5 test perturbations.")
    train_pert = train_pert[:10]
    test_pert  = test_pert[:5]

all_perts = test_pert + train_pert + ["ctrl"]

# process h5ad
raw_ad = scread_uniq_genes(ds)
raw_ad.obs = raw_ad.obs[['gene', 'gene_id']]
raw_ad.obs['gene'] = raw_ad.obs['gene'].cat.rename_categories({"non-targeting": "ctrl"})
raw_ad.var = raw_ad.var[['gene_id']]
raw_ad.var.index.name = None
raw_ad.var['gene_name'] = raw_ad.var.index.astype(str)

# normalize
sc.pp.normalize_total(raw_ad, target_sum=1e4, key_added="size_factors")
sc.pp.log1p(raw_ad)

# embeddings
embd_path_1 = '/projects/ychen756/project_perturb_data/embedding/GenePT_gene_protein_embedding_model_3_text.pickle.'
embd_path_2 = '/projects/ychen756/project_perturb_data/embedding/additional_genePT_embeddings.pkl'
with open(embd_path_1, 'rb') as f:
    embd_1 = pd.DataFrame(pickle.load(f)).T
with open(embd_path_2, 'rb') as f:
    embd_2 = pd.DataFrame(pickle.load(f)).T
ctrl_row = pd.DataFrame([np.zeros(embd_1.shape[1])], columns=embd_1.columns, index=['ctrl'])
embd = pd.concat([ctrl_row, embd_1, embd_2], axis=0)

# restrict to PerturbQA perts (train + test + ctrl) and PerturbQA genes
filtered_ad = raw_ad[raw_ad.obs.gene.isin(all_perts)].copy()
filtered_ad = filtered_ad[:, filtered_ad.var.index.isin(qa_genes)].copy()
print(f"h5ad after subsetting: {filtered_ad.shape} (cells × genes)")

raw_cell_counts = filtered_ad.obs.gene.value_counts()

scouterdata = ScouterData(adata=filtered_ad,
                          embd=embd,
                          key_label='gene',
                          key_var_genename='gene_name')
scouterdata.setup_ad('embd_index')

# hold out only test perts (no calibration split in PerturbQA)
scouterdata.split_Train_Val_Test(seed=1, val_ratio=0.2, test_conds=test_pert)

# training
scouter_model = Scouter(scouterdata)
scouter_model.model_init()
print(scouter_model.network)
n_params = sum(p.numel() for p in scouter_model.network.parameters())
print(f"Total parameters: {n_params:,}")
if args.test_run:
    scouter_model.train(n_epochs=3)
else:
    scouter_model.train()

results_dir = f'Scouter/perturbqa_expr/prediction/{ds}'
os.makedirs(results_dir, exist_ok=True)

# save model checkpoint
checkpoint = {
    'model_state_dict': scouter_model.network.state_dict(),
    'n_genes': scouter_model.n_genes,
    'loss_history': scouter_model.loss_history,
    'best_val_loss': scouter_model.best_val_loss,
    'ds': ds,
}
ckpt_path = f'{results_dir}/{ds}_scouter_checkpoint.pt'
torch.save(checkpoint, ckpt_path)
print(f"Model checkpoint saved to {ckpt_path}")

# plot train / val loss curve
history = scouter_model.loss_history
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(history['train_loss'], label='Train')
if history['val_loss']:
    ax.plot(history['val_loss'], label='Val')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
suffix = '_testrun' if args.test_run else ''
ax.set_title(f'{ds} Scouter PerturbQA loss curve{suffix}')
ax.legend()
fig.tight_layout()
curve_path = f'{results_dir}/{ds}_loss_curve{suffix}.png'
fig.savefig(curve_path, dpi=150)
plt.close(fig)
print(f"Loss curve saved to {curve_path}")


def predict_and_save(pert_list, split_name, ctrl_array, ctrl_labels):
    pred_dict = {}
    for pert in pert_list:
        n = int(raw_cell_counts.get(pert, 300))
        pred_dict.update(scouter_model.pred([pert], n_pred=n))

    dropped = set(pert_list) - set(pred_dict.keys())
    if dropped:
        print(f"WARNING: {len(dropped)} {split_name} perturbations dropped: {dropped}")
    else:
        print(f"All {len(pert_list)} {split_name} perturbations predicted.")

    pred_arrays = [ctrl_array]
    pert_labels = ctrl_labels[:]
    for pert, arr in pred_dict.items():
        pred_arrays.append(arr)
        pert_labels.extend([pert] * arr.shape[0])

    pred_matrix = np.vstack(pred_arrays)
    obs_df = pd.DataFrame({'gene': pert_labels})
    pred_adata = sc.AnnData(X=pred_matrix, obs=obs_df, var=scouterdata.test_adata.var)
    pred_adata.write(f'{results_dir}/{ds}_{split_name}_pred.h5ad')

    true_adata = scouterdata.test_adata[scouterdata.test_adata.obs.gene.isin(pert_list + ['ctrl'])].copy()
    true_adata.obs = true_adata.obs[obs_df.columns]
    true_adata.write(f'{results_dir}/{ds}_{split_name}_real.h5ad')


ctrl_adata = scouterdata.test_adata[scouterdata.test_adata.obs.gene == 'ctrl']
ctrl_array = ctrl_adata.X.toarray() if hasattr(ctrl_adata.X, 'toarray') else ctrl_adata.X
ctrl_labels = ['ctrl'] * ctrl_array.shape[0]

# predict only the test split (no calibration in PerturbQA)
predict_and_save(test_pert, 'test', ctrl_array, ctrl_labels)
