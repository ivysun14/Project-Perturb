"""
step2_train.py
==============
Adapted from STATE/step2_train_full.py for the perturbQA subset.
"""

import subprocess
import os
import json
import argparse
import toml
import pandas as pd
from sklearn.model_selection import train_test_split

os.chdir('/home/jsun140/scratchyiqunchen/jsun140/Project-Perturb')

parser = argparse.ArgumentParser()
parser.add_argument('--pred_ds', type=str, required=True, choices=['k562', 'rpe1', 'hepg2', 'jurkat'])
parser.add_argument('--test-run', action='store_true', help='Subsample perts and run 10 steps to verify pipeline end-to-end')
args = parser.parse_args()

pred_ds  = args.pred_ds
test_run = args.test_run
print(f"------------- Dataset selected: {pred_ds} -------------")
if test_run:
    print("*** TEST RUN: subsampled perts, 10 steps ***")

DS_LIST  = ['k562', 'rpe1', 'hepg2', 'jurkat']
train_ds = [ds for ds in DS_LIST if ds != pred_ds]
QA_DIR         = '/home/jsun140/scratchyiqunchen/project_perturb_data/perturbqa_datasets'
COVERAGE_DIR   = 'STATE/perturbQA_expr/pert_coverage'
TRAIN_DATA_DIR = 'STATE/perturbQA_expr/training_data'
RESULTS_DIR = f'STATE/perturbQA_expr/results/{pred_ds}'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load split information
split_df = pd.read_csv(f'{QA_DIR}/{pred_ds}-de.csv')
train_perts = split_df[split_df['split'] == 'train']['pert'].unique().tolist()
test_perts = split_df[split_df['split'] == 'test' ]['pert'].unique().tolist()

# val = random 10% of train perts (same convention as full experiment)
train_perts, val_perts = train_test_split(train_perts, test_size=0.1, random_state=305)
print(f"Train: {len(train_perts)}, Val: {len(val_perts)}, Test: {len(test_perts)}")

if test_run:
    train_perts = train_perts[:20]
    val_perts   = val_perts[:5]
    test_perts  = test_perts[:5]
    print(f"Subsampled -> Train: {len(train_perts)}, Val: {len(val_perts)}, Test: {len(test_perts)}")

# filter test perts to match what step1 included in the combined h5ad
with open(f'{COVERAGE_DIR}/dropped_test_perts.json') as f:
    dropped_test_set = set(json.load(f).get(pred_ds, []))
test_perts = [p for p in test_perts if p not in dropped_test_set]
print(f"Test perts after dropping unseen: {len(test_perts)}")

# TOML for training
# NOTE: The combined h5ad has no train/test labels; the TOML is the sole source of
# that assignment.  'training: {combined: train}' makes STATE treat every cell
# in the h5ad as training data by default.  The 'fewshot' block then overrides
# specific perts of the held-out cell line: val_perts and test_perts
# are withheld from the training loop.  All other cells — the 3 training cell
# lines (all perts) and the held-out cell line's train perts — remain in training.
training_data_path = f'{TRAIN_DATA_DIR}/combined_for_{pred_ds}_perturbqa.h5ad'
toml_train_file    = f'{TRAIN_DATA_DIR}/{pred_ds}_perturbqa.toml'
config_train = {
    'datasets':  {'combined': training_data_path},
    'training':  {'combined': 'train'},   # default: all cells are training data
    'zeroshot':  {},
    'fewshot':   {
        # override: held-out cell line val and test perts are withheld from training
        f'combined.{pred_ds}': {
            'val':  val_perts,
            'test': test_perts,   # dropped perts already excluded from h5ad in step1
        },
    },
}
with open(toml_train_file, 'w') as f:
    toml.dump(config_train, f)

# TOML for prediction on test perts of held-out cell line
toml_test_file = f'{TRAIN_DATA_DIR}/{pred_ds}_perturbqa_test.toml'
config_test = {
    'datasets': {'combined': training_data_path},
    'training': {'combined': 'train'},
    'zeroshot': {},
    'fewshot':  {
        f'combined.{pred_ds}': {'test': test_perts},
    },
}
with open(toml_test_file, 'w') as f:
    toml.dump(config_test, f)

env = os.environ.copy()
env['WANDB_MODE'] = 'disabled'

# === Train ===
subprocess.run(
    [
        'state', 'tx', 'train',
        f'data.kwargs.toml_config_path={toml_train_file}',
        'data.kwargs.embed_key=null',
        'data.kwargs.num_workers=8',
        'data.kwargs.output_space=all',
        'data.kwargs.batch_col=gem_group',
        'data.kwargs.pert_col=gene',
        'data.kwargs.cell_type_key=cell_line',
        'data.kwargs.control_pert=non-targeting',
        f'training.max_steps={10 if test_run else 50000}',
        f'training.ckpt_every_n_steps={5 if test_run else 2000}',
        'training.batch_size=32',
        'training.lr=1e-3',
        'model.kwargs.cell_set_len=64',
        'model.kwargs.hidden_dim=672',
        'model.kwargs.batch_encoder=True',
        'model=state',
        f'output_dir=STATE/perturbQA_expr/results',
        f'name={pred_ds}',
    ],
    check=True,
    env=env,
)

# === Predict ===
subprocess.run(
    [
        'state', 'tx', 'predict',
        '--output-dir', RESULTS_DIR,
        '--toml',       toml_test_file,
        '--checkpoint', 'last.ckpt' if test_run else 'best.ckpt',
        '--profile',    'anndata',
        '--predict-only',
    ],
    check=True,
    env=env,
)
