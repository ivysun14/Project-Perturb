import subprocess
import os
import json
import argparse
import toml
import pandas as pd
from sklearn.model_selection import train_test_split

os.chdir('/home/jsun140/scratchyiqunchen/jsun140/Project-Perturb')

parser = argparse.ArgumentParser()
parser.add_argument('--pred_ds', type=str, required=True, choices=['k562', 'rpe1', 'hepg2', 'jurkat'], help='Held-out cell line to predict')
parser.add_argument('--test-run', action='store_true', help='Subsample perts and run 10 steps to verify the pipeline end-to-end')
args = parser.parse_args()

pred_ds  = args.pred_ds
test_run = args.test_run
print(f"-------------Dataset selected: {pred_ds}-------------")
if test_run:
    print("*** TEST RUN: subsampled perts, 10 steps ***")

ds_list = ['k562', 'rpe1', 'hepg2', 'jurkat']
train_ds = [ds for ds in ds_list if ds != pred_ds]
ds_csv = {
    'k562':   '/home/jsun140/scratchyiqunchen/project_perturb_data/csv/DE_DOC_K562_splitted.csv',
    'rpe1':   '/home/jsun140/scratchyiqunchen/project_perturb_data/csv/DE_DOC_RPE1_splitted.csv',
    'hepg2':  '/home/jsun140/scratchyiqunchen/project_perturb_data/csv/DE_DOC_HepG2_splitted.csv',
    'jurkat': '/home/jsun140/scratchyiqunchen/project_perturb_data/csv/DE_DOC_Jurkat_splitted.csv',
}

# Define splits in the held-out cell line
split_df = pd.read_csv(ds_csv[pred_ds])
# val is a random 10% of train perts; calib split is reserved for post-prediction calibration
train_pert = split_df[split_df['split_calib'] == 'train']['pert'].unique().tolist()
test_pert  = split_df[split_df['split_calib'] == 'test']['pert'].unique().tolist()
train_pert, val_pert = train_test_split(train_pert, test_size=0.1, random_state=305)
print(f"Train: {len(train_pert)}, Val: {len(val_pert)}, Test: {len(test_pert)}")

if test_run:
    train_pert = train_pert[:20]
    val_pert   = val_pert[:5]
    test_pert  = test_pert[:5]
    print(f"Subsampled → Train: {len(train_pert)}, Val: {len(val_pert)}, Test: {len(test_pert)}")

# Load non-held-out, training cell line CSVs
train_dfs = {td: pd.read_csv(ds_csv[td]) for td in train_ds}

## ========== Documentation Purpose ==========
## We would like to know when predicting on the test set of the
## held-out cell line, how many perturbations are dropped since
## they were never seen by the training set. Similarly for calibration
## perturbations from all 3 training cell lines.

# Load dropped-pert lists from coverage analysis
with open('STATE/pert_coverage/dropped_test_perts.json') as f:
    dropped_test_map = json.load(f)
with open('STATE/pert_coverage/dropped_calib_perts.json') as f:
    dropped_calib_map = json.load(f)
dropped_test_set  = set(dropped_test_map.get(pred_ds, []))
dropped_calib_set = set(dropped_calib_map.get(pred_ds, []))

# Filter test perts of held-out cell line and report drop proportion
test_pert_filtered = [p for p in test_pert if p not in dropped_test_set]
test_df = split_df[split_df['split_calib'] == 'test']
test_pairs_total   = len(test_df)
test_pairs_dropped = len(test_df[test_df['pert'].isin(dropped_test_set)])
print(f"[TEST]  perts: {len(test_pert)} total, {len(dropped_test_set)} dropped "
      f"| pert-gene pairs dropped: {test_pairs_dropped}/{test_pairs_total} "
      f"({test_pairs_dropped/test_pairs_total:.1%})")

# Per-training-cell-line filtered calib perts and drop proportions
calib_perts_filtered = {}   # {td: [kept perts]}
calib_pairs_total   = 0
calib_pairs_dropped = 0
for td in train_ds:
    td_df = train_dfs[td]
    calib_df = td_df[td_df['split_calib'] == 'calibration']
    total   = len(calib_df)
    dropped = len(calib_df[calib_df['pert'].isin(dropped_calib_set)])
    calib_pairs_total   += total
    calib_pairs_dropped += dropped
    calib_perts_filtered[td] = [
        p for p in calib_df['pert'].unique() if p not in dropped_calib_set
    ]
    print(f"[CALIB] {td}: {len(calib_perts_filtered[td])} perts kept, "
          f"{dropped}/{total} pert-gene pairs dropped ({dropped/total:.1%})")
print(f"[CALIB] union across 3 training cell lines: "
      f"{calib_pairs_dropped}/{calib_pairs_total} pert-gene pairs dropped "
      f"({calib_pairs_dropped/calib_pairs_total:.1%})")
## ===========================================

# ========== TOML for training ==========
training_data_path = f"STATE/training_data/combined_for_{pred_ds}_full.h5ad"
toml_train_file    = f"STATE/training_data/{pred_ds}.toml"
config_train = {
    "datasets": {"combined": training_data_path},
    "training": {"combined": "train"},
    "zeroshot": {},
    "fewshot": {
        # We have two predictions to make:
        # 1). held-out cell line: val from train split, test from test split
        f"combined.{pred_ds}": {
            "val":  val_pert,
            "test": test_pert_filtered,
        },
        # 2). non-held-out training cell lines: mark calib perts as fewshot test so STATE does NOT train on them
        # we would want trained STATE model to make predictions for them as well so we can use these predictions
        # during calibration
        **{f"combined.{td}": {"test": calib_perts_filtered[td]} for td in train_ds},
    },
}
with open(toml_train_file, "w") as f:
    toml.dump(config_train, f)

# ========== TOML for predicting on test perts of held-out cell line ==========
toml_test_file = f"STATE/training_data/{pred_ds}_test.toml"
config_test = {
    "datasets": {"combined": training_data_path},
    "training": {"combined": "train"},
    "zeroshot": {},
    "fewshot": {
        f"combined.{pred_ds}": {"test": test_pert_filtered},
    },
}
with open(toml_test_file, "w") as f:
    toml.dump(config_test, f)

# ========== TOML for predicting on calib perts of 3 training cell lines ==========
toml_calib_file = f"STATE/training_data/{pred_ds}_calib.toml"
config_calib = {
    "datasets": {"combined": training_data_path},
    "training": {"combined": "train"},
    "zeroshot": {},
    "fewshot": {
        f"combined.{td}": {"test": calib_perts_filtered[td]}
        for td in train_ds
    },
}
with open(toml_calib_file, "w") as f:
    toml.dump(config_calib, f)

## Train 
env = os.environ.copy()
env["WANDB_MODE"] = "disabled"
subprocess.run(
    [
        "state", "tx", "train",
        f"data.kwargs.toml_config_path={toml_train_file}",
        "data.kwargs.embed_key=null",
        "data.kwargs.num_workers=8",
        "data.kwargs.output_space=all",
        "data.kwargs.batch_col=gem_group",
        "data.kwargs.pert_col=gene",
        "data.kwargs.cell_type_key=cell_line",
        "data.kwargs.control_pert=non-targeting",
        f"training.max_steps={10 if test_run else 50000}",
        f"training.ckpt_every_n_steps={5 if test_run else 2000}",
        "training.batch_size=32",
        "training.lr=1e-3",
        "model.kwargs.cell_set_len=64",
        "model.kwargs.hidden_dim=672",  # changed this from 128 to bypass error:
                                        # ValueError: The hidden size (128) is not a multiple of the number of attention heads (12).
        "model.kwargs.batch_encoder=True",
        "model=state",
        f"output_dir=STATE/results",
        f"name={pred_ds}",
    ],
    check=True,
    env=env,
)

## Prediction Task 1: test set (held-out cell line)
## Output → STATE/results/{pred_ds}/eval_best.ckpt/
subprocess.run(
    [
        "state", "tx", "predict",
        "--output-dir", f"STATE/results/{pred_ds}",
        "--toml",       toml_test_file,
        "--checkpoint", "best.ckpt",    # use best checkpoint saved for prediction
        "--profile",    "anndata",
        "--predict-only",
    ],
    check=True,
    env=env,
)

## Prediction Task 2: calibration sets (3 training cell lines)
## Output → STATE/results/{pred_ds}/eval_calib.ckpt/
## Use a symlink alias so output goes to eval_calib.ckpt/ instead of overwriting eval_best.ckpt/
ckpt_dir  = f"STATE/results/{pred_ds}/checkpoints"
calib_ckpt = os.path.join(ckpt_dir, "calib.ckpt")
if os.path.islink(calib_ckpt) or os.path.exists(calib_ckpt):
    os.remove(calib_ckpt)
os.symlink("best.ckpt", calib_ckpt)   # relative symlink within same directory

subprocess.run(
    [
        "state", "tx", "predict",
        "--output-dir", f"STATE/results/{pred_ds}",
        "--toml",       toml_calib_file,
        "--checkpoint", "calib.ckpt", # this is a symlink to the best checkpoint
        "--profile",    "anndata",
        "--predict-only",
    ],
    check=True,
    env=env,
)
