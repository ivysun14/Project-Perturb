"""
tune_calibration.py
===================
Array-job-friendly calibration tuner.

Array layout: --array=0-15 (4 r values × 4 splits).
  task_id = r_idx * n_splits + split_idx
  → each task processes exactly one (r, split) pair.

Cross-condition strategy (TranScouter / STATE)
----------------------------------------------
For each held-out cell line (pred_ds), the 3 training cell lines' calib
objects are POOLED before calibration.  Rationale: the goal is to learn
the model's systematic bias, which is a model property rather than a
cell-line property.  A single global r learned from the pooled set
generalises better to the held-out test set.  The calibration functions
(calibrate_variance / calibrate_zeros) derive their reference statistics
(ctrl variance / ctrl zero-ratio) from the ctrl cells in the concatenated
adata, giving a global threshold.

This is also consistent with STATE, whose calib TOML groups all 3 training
cell lines into one prediction call (so STATE naturally produces a pooled
output).

Pre-calibration metrics are already computed by the step4 metric scripts;
this script computes post-calibration 3-class DE metrics only (expensive
expression metrics are not recomputed during tuning).

Usage
-----
    # From SLURM array (see tune_calibration.sh):
    python tune_calibration.py --model TranScouter --method var --task_id $SLURM_ARRAY_TASK_ID

    # After all tasks finish, gather:
    python calibration/gather_tuning.py --model TranScouter --method var
"""

import argparse
import os
import sys
import pandas as pd
import anndata as ad
import scanpy as sc

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, 'majority_vote'))
sys.path.insert(0, os.path.join(_project_root, 'calibration'))
sys.path.insert(0, os.path.join(_project_root, 'calibration', 'var_calibration'))

from calibrate_var      import calibrate_variance
from calibrate_sparsity import calibrate_zeros
from preprocessing.DE_DOC_stats import compute_de_statistics
from metrics.class_metrics import compute_3class_metrics

os.chdir(_project_root)

# ─── Constants ────────────────────────────────────────────────────────────────

DS_LIST = ['k562', 'rpe1', 'hepg2', 'jurkat']
R_GRID  = [0.4, 0.6, 0.8, 1.0]

DS_CSV = {
    'k562':   '/projects/ychen756/project_perturb_data/csv/DE_DOC_K562_splitted.csv',
    'rpe1':   '/projects/ychen756/project_perturb_data/csv/DE_DOC_RPE1_splitted.csv',
    'hepg2':  '/projects/ychen756/project_perturb_data/csv/DE_DOC_HepG2_splitted.csv',
    'jurkat': '/projects/ychen756/project_perturb_data/csv/DE_DOC_Jurkat_splitted.csv',
}

# ─── Split definitions ────────────────────────────────────────────────────────
# Each entry has:
#   label         : human-readable identifier for this split
#   pred_h5ads    : list of pred h5ad paths (length 1 for within-cond, 3 for cross-cond)
#   real_h5ads    : list of real h5ad paths (same length)
#   truth_dses    : list of dataset keys whose calibration rows to pool for ground truth
#   split_calib   : value to filter truth df on 'split_calib' column
#   target_col    : obs column for perturbation identity
#   control_name  : control label in target_col
#   gene_row      : var column for gene names

def _scouter_splits():
    """4 splits: one per dataset (within-cond, single h5ad each)."""
    splits = []
    for ds in DS_LIST:
        base = f'Scouter/Scouter/within_cond/prediction/{ds}'
        splits.append({
            'label':        f'{ds}_calib',
            'pred_h5ads':   [f'{base}/{ds}_calib_pred.h5ad'],
            'real_h5ads':   [f'{base}/{ds}_calib_real.h5ad'],
            'truth_dses':   [ds],
            'split_calib':  'calibration',
            'target_col':   'gene',
            'control_name': 'ctrl',
            'gene_row':     'gene_name',
        })
    return splits


def _transcouter_splits():
    """4 splits: one per pred_ds, pooling 3 training cell lines' calib sets."""
    splits = []
    for pred_ds in DS_LIST:
        train_ds = [d for d in DS_LIST if d != pred_ds]
        base = f'Scouter/Scouter/cross_cond/prediction/{pred_ds}'
        splits.append({
            'label':        f'{pred_ds}_calib_pooled',
            'pred_h5ads':   [f'{base}/{pred_ds}_{td}_calib_pred.h5ad' for td in train_ds],
            'real_h5ads':   [f'{base}/{pred_ds}_{td}_calib_real.h5ad' for td in train_ds],
            'truth_dses':   train_ds,
            'split_calib':  'calibration',
            'target_col':   'gene',
            'control_name': 'ctrl',
            'gene_row':     'gene_name',
        })
    return splits


def _state_splits():
    """
    4 splits: one per pred_ds, pooling 3 training cell lines' calib sets.
    Paths are placeholders — update once STATE training finishes and the
    exact h5ad filenames under STATE/results/{pred_ds}/eval_calib.ckpt/ are known.
    """
    splits = []
    for pred_ds in DS_LIST:
        train_ds = [d for d in DS_LIST if d != pred_ds]
        base = f'STATE/results/{pred_ds}/eval_calib.ckpt'
        splits.append({
            'label':        f'{pred_ds}_calib_pooled',
            'pred_h5ads':   [f'{base}/{pred_ds}_{td}_calib_pred.h5ad' for td in train_ds],
            'real_h5ads':   [f'{base}/{pred_ds}_{td}_calib_real.h5ad' for td in train_ds],
            'truth_dses':   train_ds,
            'split_calib':  'calibration',
            'target_col':   'gene',
            'control_name': 'non-targeting',
            'gene_row':     'gene_name',
        })
    return splits


MODEL_SPLITS = {
    'Scouter':     _scouter_splits(),
    'TranScouter': _transcouter_splits(),
    'STATE':       _state_splits(),
}

# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',   required=True, choices=list(MODEL_SPLITS))
    parser.add_argument('--method',  required=True, choices=['var', 'sparsity'])
    parser.add_argument('--task_id', required=True, type=int,
                        help='SLURM_ARRAY_TASK_ID; encodes (r_idx, split_idx).')
    parser.add_argument('--out_dir', default='calibration/tuning_results')
    args = parser.parse_args()

    model  = args.model
    method = args.method
    splits = MODEL_SPLITS[model]
    n_splits = len(splits)              # 4 for all models

    r_idx     = args.task_id // n_splits
    split_idx = args.task_id % n_splits
    r         = R_GRID[r_idx]
    sp        = splits[split_idx]

    print(f"model={model}  method={method}  r={r:.2f}  split={sp['label']}")
    print(f"(task_id={args.task_id} → r_idx={r_idx}, split_idx={split_idx})\n")

    # Load ground-truth DFs
    truth_dfs = {ds: pd.read_csv(DS_CSV[ds]) for ds in DS_LIST}

    # ── Load and pool pred h5ads ───────────────────────────────────────────
    pred_ads = []
    for path in sp['pred_h5ads']:
        full = os.path.join(_project_root, path)
        if not os.path.exists(full):
            print(f"  SKIP: pred h5ad not found: {full}")
            break
        pred_ads.append(ad.read_h5ad(full))
    else:
        pass  # all found

    if len(pred_ads) != len(sp['pred_h5ads']):
        print("  Aborting — not all pred h5ads available.")
        raise SystemExit(1)

    pred_ad = sc.concat(pred_ads, join='inner') if len(pred_ads) > 1 else pred_ads[0]
    # sc.concat drops var columns; restore gene_name from the var index
    if sp['gene_row'] not in pred_ad.var.columns:
        pred_ad.var[sp['gene_row']] = pred_ad.var.index.astype(str)
    print(f"Pooled pred adata: {pred_ad.shape}")

    # ── Calibrate in memory ────────────────────────────────────────────────
    if method == 'var':
        pred_ad = calibrate_variance(
            pred_ad,
            target_col=sp['target_col'],
            control_name=sp['control_name'],
            ratio=r,
        )
    else:
        pred_ad = calibrate_zeros(
            pred_ad,
            target_col=sp['target_col'],
            control_name=sp['control_name'],
            ratio=r,
        )

    # ── Run DE on calibrated expression ────────────────────────────────────
    pred_de_df = compute_de_statistics(
        adata=pred_ad,
        control_key=sp['control_name'],
        target_col=sp['target_col'],
        gene_row=sp['gene_row'],
        alpha=0.01,
        log2fc_threshold=0.0,
        save_path=None,
    )

    # ── Pool ground-truth DE labels ────────────────────────────────────────
    real_de_df = pd.concat([
        truth_dfs[ds][truth_dfs[ds]['split_calib'] == sp['split_calib']]
        for ds in sp['truth_dses']
    ], ignore_index=True)

    merged = real_de_df.merge(
        pred_de_df[['pert', 'gene', 'label']], on=['pert', 'gene'],
        how='inner', suffixes=('_real', '_pred')
    )
    metrics = compute_3class_metrics(merged['label_real'].values, merged['label_pred'].values)

    print(f"\n  f1_macro={metrics['f1_macro']:.4f}  "
          f"balanced_acc={metrics['balanced_accuracy']:.4f}")

    row = {'model': model, 'method': method, 'r': r, 'split': sp['label']}
    row.update(metrics)

    # ── Save partial CSV (one per task) ────────────────────────────────────
    os.makedirs(os.path.join(_project_root, args.out_dir), exist_ok=True)
    out_path = os.path.join(
        _project_root, args.out_dir,
        f'{method}_{model}_task{args.task_id:02d}_partial.csv'
    )
    pd.DataFrame([row]).to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
