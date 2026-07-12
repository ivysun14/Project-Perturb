"""
eval_metrics.py
===============
Evaluate the full metric set for a single (model, dataset) on sparsity-calibrated
test predictions.  Designed to be called once per array task from eval_metrics.sh.

Metrics computed (via perturb_metrics.compute_all_metrics_from_dfs):
  3-class DE classification:
    accuracy, balanced_accuracy, f1_macro, per-class recall/precision/f1
  DE overlap:
    de_overlap_N  (k = number of true DEGs per pert)
    de_overlap_50 (k = 50)
    de_precision_50 (top-50 precision; denominator = |G_pred^50|)
  Expression (on per-perturbation mean profiles):
    mae, mse
    pearson_global  (single r over all (pert, gene) pairs)
    pearson_delta   (Pearson of |pred - ctrl_pred| vs |real - ctrl_real|)
    pearson_per_pert (mean over perts of per-pert gene-profile r)
    pearson_per_gene (mean over genes of per-gene pert-profile r)
    w2_mean         (mean 1-D Wasserstein-2 per gene over pert means)

Usage
-----
    python calibration/eval_metrics.py --model Scouter     --dataset hepg2
    python calibration/eval_metrics.py --model TranScouter --dataset k562 \
        --out_dir calibration/eval_results
"""

import argparse
import os
import sys
import pandas as pd
import scanpy as sc

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_project_root)
sys.path.append(os.path.join(_project_root, 'Scouter', 'Scouter'))
from perturb_metrics import compute_all_metrics_from_dfs

# ── Ground-truth DE CSVs ──────────────────────────────────────────────────────
_DATA_CSV = '/weka/home/jsun140/scratchyiqunchen/project_perturb_data/csv'
_TRUTH = {
    'hepg2':  os.path.join(_DATA_CSV, 'DE_DOC_HepG2_splitted.csv'),
    'jurkat': os.path.join(_DATA_CSV, 'DE_DOC_Jurkat_splitted.csv'),
    'k562':   os.path.join(_DATA_CSV, 'DE_DOC_K562_splitted.csv'),
    'rpe1':   os.path.join(_DATA_CSV, 'DE_DOC_RPE1_splitted.csv'),
}

# ── Per-model path templates (relative to project root) ──────────────────────
_BASE = '/weka/home/jsun140/scratchyiqunchen/jsun140/Project-Perturb'
_MODEL_CFG = {
    'Scouter': {
        'de_dir':   'calibration/sparsity_de/Scouter',
        'pred_dir': 'calibration/sparsity_calibrated/Scouter',
        'real_tpl': 'Scouter/Scouter/within_cond/prediction/{ds}/{ds}_test_real.h5ad',
    },
    'TranScouter': {
        'de_dir':   'calibration/sparsity_de/TranScouter',
        'pred_dir': 'calibration/sparsity_calibrated/TranScouter',
        'real_tpl': 'Scouter/Scouter/cross_cond/prediction/{ds}/{ds}_test_real.h5ad',
    },
}


def main():
    parser = argparse.ArgumentParser(description='Evaluate calibrated DE predictions.')
    parser.add_argument('--model',     required=True, choices=list(_MODEL_CFG.keys()))
    parser.add_argument('--dataset',   required=True, choices=list(_TRUTH.keys()))
    parser.add_argument('--out_dir',   default='calibration/eval_results')
    parser.add_argument('--split_col', default='split_calib',
                        help='Column in truth CSV used to select the test split.')
    parser.add_argument('--split_val', default='test',
                        help='Value of split_col that identifies test rows.')
    args = parser.parse_args()

    cfg = _MODEL_CFG[args.model]
    ds  = args.dataset

    pred_de_path = os.path.join(_BASE, cfg['de_dir'],   f'{ds}_pred_DE.csv')
    pred_h5ad    = os.path.join(_BASE, cfg['pred_dir'], f'{ds}_test_pred.h5ad')
    real_h5ad    = os.path.join(_BASE, cfg['real_tpl'].format(ds=ds))
    truth_csv    = _TRUTH[ds]

    print(f"Model={args.model}  Dataset={ds}")
    print(f"  pred DE CSV : {pred_de_path}")
    print(f"  pred h5ad   : {pred_h5ad}")
    print(f"  real h5ad   : {real_h5ad}")
    print(f"  truth CSV   : {truth_csv}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nLoading data ...")
    pred_de_df = pd.read_csv(pred_de_path)
    print(f"  pred DE: {len(pred_de_df)} rows")

    real_de_df = pd.read_csv(truth_csv)
    real_de_df = real_de_df[real_de_df[args.split_col] == args.split_val].copy()
    print(f"  truth DE ({args.split_col}=={args.split_val!r}): {len(real_de_df)} rows")

    print(f"  Loading pred h5ad ...")
    pred_ad = sc.read_h5ad(pred_h5ad)
    print(f"    {pred_ad.n_obs} cells x {pred_ad.n_vars} genes")

    print(f"  Loading real h5ad ...")
    real_ad = sc.read_h5ad(real_h5ad)
    print(f"    {real_ad.n_obs} cells x {real_ad.n_vars} genes")

    # ── Compute metrics ───────────────────────────────────────────────────────
    print("\nComputing metrics ...")
    metrics = compute_all_metrics_from_dfs(pred_de_df, real_de_df, pred_ad, real_ad, skip_w2=True)

    print(f"\n--- {ds.upper()} ({args.model}, sparsity-calibrated) ---")
    for k, v in metrics.items():
        print(f"  {k:<22}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # ── Save ──────────────────────────────────────────────────────────────────
    row = {'model': args.model, 'dataset': ds, **metrics}
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f'{args.model}_{ds}_metrics.csv')
    pd.DataFrame([row]).to_csv(out_path, index=False)
    print(f"\nMetrics saved to {out_path}")


if __name__ == '__main__':
    main()
