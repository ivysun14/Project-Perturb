"""
tune_calibration.py
===================
Tune calibration strength r for a single (method, model, dataset) combination
using the calibration split (split_calib == 'calibration').

For each r in the search grid the script:
  1. Applies calibration in memory (no h5ad written to disk)
  2. Runs DE calling on calibration-split perturbations only
  3. Evaluates against ground-truth DE labels
  4. Appends the full metric row to a results CSV

After this script, one should choose an optimal r and run calibration on the test set

Usage
-----
    python tune_calibration.py \\
        --method      var \\
        --model       STATE \\
        --dataset     hepg2 \\
        --input       ../STATE/data/hepg2_adata_pred.h5ad \\
        --truth       ../data/DE_DOC_hepg2_splitted.csv \\
        --target_col  gene \\
        --out_dir     tuning_results

    python tune_calibration.py \\
        --method      sparsity \\
        --model       Scouter \\
        --dataset     jurkat \\
        --input       ../Scouter/shared/cross_condition/jurkat_pred.h5ad \\
        --truth       ../data/DE_DOC_jurkat_splitted.csv \\
        --target_col  pert \\
        --out_dir     tuning_results
"""

import argparse
import os
import sys
import warnings
import numpy as np
import pandas as pd
import anndata as ad

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_project_root)
sys.path.append(os.path.join(_project_root, "majority_vote"))

from calibrate_var import calibrate_variance
from calibrate_sparsity import calibrate_zeros
from majority_vote.DE_DOC_stats import compute_de_statistics
from majority_vote.analysis_utility import resolve_repeats, compute_3class_metrics

ALPHA = 0.01
LOG2FC_THRESHOLD = 0.00


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune calibration strength r on the calibration split.")
    parser.add_argument("--method", required=True, choices=["var", "sparsity"], help="Calibration method.")
    parser.add_argument("--model", required=True, help="Model label used in output filename (e.g. STATE, Scouter).")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. hepg2, jurkat, rpe1).")
    parser.add_argument("--input", required=True, help="Path to predicted h5ad.")
    parser.add_argument("--truth", required=True, help="Path to ground-truth splitted CSV (must have split_calib column).")
    parser.add_argument("--target_col", default="gene", help="obs column for perturbation identity.")
    parser.add_argument("--control_name", default="ctrl", help="Control label in target_col.")
    parser.add_argument("--r_min", type=float, default=0.2, help="Minimum r value (inclusive).")
    parser.add_argument("--r_max", type=float, default=1.0, help="Maximum r value (inclusive).")
    parser.add_argument("--r_step", type=float, default=0.2, help="Step size for r grid.")
    parser.add_argument("--out_dir", default="tuning_results", help="Directory for output CSV.")
    args = parser.parse_args()

    # generate a grid of r values for tuning
    r_grid = [round(r, 10) for r in np.arange(args.r_min, args.r_max + args.r_step / 2, args.r_step)]
    print(f"Tuning {args.method} calibration — {args.model} / {args.dataset}")
    print(f"r grid: {r_grid}")

    # load predicted expression h5ad + ground truth csv
    adata_full = ad.read_h5ad(args.input)
    truth_df = pd.read_csv(args.truth)
    
    # single out calibration perturbations and their ground truth labels
    truth_calib = (
        truth_df[truth_df["split_calib"] == "calibration"][["pert", "gene", "label"]]
        .rename(columns={"label": "y_true"})
    )
    print(f"  Ground-truth pairs        : {len(truth_calib)}")

    # tune hyperparameter
    rows = []
    for r in r_grid:
        print(f"\n{'='*50}")
        print(f"  r = {r:.2f}")
        print(f"{'='*50}")

        # adata_r stores calibrated expressions
        adata_r = adata_full.copy()
        if args.method == "var":
            adata_r = calibrate_variance(
                adata_r,
                target_col=args.target_col,
                control_name=args.control_name,
                ratio=r,
            )
        else:
            adata_r = calibrate_zeros(
                adata_r,
                target_col=args.target_col,
                control_name=args.control_name,
                ratio=r,
            )

        # run DE in memory (do not save calibrated object or results)
        pred_df = compute_de_statistics(
            adata=adata_r,
            control_key=args.control_name,
            target_col=args.target_col,
            gene_row="gene_name",
            alpha=ALPHA,
            log2fc_threshold=LOG2FC_THRESHOLD,
            save_path=None,
        )[["pert", "gene", "label"]].rename(columns={"label": "y_pred"})

        # merge ground truth with predictions into one dataframe
        merged = truth_calib.merge(pred_df, on=["pert", "gene"], how="inner")
        if len(merged) != len(pred_df):
            warnings.warn(f"{len(pred_df) - len(merged)} pred pairs have no match in truth_calib — truth_df may not be a superset.")

        metrics = compute_3class_metrics(merged["y_true"].values, merged["y_pred"].values)

        row = {
            "model":   args.model,
            "dataset": args.dataset,
            "method":  args.method,
            "r":       r,
            "n_pairs": len(merged),
        }
        row.update(metrics)
        rows.append(row)

        print(f"  f1_macro          : {metrics['f1_macro']:.4f}")
        print(f"  balanced_accuracy : {metrics['balanced_accuracy']:.4f}")
        for ci, cname in enumerate(["Non-DE", "Up", "Down"]):
            p = metrics[f"precision_class_{ci}"]
            rec = metrics[f"recall_class_{ci}"]
            f = metrics[f"f1_class_{ci}"]
            print(f"  {cname:7s}  prec={p:.4f}  rec={rec:.4f}  f1={f:.4f}")

    # save tuning result, look into the csv to determine best r
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.method}_{args.model}_{args.dataset}_tuning.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nTuning results saved to {out_path}")