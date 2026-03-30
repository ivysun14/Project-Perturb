"""
eval_metrics.py
===============
Compare predicted DE labels (from compute_de.py) against ground-truth DE labels
and compute 3-class (Non-DE / Up / Down) metrics across multiple datasets.
"""

import argparse
import os
import sys
import pandas as pd

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_project_root)
sys.path.append(os.path.join(_project_root, "majority_vote"))
from majority_vote.analysis_utility import compute_3class_metrics

_TRUTH = {
    "hepg2":  "../data/DE_DOC_hepg2_splitted.csv",
    "jurkat": "../data/DE_DOC_jurkat_splitted.csv",
    "rpe1":   "../data/DE_DOC_rpe1_splitted.csv",
}
_DS_NAMES = list(_TRUTH.keys())
SPLIT = None    # filter truth to this split; None to use all rows


def _build_datasets(pred_dir, ds_prefix=""):
    return [
        {
            "dataset": ds,
            "pred":    os.path.join(pred_dir, f"{ds_prefix}{ds}_pred_DE.csv"),
            "truth":   _TRUTH[ds],
        }
        for ds in _DS_NAMES
    ]


def eval_one(dataset, pred_path, truth_path):
    print(f"  Loading predictions from {pred_path} ...")
    df_pred = pd.read_csv(pred_path)[["pert", "gene", "label"]].rename(columns={"label": "y_pred"})
    print(f"    {len(df_pred)} predicted pairs")

    print(f"  Loading ground truth from {truth_path} ...")
    df_truth = pd.read_csv(truth_path)
    print(f"    Total rows: {len(df_truth)}")

    if SPLIT is not None:
        df_truth = df_truth[df_truth["split"] == SPLIT].copy()
        print(f"    After filtering to split='{SPLIT}': {len(df_truth)} rows")
    
    df_truth = df_truth[["pert", "gene", "label"]].rename(columns={"label": "y_true"})

    # merge on (pert, gene)
    merged = df_truth.merge(df_pred, on=["pert", "gene"], how="inner")
    print(f"\n    Ground-truth pairs : {len(df_truth)}")
    print(f"    Predicted pairs    : {len(df_pred)}")
    print(f"    Matched pairs      : {len(merged)}")

    # There shouldn't be any such pairs. If there is, need to check the files for potential errors
    unmatched = set(zip(df_pred["pert"], df_pred["gene"])) - set(zip(df_truth["pert"], df_truth["gene"]))
    if unmatched:
        print(f"\n    WARNING: {len(unmatched)} predicted pairs have no match in ground truth.")
        for pert, gene in sorted(unmatched)[:10]:
            print(f"      {pert}  {gene}")
        if len(unmatched) > 10:
            print(f"      ... and {len(unmatched) - 10} more.")

    if len(merged) == 0:
        print(f"  ERROR: No matched pairs for {dataset}. Skipping.")
        return None

    y_true = merged["y_true"].to_numpy()
    y_pred = merged["y_pred"].to_numpy()
    metrics = compute_3class_metrics(y_true, y_pred)

    print(f"\n  --- {dataset.upper()} (calibrated) ---")
    print(f"    Accuracy          : {metrics['accuracy']:.4f}")
    print(f"    Balanced Accuracy : {metrics['balanced_accuracy']:.4f}")
    print(f"    F1 (macro)        : {metrics['f1_macro']:.4f}")
    print(f"\n    Per-class metrics:")
    for ci, cname in enumerate(["Non-DE", "Up", "Down"]):
        r = metrics[f"recall_class_{ci}"]
        p = metrics[f"precision_class_{ci}"]
        f = metrics[f"f1_class_{ci}"]
        print(f"      {cname:7s}  recall={r:.4f}  precision={p:.4f}  f1={f:.4f}")

    row = {"dataset": dataset, "n_pairs": len(merged)}
    row.update(metrics)
    return row


def main():
    parser = argparse.ArgumentParser(description="Evaluate calibrated DE predictions.")
    parser.add_argument("--pred_dir", help="Directory containing {dataset}_pred_DE.csv files.")
    parser.add_argument("--out_dir", help="Directory for output metrics CSV.")
    parser.add_argument("--model", default="", help="Optional model label included in output filename.")
    parser.add_argument("--ds_prefix", default="", help="Prefix prepended to dataset name in pred filenames (e.g. 'STATE_').")
    args = parser.parse_args()

    datasets = _build_datasets(args.pred_dir, args.ds_prefix)
    out_csv = f"{args.model + '_' if args.model else ''}metrics.csv"
    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    for entry in datasets:
        print(f"\n{'='*60}")
        print(f"  Dataset: {entry['dataset'].upper()}")
        print(f"{'='*60}")
        row = eval_one(entry["dataset"], entry["pred"], entry["truth"])
        if row is not None:
            rows.append(row)

    if rows:
        out_path = os.path.join(args.out_dir, out_csv)
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"\nAll metrics saved to {out_path}")


if __name__ == "__main__":
    main()
