"""
collect_within_sensitivity_metrics.py
======================================
Collect all within-cell-line sensitivity analysis metrics into one CSV.

Reads from KNN_sensitivity_full_within_expr/results/ and produces:
  KNN_sensitivity_full_within_expr/within_sensitivity_metrics.csv

Columns:
  model, task, dataset, embedding_type, k, split,
  STATE_eval_set, unique_genes, unique_pert, <metric columns>

Run from KNN_vote/:
  python collect_within_sensitivity_metrics.py
"""

import re
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("KNN_sensitivity_full_within_expr/results")
OUT_PATH    = Path("KNN_sensitivity_full_within_expr/within_sensitivity_metrics.csv")

# Metric columns to drop (redundant with the structured columns we add)
DROP_COLS = {"best_k", "dataset"}

# Regex: {dataset}_{mode}_k{K}_covfilt_knn_metrics_combined.csv
FNAME_RE = re.compile(
    r"^(?P<dataset>.+?)_(?P<mode>genept|esm2|wikicrow|copath)_k(?P<k>\d+)_covfilt_knn_metrics_combined\.csv$"
)


def main():
    rows = []

    metric_files = sorted(RESULTS_DIR.glob("*_knn_metrics_combined.csv"))
    if not metric_files:
        raise FileNotFoundError(f"No metrics files found in {RESULTS_DIR}")

    for mf in metric_files:
        m = FNAME_RE.match(mf.name)
        if m is None:
            print(f"  Skipping unrecognised filename: {mf.name}")
            continue

        dataset = m.group("dataset")
        mode    = m.group("mode")
        k       = int(m.group("k"))

        # Load metrics
        metrics_df = pd.read_csv(mf)
        if len(metrics_df) != 1:
            print(f"  WARNING: expected 1 row in {mf.name}, got {len(metrics_df)} — skipping")
            continue
        metrics = metrics_df.iloc[0].drop(labels=[c for c in DROP_COLS if c in metrics_df.columns])

        # Load predictions to count unique genes and perts evaluated
        pred_file = RESULTS_DIR / mf.name.replace("_knn_metrics_", "_knn_predictions_")
        if not pred_file.exists():
            print(f"  WARNING: predictions file not found for {mf.name} — skipping")
            continue
        pred_df = pd.read_csv(pred_file)
        unique_genes = pred_df["gene"].nunique()
        unique_perts = pred_df["pert"].nunique()

        row = {
            "model":          f"KNN_k{k}",
            "task":           "within",
            "dataset":        dataset,
            "embedding_type": mode,
            "k":              k,
            "split":          "combined",
            "STATE_eval_set": pd.NA,   # not applicable for within task
            "unique_genes":   unique_genes,
            "unique_pert":    unique_perts,
        }
        row.update(metrics.to_dict())
        rows.append(row)

    if not rows:
        raise RuntimeError("No rows collected — check that results files match the expected naming pattern.")

    out_df = pd.DataFrame(rows)

    # Canonical column order: identifiers first, then all metric columns
    id_cols     = ["model", "task", "dataset", "embedding_type", "k",
                   "split", "STATE_eval_set", "unique_genes", "unique_pert"]
    metric_cols = [c for c in out_df.columns if c not in id_cols]
    out_df      = out_df[id_cols + metric_cols]

    out_df.to_csv(OUT_PATH, index=False)
    print(f"\nCollected {len(out_df)} rows from {len(metric_files)} files.")
    print(f"Saved → {OUT_PATH}")
    print(f"\nDatasets:        {sorted(out_df['dataset'].unique())}")
    print(f"Embedding types: {sorted(out_df['embedding_type'].unique())}")
    print(f"k values:        {sorted(out_df['k'].unique())}")
    missing = set(["genept", "esm2", "wikicrow", "copath"]) - set(out_df["embedding_type"].unique())
    if missing:
        print(f"\nWARNING: missing results for embedding type(s): {sorted(missing)}")


if __name__ == "__main__":
    main()
