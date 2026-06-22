"""
collect_within_sensitivity_metrics_perturbQA.py
================================================
Collect within-cell-line sensitivity analysis metrics and zero-neighbor
percentages for the PerturbQA experiment.

Reads from KNN_sensitivity_perturbQA_within_expr/results/ and produces:
  KNN_sensitivity_perturbQA_within_expr/within_sensitivity_metrics.csv
  KNN_sensitivity_perturbQA_within_expr/zero_neighbor_pcts.csv

Differences from the full-dataset equivalent:
  - No split suffix on filenames (PerturbQA within has a single test split,
    not calib/test/combined).
  - zero_neighbor is read directly from the predictions boolean column
    rather than inferred from uniform probabilities.

Run from KNN_vote/:
  python collect_within_sensitivity_metrics_perturbQA.py
"""

import re
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("KNN_sensitivity_perturbQA_within_expr/results")
METRICS_OUT = Path("KNN_sensitivity_perturbQA_within_expr/within_sensitivity_metrics.csv")
ZERO_NB_OUT = Path("KNN_sensitivity_perturbQA_within_expr/zero_neighbor_pcts.csv")

DROP_COLS = {"best_k", "dataset"}

# {dataset}_{mode}_k{K}_covfilt_knn_metrics.csv
FNAME_RE = re.compile(
    r"^(?P<dataset>.+?)_(?P<mode>genept|esm2|wikicrow|copath)"
    r"_k(?P<k>\d+)_covfilt_knn_metrics\.csv$"
)


def collect_metrics() -> pd.DataFrame:
    rows = []

    metric_files = sorted(RESULTS_DIR.glob("*_covfilt_knn_metrics.csv"))
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

        metrics_df = pd.read_csv(mf)
        if len(metrics_df) != 1:
            print(f"  WARNING: expected 1 row in {mf.name}, got {len(metrics_df)} — skipping")
            continue
        metrics = metrics_df.iloc[0].drop(labels=[c for c in DROP_COLS if c in metrics_df.columns])

        pred_file = RESULTS_DIR / mf.name.replace("_knn_metrics.", "_knn_predictions.")
        if not pred_file.exists():
            print(f"  WARNING: predictions file not found for {mf.name} — skipping")
            continue
        pred_df      = pd.read_csv(pred_file, usecols=["gene", "pert"])
        unique_genes = pred_df["gene"].nunique()
        unique_perts = pred_df["pert"].nunique()

        row = {
            "model":          f"KNN_k{k}",
            "task":           "within",
            "dataset":        dataset,
            "embedding_type": mode,
            "k":              k,
            "split":          "test",
            "STATE_eval_set": pd.NA,
            "unique_genes":   unique_genes,
            "unique_pert":    unique_perts,
        }
        row.update(metrics.to_dict())
        rows.append(row)

    if not rows:
        raise RuntimeError("No rows collected — check that results files match the expected naming pattern.")

    out_df = pd.DataFrame(rows)
    id_cols     = ["model", "task", "dataset", "embedding_type", "k",
                   "split", "STATE_eval_set", "unique_genes", "unique_pert"]
    metric_cols = [c for c in out_df.columns if c not in id_cols]
    return out_df[id_cols + metric_cols]


def collect_zero_neighbor() -> pd.DataFrame:
    rows = []

    pred_files = sorted(RESULTS_DIR.glob("*_covfilt_knn_predictions.csv"))
    if not pred_files:
        print(f"  No prediction files found in {RESULTS_DIR}")
        return pd.DataFrame()

    for pf in pred_files:
        m = FNAME_RE.match(pf.name.replace("_knn_predictions.", "_knn_metrics."))
        if m is None:
            continue

        dataset = m.group("dataset")
        mode    = m.group("mode")
        k       = int(m.group("k"))

        pred_df   = pd.read_csv(pf, usecols=["zero_neighbor"])
        n_total   = len(pred_df)
        n_zero_nb = int(pred_df["zero_neighbor"].sum())
        pct       = 100.0 * n_zero_nb / n_total if n_total > 0 else float("nan")

        rows.append({
            "dataset":       dataset,
            "mode":          mode,
            "k":             k,
            "total_pairs":   n_total,
            "zero_nb_pairs": n_zero_nb,
            "pct_zero_nb":   pct,
        })
        print(f"  {dataset:8s} | {mode:8s} | k={k:2d} | "
              f"total={n_total:>9,} | zero_nb={n_zero_nb:>7,} | {pct:.4f}%")

    return pd.DataFrame(rows).sort_values(["dataset", "mode", "k"]).reset_index(drop=True)


def main():
    print("=" * 70)
    print("Collecting PerturbQA within-cell-line sensitivity metrics")
    print("=" * 70)

    print(f"\nMetrics files in {RESULTS_DIR}:")
    metrics_df = collect_metrics()
    metrics_df.to_csv(METRICS_OUT, index=False)
    print(f"\nCollected {len(metrics_df)} rows.")
    print(f"Saved → {METRICS_OUT}")
    print(f"\nDatasets:        {sorted(metrics_df['dataset'].unique())}")
    print(f"Embedding types: {sorted(metrics_df['embedding_type'].unique())}")
    print(f"k values:        {sorted(metrics_df['k'].unique())}")
    missing = {"genept", "esm2", "wikicrow", "copath"} - set(metrics_df["embedding_type"].unique())
    if missing:
        print(f"WARNING: missing results for embedding type(s): {sorted(missing)}")

    print(f"\n{'=' * 70}")
    print("Zero-neighbor percentages")
    print(f"{'=' * 70}")
    zero_df = collect_zero_neighbor()
    if not zero_df.empty:
        zero_df.to_csv(ZERO_NB_OUT, index=False)
        print(f"\nSaved → {ZERO_NB_OUT}")

        pivot = zero_df.pivot_table(
            index=["dataset", "mode"], columns="k",
            values="pct_zero_nb", aggfunc="first"
        )
        pivot.columns = [f"k={c}" for c in pivot.columns]
        print("\nSummary (% zero-neighbor pairs):")
        print(pivot.to_string(float_format=lambda x: f"{x:.4f}%"))


if __name__ == "__main__":
    main()
