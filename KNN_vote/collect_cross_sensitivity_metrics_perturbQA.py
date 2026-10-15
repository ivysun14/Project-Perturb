"""
collect_cross_sensitivity_metrics_perturbQA.py
===============================================
Collect cross-cell-line sensitivity analysis metrics and zero-neighbor
percentages for the PerturbQA experiment.

Follows the same methodology as collect_cross_sensitivity_metrics.py:
  1. Read raw predictions from KNN_sensitivity_perturbQA_cross_expr/results/.
  2. Restrict genes to common_gene_universe_QA.csv (1337 genes).
  3. Drop perturbations listed in dropped_test_perts_QA.json.
  4. Recompute metrics from the filtered predictions.
  5. STATE_eval_set = True.

Produces:
  KNN_sensitivity_perturbQA_cross_expr/cross_sensitivity_metrics.csv
  KNN_sensitivity_perturbQA_cross_expr/zero_neighbor_pcts.csv

Run from KNN_vote/:
  python collect_cross_sensitivity_metrics_perturbQA.py
"""

import re
import json
import sys
import os
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics.class_metrics import compute_3class_metrics

RESULTS_DIR = Path("KNN_sensitivity_perturbQA_cross_expr/results")
METRICS_OUT = Path("KNN_sensitivity_perturbQA_cross_expr/cross_sensitivity_metrics.csv")
ZERO_NB_OUT = Path("KNN_sensitivity_perturbQA_cross_expr/zero_neighbor_pcts.csv")

DATA_DIR           = Path("/projects/ychen756/project_perturb_data")
GENE_UNIVERSE_CSV  = DATA_DIR / "common_gene_universe_QA.csv"
DROPPED_PERTS_JSON = DATA_DIR / "dropped_test_perts_QA.json"

# {dataset}_{mode}_k{K}_covfilt_knn_transfer_predictions.csv
FNAME_RE = re.compile(
    r"^(?P<dataset>.+?)_(?P<mode>genept|esm2|wikicrow|copath)"
    r"_k(?P<k>\d+)_covfilt_knn_transfer_predictions\.csv$"
)


def main():
    gene_universe = set(pd.read_csv(GENE_UNIVERSE_CSV)["gene_name"])
    print(f"Common gene universe: {len(gene_universe)} genes")

    with open(DROPPED_PERTS_JSON) as f:
        dropped_perts_map = json.load(f)
    for ds, perts in dropped_perts_map.items():
        print(f"  Drop perts {ds}: {len(perts)}")

    pred_files = sorted(RESULTS_DIR.glob("*_covfilt_knn_transfer_predictions.csv"))
    if not pred_files:
        raise FileNotFoundError(f"No prediction files found in {RESULTS_DIR}")

    metric_rows  = []
    zero_nb_rows = []

    for pf in pred_files:
        m = FNAME_RE.match(pf.name)
        if m is None:
            print(f"  Skipping unrecognised filename: {pf.name}")
            continue

        dataset = m.group("dataset")
        mode    = m.group("mode")
        k       = int(m.group("k"))
        ds_key  = dataset.lower()

        if ds_key not in dropped_perts_map:
            print(f"  WARNING: no dropped_perts entry for '{dataset}' — skipping")
            continue

        df = pd.read_csv(pf)
        n_before = len(df)

        df = df[df["gene"].isin(gene_universe)]
        n_after_gene = len(df)

        drop_set = set(dropped_perts_map[ds_key])
        df = df[~df["pert"].isin(drop_set)]
        n_after_pert = len(df)

        print(f"\n{dataset} | {mode} | k={k}:")
        print(f"  rows before: {n_before}, after gene filter: {n_after_gene} "
              f"(-{n_before - n_after_gene}), after pert filter: {n_after_pert} "
              f"(-{n_after_gene - n_after_pert})")

        if len(df) == 0:
            print(f"  WARNING: 0 rows remain after filtering — skipping")
            continue

        y_true  = df["label"].values
        y_pred  = df["prediction"].values
        y_proba = df[["prob_class_0", "prob_class_1", "prob_class_2"]].values

        metrics = compute_3class_metrics(y_true, y_pred, y_proba=y_proba)

        row = {
            "model":          f"KNN_k{k}",
            "task":           "cross",
            "dataset":        dataset,
            "embedding_type": mode,
            "k":              k,
            "split":          "test",
            "STATE_eval_set": True,
            "unique_genes":   df["gene"].nunique(),
            "unique_pert":    df["pert"].nunique(),
        }
        row.update(metrics)
        metric_rows.append(row)

        # Zero-neighbor count from the filtered predictions
        n_total   = len(df)
        n_zero_nb = int(df["zero_neighbor"].sum())
        pct       = 100.0 * n_zero_nb / n_total if n_total > 0 else float("nan")
        zero_nb_rows.append({
            "dataset":       dataset,
            "mode":          mode,
            "k":             k,
            "total_pairs":   n_total,
            "zero_nb_pairs": n_zero_nb,
            "pct_zero_nb":   pct,
        })
        print(f"  zero_nb: {n_zero_nb}/{n_total} ({pct:.4f}%)")

    if not metric_rows:
        raise RuntimeError(
            "No rows collected — check that prediction files match the expected naming pattern."
        )

    # Save metrics CSV
    out_df = pd.DataFrame(metric_rows)
    id_cols     = ["model", "task", "dataset", "embedding_type", "k",
                   "split", "STATE_eval_set", "unique_genes", "unique_pert"]
    metric_cols = [c for c in out_df.columns if c not in id_cols]
    out_df      = out_df[id_cols + metric_cols]
    out_df.to_csv(METRICS_OUT, index=False)
    print(f"\nCollected {len(out_df)} rows.")
    print(f"Saved → {METRICS_OUT}")
    print(f"\nDatasets:        {sorted(out_df['dataset'].unique())}")
    print(f"Embedding types: {sorted(out_df['embedding_type'].unique())}")
    print(f"k values:        {sorted(out_df['k'].unique())}")
    missing = {"genept", "esm2", "wikicrow", "copath"} - set(out_df["embedding_type"].unique())
    if missing:
        print(f"\nWARNING: missing results for embedding type(s): {sorted(missing)}")

    # Save zero-neighbor CSV with pivot summary
    zero_df = (pd.DataFrame(zero_nb_rows)
               .sort_values(["dataset", "mode", "k"])
               .reset_index(drop=True))
    zero_df.to_csv(ZERO_NB_OUT, index=False)
    print(f"\nSaved → {ZERO_NB_OUT}")

    pivot = zero_df.pivot_table(
        index=["dataset", "mode"], columns="k",
        values="pct_zero_nb", aggfunc="first"
    )
    pivot.columns = [f"k={c}" for c in pivot.columns]
    print("\nSummary (% zero-neighbor pairs after filtering):")
    print(pivot.to_string(float_format=lambda x: f"{x:.4f}%"))


if __name__ == "__main__":
    main()
