"""
collect_cross_sensitivity_metrics.py
=====================================
Collect all cross-cell-line sensitivity analysis metrics into one CSV.

Reads prediction files from KNN_sensitivity_full_cross_expr/results/ and
applies the same perturbation and gene filters used in metric_drop_perts_*nn.py:
  1. Restrict genes to the common gene universe.
  2. Drop perturbations in the per-dataset dropped_test_perts list.

Because filtering changes the evaluation set, metrics are recomputed from the
filtered predictions (the pre-computed metrics CSVs are not reused).

STATE_eval_set=True indicates that pert and gene subsetting was applied.

Only 'combined' split results are collected.

Produces:
  KNN_sensitivity_full_cross_expr/cross_sensitivity_metrics.csv

Run from KNN_vote/:
  python collect_cross_sensitivity_metrics.py
"""

import re
import json
import numpy as np
import pandas as pd
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from majority_vote.analysis_utility import compute_3class_metrics

RESULTS_DIR  = Path("KNN_sensitivity_full_cross_expr/results")
OUT_PATH     = Path("KNN_sensitivity_full_cross_expr/cross_sensitivity_metrics.csv")

DATA_DIR = Path("/home/jsun140/scratchyiqunchen/project_perturb_data")
GENE_UNIVERSE_CSV  = DATA_DIR / "common_gene_universe.csv"
DROPPED_PERTS_JSON = DATA_DIR / "dropped_test_perts.json"

# Regex: {dataset}_{mode}_k{K}_covfilt_knn_transfer_predictions_combined.csv
FNAME_RE = re.compile(
    r"^(?P<dataset>.+?)_(?P<mode>genept|esm2|wikicrow|copath)_k(?P<k>\d+)"
    r"_covfilt_knn_transfer_predictions_combined\.csv$"
)


def main():
    gene_universe = set(pd.read_csv(GENE_UNIVERSE_CSV)["gene_name"])
    print(f"Common gene universe: {len(gene_universe)} genes")

    with open(DROPPED_PERTS_JSON) as f:
        dropped_perts_map = json.load(f)
    for ds, perts in dropped_perts_map.items():
        print(f"  Drop perts {ds}: {len(perts)}")

    pred_files = sorted(RESULTS_DIR.glob("*_knn_transfer_predictions_combined.csv"))
    if not pred_files:
        raise FileNotFoundError(f"No prediction files found in {RESULTS_DIR}")

    rows = []
    for pf in pred_files:
        m = FNAME_RE.match(pf.name)
        if m is None:
            print(f"  Skipping unrecognised filename: {pf.name}")
            continue

        dataset = m.group("dataset")   # e.g. K562
        mode    = m.group("mode")      # genept / esm2 / wikicrow / copath
        k       = int(m.group("k"))

        ds_key = dataset.lower()
        if ds_key not in dropped_perts_map:
            print(f"  WARNING: no dropped_perts entry for dataset '{dataset}' (key '{ds_key}') — skipping")
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

        y_true = df["label"].values
        y_pred = df["prediction"].values
        y_proba = df[["prob_class_0", "prob_class_1", "prob_class_2"]].values

        metrics = compute_3class_metrics(y_true, y_pred, y_proba=y_proba)

        row = {
            "model":          f"KNN_k{k}",
            "task":           "cross",
            "dataset":        dataset,
            "embedding_type": mode,
            "k":              k,
            "split":          "combined",
            "STATE_eval_set": True,
            "unique_genes":   df["gene"].nunique(),
            "unique_pert":    df["pert"].nunique(),
        }
        row.update(metrics)
        rows.append(row)

    if not rows:
        raise RuntimeError("No rows collected — check that prediction files match the expected naming pattern.")

    out_df = pd.DataFrame(rows)

    id_cols     = ["model", "task", "dataset", "embedding_type", "k",
                   "split", "STATE_eval_set", "unique_genes", "unique_pert"]
    metric_cols = [c for c in out_df.columns if c not in id_cols]
    out_df      = out_df[id_cols + metric_cols]

    out_df.to_csv(OUT_PATH, index=False)
    print(f"\nCollected {len(out_df)} rows.")
    print(f"Saved → {OUT_PATH}")
    print(f"\nDatasets:        {sorted(out_df['dataset'].unique())}")
    print(f"Embedding types: {sorted(out_df['embedding_type'].unique())}")
    print(f"k values:        {sorted(out_df['k'].unique())}")
    missing = set(["genept", "esm2", "wikicrow", "copath"]) - set(out_df["embedding_type"].unique())
    if missing:
        print(f"\nWARNING: missing results for embedding type(s): {sorted(missing)}")


if __name__ == "__main__":
    main()
