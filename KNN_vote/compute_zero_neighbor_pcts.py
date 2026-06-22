"""
compute_zero_neighbor_pcts.py
==============================
Compute the percentage of (pert, gene) pairs that defaulted to class 0
due to zero valid neighbors, for both within- and cross-cell-line sensitivity
experiments.

Zero-neighbor pairs are identified by
  prob_class_0 == prob_class_1 == prob_class_2 == 1/3
which is the fixed uniform probability assigned to no-neighbor fallbacks in
both knn_classifier.py and knn_classifier_transfer.py.

For cross experiments, pairs are first restricted to the common gene universe
and the per-dataset excluded perturbation list (same filters as
collect_cross_sensitivity_metrics.py).

For within experiments, no post-hoc filtering is applied (coverage_filter was
already enforced at run time).

Run from KNN_vote/:
  python compute_zero_neighbor_pcts.py
"""

import re
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────────
WITHIN_RESULTS = Path("KNN_sensitivity_full_within_expr/results")
CROSS_RESULTS  = Path("KNN_sensitivity_full_cross_expr/results")
DATA_DIR       = Path("/home/jsun140/scratchyiqunchen/project_perturb_data")
GENE_UNIVERSE_CSV  = DATA_DIR / "common_gene_universe.csv"
DROPPED_PERTS_JSON = DATA_DIR / "dropped_test_perts.json"

WITHIN_RE = re.compile(
    r"^(?P<dataset>.+?)_(?P<mode>genept|esm2|wikicrow|copath)_k(?P<k>\d+)"
    r"_covfilt_knn_predictions_combined\.csv$"
)
CROSS_RE = re.compile(
    r"^(?P<dataset>.+?)_(?P<mode>genept|esm2|wikicrow|copath)_k(?P<k>\d+)"
    r"_covfilt_knn_transfer_predictions_combined\.csv$"
)

ZERO_NB_PROB = 1.0 / 3.0
ZERO_NB_TOL  = 1e-9
PROB_COLS    = ["prob_class_0", "prob_class_1", "prob_class_2"]


def is_zero_neighbor(chunk: pd.DataFrame) -> pd.Series:
    return (
        (np.abs(chunk["prob_class_0"] - ZERO_NB_PROB) < ZERO_NB_TOL) &
        (np.abs(chunk["prob_class_1"] - ZERO_NB_PROB) < ZERO_NB_TOL) &
        (np.abs(chunk["prob_class_2"] - ZERO_NB_PROB) < ZERO_NB_TOL)
    )


def count_zero_nb(pf: Path,
                  gene_filter: set | None = None,
                  pert_drop: set | None = None) -> tuple[int, int]:
    """Return (n_total, n_zero_nb) after optional gene/pert filters."""
    n_total, n_zero_nb = 0, 0
    read_cols = ["pert", "gene"] + PROB_COLS
    for chunk in pd.read_csv(pf, chunksize=200_000, usecols=read_cols):
        if gene_filter is not None:
            chunk = chunk[chunk["gene"].isin(gene_filter)]
        if pert_drop is not None:
            chunk = chunk[~chunk["pert"].isin(pert_drop)]
        n_total   += len(chunk)
        n_zero_nb += is_zero_neighbor(chunk).sum()
    return n_total, n_zero_nb


def print_pivot(df: pd.DataFrame, title: str) -> None:
    print(f"\n{'=' * 72}")
    print(title)
    print(f"{'=' * 72}")
    pivot = df.pivot_table(index=["dataset", "mode"], columns="k",
                           values="pct_zero_nb", aggfunc="first")
    pivot.columns = [f"k={c}" for c in pivot.columns]
    print(pivot.to_string(float_format=lambda x: f"{x:.4f}%"))


def compute_within(rows: list) -> None:
    print("\n── Within-cell-line ──────────────────────────────────────────────────")
    pred_files = sorted(WITHIN_RESULTS.glob("*_knn_predictions_combined.csv"))
    if not pred_files:
        print(f"  No files found in {WITHIN_RESULTS}")
        return

    for pf in pred_files:
        m = WITHIN_RE.match(pf.name)
        if m is None:
            continue
        dataset, mode, k = m.group("dataset"), m.group("mode"), int(m.group("k"))

        n_total, n_zero_nb = count_zero_nb(pf)
        pct = 100.0 * n_zero_nb / n_total if n_total > 0 else float("nan")
        rows.append({"task": "within", "dataset": dataset, "mode": mode, "k": k,
                     "total_pairs": n_total, "zero_nb_pairs": n_zero_nb,
                     "pct_zero_nb": pct})
        print(f"  {dataset:8s} | {mode:8s} | k={k:2d} | "
              f"total={n_total:>9,} | zero_nb={n_zero_nb:>7,} | {pct:.4f}%")


def compute_cross(rows: list,
                  gene_universe: set,
                  dropped_perts_map: dict) -> None:
    print("\n── Cross-cell-line (after common gene + dropped pert filtering) ───────")
    pred_files = sorted(CROSS_RESULTS.glob("*_knn_transfer_predictions_combined.csv"))
    if not pred_files:
        print(f"  No files found in {CROSS_RESULTS}")
        return

    for pf in pred_files:
        m = CROSS_RE.match(pf.name)
        if m is None:
            continue
        dataset, mode, k = m.group("dataset"), m.group("mode"), int(m.group("k"))
        ds_key = dataset.lower()

        if ds_key not in dropped_perts_map:
            print(f"  WARNING: no dropped_perts entry for '{dataset}' — skipping")
            continue

        drop_set = set(dropped_perts_map[ds_key])
        n_total, n_zero_nb = count_zero_nb(pf, gene_filter=gene_universe,
                                           pert_drop=drop_set)
        pct = 100.0 * n_zero_nb / n_total if n_total > 0 else float("nan")
        rows.append({"task": "cross", "dataset": dataset, "mode": mode, "k": k,
                     "total_pairs": n_total, "zero_nb_pairs": n_zero_nb,
                     "pct_zero_nb": pct})
        print(f"  {dataset:8s} | {mode:8s} | k={k:2d} | "
              f"total={n_total:>9,} | zero_nb={n_zero_nb:>7,} | {pct:.4f}%")


def main():
    gene_universe = set(pd.read_csv(GENE_UNIVERSE_CSV)["gene_name"])
    with open(DROPPED_PERTS_JSON) as f:
        dropped_perts_map = json.load(f)

    rows: list[dict] = []
    compute_within(rows)
    compute_cross(rows, gene_universe, dropped_perts_map)

    df = pd.DataFrame(rows).sort_values(["task", "dataset", "mode", "k"])

    within_df = df[df["task"] == "within"]
    cross_df  = df[df["task"] == "cross"]

    print_pivot(within_df,
                "Within-cell-line: % (pert,gene) pairs defaulted to class 0")
    print_pivot(cross_df,
                "Cross-cell-line (filtered): % (pert,gene) pairs defaulted to class 0")

    within_out = WITHIN_RESULTS.parent / "zero_neighbor_pcts.csv"
    cross_out  = CROSS_RESULTS.parent  / "zero_neighbor_pcts.csv"

    within_df.drop(columns="task").to_csv(within_out, index=False)
    cross_df.drop(columns="task").to_csv(cross_out, index=False)
    print(f"\nWithin saved → {within_out}")
    print(f"Cross  saved → {cross_out}")


if __name__ == "__main__":
    main()
