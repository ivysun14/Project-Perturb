"""
compute_cross_filtered_metrics.py
===================================
Recompute GenePT cross-cell-line metrics after applying gene universe and
dropped-pert filters, for both the full and PerturbQA experiments.

Full dataset:
  - Predictions: KNN_full_cross_expr/results/{DS}_k{K}_knn_transfer_predictions_{calib,test}.csv
    (calib + test concatenated = equivalent to _combined)
  - Gene universe:  /home/jsun140/scratchyiqunchen/project_perturb_data/common_gene_universe.csv
  - Dropped perts:  /home/jsun140/scratchyiqunchen/project_perturb_data/dropped_test_perts.json
  - Output: KNN_full_cross_expr/cross_genept_metrics_filtered.csv

PerturbQA dataset:
  - Predictions: KNN_perturbQA_cross_expr/results/{DS}_genept_k{K}_knn_transfer_predictions.csv
  - Gene universe:  /home/jsun140/scratchyiqunchen/project_perturb_data/common_gene_universe_QA.csv
  - Dropped perts:  /home/jsun140/scratchyiqunchen/project_perturb_data/dropped_test_perts_QA.json
  - Output: KNN_perturbQA_cross_expr/cross_genept_metrics_filtered.csv

Verification:
  Checks that full-dataset metrics match the reference in drop_pert_gene_results/.
  If all 8 comparisons pass, deletes drop_pert_gene_results/ and logs/.

Run from KNN_vote/:
  python compute_cross_filtered_metrics.py
"""

import os
import sys
import json
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from majority_vote.analysis_utility import compute_3class_metrics

DATA_DIR = Path("/home/jsun140/scratchyiqunchen/project_perturb_data")
DATASETS = ["K562", "RPE1", "HepG2", "Jurkat"]
K_VALUES = [1, 10]


def compute_filtered_metrics_full(results_dir: Path,
                                  gene_universe_path: Path,
                                  dropped_perts_path: Path) -> pd.DataFrame:
    """
    Full cross-cell-line experiment (60/20/20 split).
    Combines calib + test prediction files, applies gene and pert filters,
    then recomputes metrics.
    """
    gene_universe = set(pd.read_csv(gene_universe_path)["gene_name"])
    with open(dropped_perts_path) as f:
        dropped_perts_map = json.load(f)

    rows = []
    for ds in DATASETS:
        ds_key = ds.lower()
        drop_set = set(dropped_perts_map[ds_key])
        for k in K_VALUES:
            calib_f = results_dir / f"{ds}_k{k}_knn_transfer_predictions_calib.csv"
            test_f  = results_dir / f"{ds}_k{k}_knn_transfer_predictions_test.csv"
            df = pd.concat([pd.read_csv(calib_f), pd.read_csv(test_f)], ignore_index=True)
            n_before = len(df)

            df = df[df["gene"].isin(gene_universe)]
            n_after_gene = len(df)

            df = df[~df["pert"].isin(drop_set)]
            n_after_pert = len(df)

            print(f"  {ds} | k={k}: {n_before} → {n_after_gene} (gene) "
                  f"→ {n_after_pert} (pert drop)  [{df['pert'].nunique()} perts]")

            y_true  = df["label"].values
            y_pred  = df["prediction"].values
            y_proba = df[["prob_class_0", "prob_class_1", "prob_class_2"]].values
            metrics = compute_3class_metrics(y_true, y_pred, y_proba=y_proba)

            rows.append({"dataset": ds, "k": k, **metrics})

    return pd.DataFrame(rows)


def compute_filtered_metrics_perturbqa(results_dir: Path,
                                       gene_universe_path: Path,
                                       dropped_perts_path: Path) -> pd.DataFrame:
    """
    PerturbQA cross-cell-line experiment (test split only).
    Applies gene and pert filters, then recomputes metrics.
    """
    gene_universe = set(pd.read_csv(gene_universe_path)["gene_name"])
    with open(dropped_perts_path) as f:
        dropped_perts_map = json.load(f)

    rows = []
    for ds in DATASETS:
        ds_key = ds.lower()
        drop_set = set(dropped_perts_map[ds_key])
        for k in K_VALUES:
            pred_f = results_dir / f"{ds}_genept_k{k}_knn_transfer_predictions.csv"
            df = pd.read_csv(pred_f)
            n_before = len(df)

            df = df[df["gene"].isin(gene_universe)]
            n_after_gene = len(df)

            df = df[~df["pert"].isin(drop_set)]
            n_after_pert = len(df)

            print(f"  {ds} | k={k}: {n_before} → {n_after_gene} (gene) "
                  f"→ {n_after_pert} (pert drop)  [{df['pert'].nunique()} perts]")

            y_true  = df["label"].values
            y_pred  = df["prediction"].values
            y_proba = df[["prob_class_0", "prob_class_1", "prob_class_2"]].values
            metrics = compute_3class_metrics(y_true, y_pred, y_proba=y_proba)

            rows.append({"dataset": ds, "k": k, **metrics})

    return pd.DataFrame(rows)


def verify_against_reference(full_df: pd.DataFrame,
                              ref_dir: Path,
                              tol: float = 1e-10) -> bool:
    """
    Check that full_df metrics match the 8 reference CSVs in ref_dir.
    Reference files have no AUROC; only compare the shared metric columns.
    Returns True if all comparisons pass.
    """
    ref_cols = ["accuracy", "balanced_accuracy", "precision_macro", "recall_macro",
                "f1_macro", "precision_class_0", "recall_class_0", "f1_class_0",
                "precision_class_1", "recall_class_1", "f1_class_1",
                "precision_class_2", "recall_class_2", "f1_class_2"]
    all_ok = True
    for k in K_VALUES:
        prefix = "1nn" if k == 1 else "10nn"
        for ds in DATASETS:
            ds_key = ds.lower()
            ref_f = ref_dir / f"{prefix}_metric_drop_perts_{ds_key}.csv"
            if not ref_f.exists():
                print(f"  MISSING reference file: {ref_f}")
                all_ok = False
                continue

            ref = pd.read_csv(ref_f).iloc[0]
            new = full_df[(full_df["dataset"] == ds) & (full_df["k"] == k)].iloc[0]

            for col in ref_cols:
                diff = abs(float(new[col]) - float(ref[col]))
                if diff > tol:
                    print(f"  MISMATCH {ds} k={k} {col}: new={new[col]:.10f} "
                          f"ref={ref[col]:.10f} diff={diff:.2e}")
                    all_ok = False

    return all_ok


def main():
    knn_dir = Path(__file__).parent

    # ── Full cross-cell-line ────────────────────────────────────────────────
    print("=" * 70)
    print("Full cross-cell-line experiment")
    print("=" * 70)
    full_results = knn_dir / "KNN_full_cross_expr" / "results"
    full_df = compute_filtered_metrics_full(
        results_dir      = full_results,
        gene_universe_path = DATA_DIR / "common_gene_universe.csv",
        dropped_perts_path = DATA_DIR / "dropped_test_perts.json",
    )
    full_out = knn_dir / "KNN_full_cross_expr" / "cross_genept_metrics_filtered.csv"
    full_df.to_csv(full_out, index=False)
    print(f"\nSaved → {full_out}")

    # ── Verify against reference ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Verifying against drop_pert_gene_results/")
    print("=" * 70)
    ref_dir = knn_dir / "drop_pert_gene_results"
    ok = verify_against_reference(full_df, ref_dir)
    if ok:
        print("  All metrics match reference exactly. Cleaning up old files.")
        shutil.rmtree(ref_dir)
        logs_dir = knn_dir / "logs"
        if logs_dir.exists():
            shutil.rmtree(logs_dir)
        print(f"  Deleted: {ref_dir}")
        print(f"  Deleted: {logs_dir}")
    else:
        print("  VERIFICATION FAILED — old files kept for inspection.")

    # ── PerturbQA cross-cell-line ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PerturbQA cross-cell-line experiment")
    print("=" * 70)
    qa_results = knn_dir / "KNN_perturbQA_cross_expr" / "results"
    qa_df = compute_filtered_metrics_perturbqa(
        results_dir        = qa_results,
        gene_universe_path = DATA_DIR / "common_gene_universe_QA.csv",
        dropped_perts_path = DATA_DIR / "dropped_test_perts_QA.json",
    )
    qa_out = knn_dir / "KNN_perturbQA_cross_expr" / "cross_genept_metrics_filtered.csv"
    qa_df.to_csv(qa_out, index=False)
    print(f"\nSaved → {qa_out}")

    # ── Print summaries ─────────────────────────────────────────────────────
    metric_cols = ["accuracy", "balanced_accuracy", "f1_macro"]
    print("\n── Full cross (filtered) ──────────────────────────────────────────")
    print(full_df[["dataset", "k"] + metric_cols].to_string(index=False))
    print("\n── PerturbQA cross (filtered) ─────────────────────────────────────")
    print(qa_df[["dataset", "k"] + metric_cols].to_string(index=False))


if __name__ == "__main__":
    main()
