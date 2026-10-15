"""
metric_drop_perts.py
====================
Recompute logistic-regression cross-cell-line metrics after applying gene
universe and dropped-pert filters, for both the full and PerturbQA experiments.

Full dataset:
  - Predictions: logreg_0.01_full_cross_expr/results/
                 {DS}_logreg_transfer_logreg_predictions_{calib,test}.csv
  - Gene universe:  /projects/ychen756/project_perturb_data/common_gene_universe.csv
  - Dropped perts:  /projects/ychen756/project_perturb_data/dropped_test_perts.json
  - Output: logreg_0.01_full_cross_expr/cross_logreg_metrics_filtered.csv

PerturbQA dataset:
  - Predictions: logreg_0.01_perturbQA_cross_expr/results/
                 {DS}_logreg_transfer_predictions.csv
  - Gene universe:  /projects/ychen756/project_perturb_data/common_gene_universe_QA.csv
  - Dropped perts:  /projects/ychen756/project_perturb_data/dropped_test_perts_QA.json
  - Output: logreg_0.01_perturbQA_cross_expr/cross_logreg_metrics_filtered.csv

Run from logistic_reg/:
  python metric_drop_perts.py
"""

import json
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from metrics.class_metrics import compute_3class_metrics

DATA_DIR = Path("/projects/ychen756/project_perturb_data")
DATASETS = ["K562", "RPE1", "HepG2", "Jurkat"]


def compute_filtered_metrics_full(results_dir: Path,
                                  gene_universe_path: Path,
                                  dropped_perts_path: Path) -> pd.DataFrame:
    """
    Full cross-cell-line experiment.
    Concatenates calib+test predictions (combined file not saved in results/),
    applies gene and pert filters, then recomputes metrics with probabilities.
    """
    gene_universe = set(pd.read_csv(gene_universe_path)["gene_name"])
    with open(dropped_perts_path) as f:
        dropped_perts_map = json.load(f)

    rows = []
    for ds in DATASETS:
        ds_key = ds.lower()
        drop_set = set(dropped_perts_map[ds_key])

        calib_f = results_dir / f"{ds}_logreg_transfer_logreg_predictions_calib.csv"
        test_f  = results_dir / f"{ds}_logreg_transfer_logreg_predictions_test.csv"
        df = pd.concat([pd.read_csv(calib_f), pd.read_csv(test_f)], ignore_index=True)
        n_before = len(df)

        df = df[df["gene"].isin(gene_universe)]
        n_after_gene = len(df)

        df = df[~df["pert"].isin(drop_set)]
        n_after_pert = len(df)

        print(f"  {ds}: {n_before} → {n_after_gene} (gene) "
              f"→ {n_after_pert} (pert drop)  [{df['pert'].nunique()} perts]")

        y_true  = df["label"].values
        y_pred  = df["prediction"].values
        y_proba = df[["prob_class_0", "prob_class_1", "prob_class_2"]].values
        metrics = compute_3class_metrics(y_true, y_pred, y_proba=y_proba)

        rows.append({"dataset": ds, **metrics})

    return pd.DataFrame(rows)


def compute_filtered_metrics_perturbqa(results_dir: Path,
                                       gene_universe_path: Path,
                                       dropped_perts_path: Path) -> pd.DataFrame:
    """
    PerturbQA cross-cell-line experiment.
    Reads held-out test split predictions, applies gene and pert filters,
    then recomputes metrics with probabilities.
    """
    gene_universe = set(pd.read_csv(gene_universe_path)["gene_name"])
    with open(dropped_perts_path) as f:
        dropped_perts_map = json.load(f)

    rows = []
    for ds in DATASETS:
        ds_key = ds.lower()
        drop_set = set(dropped_perts_map[ds_key])

        pred_f = results_dir / f"{ds}_logreg_transfer_predictions.csv"
        df = pd.read_csv(pred_f)
        n_before = len(df)

        df = df[df["gene"].isin(gene_universe)]
        n_after_gene = len(df)

        df = df[~df["pert"].isin(drop_set)]
        n_after_pert = len(df)

        print(f"  {ds}: {n_before} → {n_after_gene} (gene) "
              f"→ {n_after_pert} (pert drop)  [{df['pert'].nunique()} perts]")

        y_true  = df["label"].values
        y_pred  = df["prediction"].values
        y_proba = df[["prob_class_0", "prob_class_1", "prob_class_2"]].values
        metrics = compute_3class_metrics(y_true, y_pred, y_proba=y_proba)

        rows.append({"dataset": ds, **metrics})

    return pd.DataFrame(rows)


def main():
    lr_dir = Path(__file__).parent

    # ── Full cross-cell-line ────────────────────────────────────────────────
    print("=" * 70)
    print("Full cross-cell-line experiment")
    print("=" * 70)
    full_results = lr_dir / "logreg_0.01_full_cross_expr" / "results"
    full_df = compute_filtered_metrics_full(
        results_dir        = full_results,
        gene_universe_path = DATA_DIR / "common_gene_universe.csv",
        dropped_perts_path = DATA_DIR / "dropped_test_perts.json",
    )
    full_out = lr_dir / "logreg_0.01_full_cross_expr" / "cross_logreg_metrics_filtered.csv"
    full_df.to_csv(full_out, index=False)
    print(f"\nSaved → {full_out}")

    # ── PerturbQA cross-cell-line ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PerturbQA cross-cell-line experiment")
    print("=" * 70)
    qa_results = lr_dir / "logreg_0.01_perturbQA_cross_expr" / "results"
    qa_df = compute_filtered_metrics_perturbqa(
        results_dir        = qa_results,
        gene_universe_path = DATA_DIR / "common_gene_universe_QA.csv",
        dropped_perts_path = DATA_DIR / "dropped_test_perts_QA.json",
    )
    qa_out = lr_dir / "logreg_0.01_perturbQA_cross_expr" / "cross_logreg_metrics_filtered.csv"
    qa_df.to_csv(qa_out, index=False)
    print(f"\nSaved → {qa_out}")

    # ── Print summaries ─────────────────────────────────────────────────────
    metric_cols = ["accuracy", "balanced_accuracy", "f1_macro"]
    print("\n── Full cross (filtered) ──────────────────────────────────────────")
    print(full_df[["dataset"] + metric_cols].to_string(index=False))
    print("\n── PerturbQA cross (filtered) ─────────────────────────────────────")
    print(qa_df[["dataset"] + metric_cols].to_string(index=False))


if __name__ == "__main__":
    main()
