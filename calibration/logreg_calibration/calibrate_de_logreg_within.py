"""
calibrate_de_logreg.py
======================
Post-DE calibration via multinomial logistic regression.

A logistic regression is trained on the calibration split to map predicted
DE features to ground-truth DE labels {0: Non-DE, 1: Up, 2: Down}.

Feature sets
---------------------------------
v1 : [-log10(p_value), log2FC, freq_de_train]
v2 : [-log10(p_value), log2FC, freq_de_pred_calib, freq_de_train]

  -log10(p_value)  : from running DE calling on the predicted h5ad
  log2FC           : from the same predicted DE output
  freq_de_train      : per-gene fraction of TRAINING perturbations where the gene
                       is DE (label != 0), from the ground-truth splitted CSV
  freq_de_pred_calib : per-gene fraction of CALIBRATION perturbations where the
                       model predicts DE (label != 0), from the predicted DE CSV

Usage
-----
    python calibrate_de_logreg.py \\
        --pred_de_calib  hepg2-calib.csv \\
        --pred_de_test   hepg2-test.csv \\
        --truth          ../data/DE_DOC_hepg2_splitted.csv \\
        --model          SouterWithin \\
        --dataset        hepg2 \\
        --calib_method   none \\
        --feature_set    v1 \\
        --out_dir        logreg_results
"""

import argparse
import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(_project_root)
sys.path.append(os.path.join(_project_root, "majority_vote"))
from metrics.class_metrics import compute_3class_metrics

# ============================================================
# Feature set registry
# ============================================================
# Each registered function receives a DataFrame that already contains ALL
# possible feature columns (both pair-level and gene-level).  It returns
# (X: np.ndarray, feature_names: list[str]).
#
# To add a new feature set: decorate a function with @register_feature_set("vN").
# To add a new raw feature: compute it in build_gene_features() or in
# _add_pair_features(), then reference it in any feature set function.
# ============================================================

_FEATURE_REGISTRY: dict = {}

def register_feature_set(name: str):
    def decorator(fn):
        _FEATURE_REGISTRY[name] = fn
        return fn
    return decorator


@register_feature_set("v1")
def _feature_set_v1(df: pd.DataFrame):
    """[-log10(p_value), log2FC, freq_de_train]"""
    cols = ["neg_log10_pval", "log2FC", "freq_de_train"]
    return df[cols].values, cols


@register_feature_set("v2")
def _feature_set_v2(df: pd.DataFrame):
    """[-log10(p_value), log2FC, freq_de_pred_calib, freq_de_train]"""
    cols = ["neg_log10_pval", "log2FC", "freq_de_pred_calib", "freq_de_train"]
    return df[cols].values, cols
# ============================================================

def build_gene_features(truth_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute gene-level frequency features from the TRAINING split of the
    ground-truth CSV.  Returns a DataFrame indexed by gene with columns:
        freq_de_train, freq_up_train, freq_down_train
    """
    train = truth_df[truth_df["split_calib"] == "train"].copy()
    total = train.groupby("gene")["pert"].count().rename("total")
    up    = (train[train["label"] == 1].groupby("gene")["pert"].count()
             .rename("n_up").reindex(total.index, fill_value=0))
    down  = (train[train["label"] == 2].groupby("gene")["pert"].count()
             .rename("n_down").reindex(total.index, fill_value=0))

    gene_feats = pd.DataFrame({
        "freq_up_train":   up   / total,
        "freq_down_train": down / total,
    })
    gene_feats["freq_de_train"] = gene_feats["freq_up_train"] + gene_feats["freq_down_train"]
    gene_feats = gene_feats.fillna(0.0)
    return gene_feats


def _add_pair_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add pair-level derived columns to a predicted DE DataFrame."""
    df = df.copy()
    # prevent numerical overflow, -log10(0) is undefined -> +infty
    df["neg_log10_pval"] = -np.log10(df["p_value"].clip(lower=1e-300))
    return df


def evaluate(y_true: np.ndarray, y_pred: np.ndarray,
             label: str, n_pairs: int) -> dict:
    metrics = compute_3class_metrics(y_true, y_pred)
    row = {"split": label, "n_pairs": n_pairs}
    row.update(metrics)
    print(f"\n  [{label}]  n={n_pairs}")
    print(f"    f1_macro          : {metrics['f1_macro']:.4f}")
    print(f"    balanced_accuracy : {metrics['balanced_accuracy']:.4f}")
    for ci, cname in enumerate(["Non-DE", "Up", "Down"]):
        p = metrics[f"precision_class_{ci}"]
        r = metrics[f"recall_class_{ci}"]
        f = metrics[f"f1_class_{ci}"]
        print(f"    {cname:7s}  prec={p:.4f}  rec={r:.4f}  f1={f:.4f}")
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-DE logistic regression calibration.")
    parser.add_argument("--pred_de_calib", required=True, help="Predicted DE CSV for the calibration split.")
    parser.add_argument("--pred_de_test",  required=True, help="Predicted DE CSV for the test split.")
    parser.add_argument("--truth", required=True, help="Ground-truth splitted CSV (must have split_calib column).")
    parser.add_argument("--model", default="", help="Model label for output naming.")
    parser.add_argument("--dataset", default="", help="Dataset label for output naming.")
    parser.add_argument("--calib_method", default="", help="Calibration method label (var/sparsity/none).")
    parser.add_argument("--feature_set", default="v1", choices=list(_FEATURE_REGISTRY), help="Feature set to use.")
    parser.add_argument("--out_dir", default="logreg_results", help="Directory for output CSVs.")
    args = parser.parse_args()

    print(f"Model={args.model}  Dataset={args.dataset}  "
          f"CalibMethod={args.calib_method}  FeatureSet={args.feature_set}")

    required_cols = {"pert", "gene", "label", "p_value", "log2FC"}
    print(f"\nLoading calibration predicted DE from {args.pred_de_calib} ...")
    pred_calib_df = pd.read_csv(args.pred_de_calib)
    missing = required_cols - set(pred_calib_df.columns)
    if missing:
        raise ValueError(f"Calibration DE CSV missing columns: {missing}")

    print(f"Loading test predicted DE from {args.pred_de_test} ...")
    pred_test_df = pd.read_csv(args.pred_de_test)
    missing = required_cols - set(pred_test_df.columns)
    if missing:
        raise ValueError(f"Test DE CSV missing columns: {missing}")

    print(f"Loading ground truth from {args.truth} ...")
    truth_df = pd.read_csv(args.truth)
    if "split_calib" not in truth_df.columns:
        raise ValueError("Ground-truth CSV has no 'split_calib' column. "
                         "Run calibration_split.py first.")

    print("\nComputing gene-level frequency features from training split ...")
    gene_feats = build_gene_features(truth_df)
    print(f"  Genes with training stats: {gene_feats.shape[0]}")

    # freq_de_pred_calib: per-gene fraction of calibration perturbations where
    # the model predicts DE — captures model's systematic gene-level calling tendency
    n_calib_perts = pred_calib_df["pert"].nunique()
    freq_de_pred_calib = (
        pred_calib_df[pred_calib_df["label"] != 0]
        .groupby("gene")["pert"].nunique()
        .rename("freq_de_pred_calib") / n_calib_perts
    )
    gene_feats = gene_feats.join(freq_de_pred_calib, how="left").fillna(0.0)

    def build_split(split_label: str, pred_df: pd.DataFrame):
        """Return (merged_df, X, y, feat_names) for a given split_calib label."""
        truth_split = (truth_df[truth_df["split_calib"] == split_label]
                       [["pert", "gene", "label"]].rename(columns={"label": "y_true"}))
        pred_split = _add_pair_features(pred_df.copy())
        pred_split = pred_split.merge(gene_feats.reset_index(), on="gene", how="left").fillna(0.0)
        merged = truth_split.merge(pred_split, on=["pert", "gene"], how="inner")
        X, feat_names = _FEATURE_REGISTRY[args.feature_set](merged)
        y = merged["y_true"].values
        return merged, X, y, feat_names

    # build model using calibration split
    calib_merged, X_calib_final, y_calib_final, feat_names = build_split("calibration", pred_calib_df)
    print(f"\nCalibration split: {len(calib_merged)} matched pairs")
    print(f"Features used    : {feat_names}")
    print("\nFitting logistic regression ...")
    clf = LogisticRegression(
        C=np.inf,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=1000,
    )
    clf.fit(X_calib_final, y_calib_final)
    if max(clf.n_iter_) >= 1000:
        warnings.warn("Logistic regression did not converge!.")

    print(f"  Coefficients (classes 0/1/2):")
    param_rows = []
    for cls_idx, cls_name in enumerate(["Non-DE", "Up", "Down"]):
        coef_str = "  ".join(f"{feat}={v:.4f}"
                             for feat, v in zip(feat_names, clf.coef_[cls_idx]))
        print(f"    {cls_name}: {coef_str}  bias={clf.intercept_[cls_idx]:.4f}")
        row = {"class": cls_name}
        row.update(dict(zip(feat_names, clf.coef_[cls_idx])))
        row["bias"] = clf.intercept_[cls_idx]
        param_rows.append(row)

    
    print(f"\n{'='*55}")
    print("  Evaluation")
    print(f"{'='*55}")

    rows = []

    # evaluate on calibration split (in-sample sanity check)
    y_pred_calib = clf.predict(X_calib_final)
    prob_calib   = clf.predict_proba(X_calib_final)
    rows.append(evaluate(y_calib_final, y_pred_calib, "calibration (in-sample)", len(calib_merged)))

    # evaluate on test split
    test_merged, X_test_final, y_test_final, _ = build_split("test", pred_test_df)
    print(f"\nTest split: {len(test_merged)} matched pairs")
    y_pred_test = clf.predict(X_test_final)
    prob_test   = clf.predict_proba(X_test_final)
    rows.append(evaluate(y_test_final, y_pred_test, "test", len(test_merged)))

    # save results
    os.makedirs(args.out_dir, exist_ok=True)
    tag = "_".join(filter(None, [args.calib_method, args.model, args.dataset, args.feature_set]))
    metrics_path = os.path.join(args.out_dir, f"{tag}_metrics.csv")
    meta = {"model": args.model, "dataset": args.dataset, "calib_method": args.calib_method, "feature_set": args.feature_set}
    metrics_df = pd.DataFrame([{**meta, **r} for r in rows])
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nMetrics saved to {metrics_path}")

    params_path = os.path.join(args.out_dir, f"{tag}_model_params.csv")
    pd.DataFrame([{**meta, **r} for r in param_rows]).to_csv(params_path, index=False)
    print(f"Model parameters saved to {params_path}")

    pred_cols = ["pert", "gene", "y_true", "y_pred_before", "y_pred", "prob_non_de", "prob_up", "prob_down"]

    calib_merged["y_pred_before"] = calib_merged["label"].values
    calib_merged["y_pred"] = y_pred_calib
    calib_merged[["prob_non_de", "prob_up", "prob_down"]] = prob_calib
    calib_preds_path = os.path.join(args.out_dir, f"{tag}_calib_predictions.csv")
    calib_merged[pred_cols].to_csv(calib_preds_path, index=False)
    print(f"Calibration predictions saved to {calib_preds_path}")

    test_merged["y_pred_before"] = test_merged["label"].values
    test_merged["y_pred"] = y_pred_test
    test_merged[["prob_non_de", "prob_up", "prob_down"]] = prob_test
    preds_path = os.path.join(args.out_dir, f"{tag}_test_predictions.csv")
    test_merged[pred_cols].to_csv(preds_path, index=False)
    print(f"Test predictions saved to {preds_path}")
