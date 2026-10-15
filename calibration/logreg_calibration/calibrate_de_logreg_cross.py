"""
calibrate_de_logreg_cross.py
============================
Post-DE calibration via multinomial logistic regression — cross-cell-line.
Fits logistic regression on the pooled calibration set from training cell lines,
then evaluates on the calib+test split of the held-out cell line.

Feature sets
------------
  v1 : [-log10(p_value), log2FC, freq_de_train]
  v2 : [-log10(p_value), log2FC, freq_de_pred_calib, freq_de_train]

Arguments
---------
  --pred_de_calib      Predicted DE CSV for calibration (training cell lines' calib perts)
  --pred_de_test       Predicted DE CSV for test (held-out cell line test+calib perts)
  --truth              Ground-truth CSV(s) for training cell lines (space-separated)
  --truth_test         Ground-truth CSV for the held-out cell line
  --train_splits       split_calib values used for freq_de_train (default: ['train'])
                       TranScouter/STATE: ['train', 'test']
  --truth_gene_features  Optional extra truth CSV(s) contributing additional rows to
                         freq_de_train (e.g. held-out cell line train split for STATE)
  --train_splits_extra   split_calib values for --truth_gene_features (default: ['train'])

Usage (TranScouter, k562 held-out)
-----------------------------------
    python calibrate_de_logreg_cross.py \\
        --pred_de_calib  cross_cond/de_label/k562/k562_training_calib_pooled.csv \\
        --pred_de_test   cross_cond/de_label/k562/k562-testcalib.csv \\
        --truth          DE_DOC_RPE1_splitted.csv DE_DOC_HepG2_splitted.csv DE_DOC_Jurkat_splitted.csv \\
        --truth_test     DE_DOC_K562_splitted.csv \\
        --train_splits   train test \\
        --model TranScouter --dataset k562 --feature_set v1

Usage (STATE, k562 held-out)
------------------------------
    python calibrate_de_logreg_cross.py \\
        --pred_de_calib  STATE/results/k562/de_task2_pred.csv \\
        --pred_de_test   STATE/results/k562/de_task1_merged_pred.csv \\
        --truth          DE_DOC_RPE1_splitted.csv DE_DOC_HepG2_splitted.csv DE_DOC_Jurkat_splitted.csv \\
        --truth_test     DE_DOC_K562_splitted.csv \\
        --truth_gene_features DE_DOC_K562_splitted.csv \\
        --train_splits   train test \\
        --train_splits_extra train \\
        --model STATE --dataset k562 --feature_set v1
"""

import argparse
import json
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


## Feature set registry

_FEATURE_REGISTRY: dict = {}

def register_feature_set(name):
    def decorator(fn):
        _FEATURE_REGISTRY[name] = fn
        return fn
    return decorator

@register_feature_set("v1")
def _feature_set_v1(df):
    cols = ["neg_log10_pval", "log2FC", "freq_de_train"]
    return df[cols].values, cols

@register_feature_set("v2")
def _feature_set_v2(df):
    cols = ["neg_log10_pval", "log2FC", "freq_de_pred_calib", "freq_de_train"]
    return df[cols].values, cols


def build_gene_features(train_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-gene DE frequency from pre-filtered training rows.
    """
    # counts every row per gene, so denominator is total number of (cell line, pert) observations for that gene
    total = train_rows.groupby("gene")["pert"].count().rename("total")
    # among these rows, how many are up
    up    = (train_rows[train_rows["label"] == 1].groupby("gene")["pert"].count()
             .rename("n_up").reindex(total.index, fill_value=0))
    # among these rows, how many are down
    down  = (train_rows[train_rows["label"] == 2].groupby("gene")["pert"].count()
             .rename("n_down").reindex(total.index, fill_value=0))
    gene_feats = pd.DataFrame({
        "freq_up_train":   up   / total,
        "freq_down_train": down / total,
    })
    gene_feats["freq_de_train"] = gene_feats["freq_up_train"] + gene_feats["freq_down_train"]
    return gene_feats.fillna(0.0)


def _add_pair_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["neg_log10_pval"] = -np.log10(df["p_value"].clip(lower=1e-300))
    return df


def evaluate(y_true, y_pred, label, n_pairs):
    metrics = compute_3class_metrics(y_true, y_pred)
    row = {"split": label, "n_pairs": n_pairs}
    row.update(metrics)
    print(f"\n  [{label}]  n={n_pairs}")
    print(f"    f1_macro          : {metrics['f1_macro']:.4f}")
    print(f"    balanced_accuracy : {metrics['balanced_accuracy']:.4f}")
    for ci, cname in enumerate(["Non-DE", "Up", "Down"]):
        print(f"    {cname:7s}  prec={metrics[f'precision_class_{ci}']:.4f}"
              f"  rec={metrics[f'recall_class_{ci}']:.4f}"
              f"  f1={metrics[f'f1_class_{ci}']:.4f}")
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_de_calib",        required=True)
    parser.add_argument("--pred_de_test",         required=True)
    parser.add_argument("--truth",                required=True, nargs='+')
    parser.add_argument("--truth_test",           required=True)
    parser.add_argument("--truth_gene_features",  nargs='*', default=[])
    parser.add_argument("--train_splits",         nargs='+', default=['train'])
    parser.add_argument("--train_splits_extra",   nargs='*', default=None)
    parser.add_argument("--model",                default="")
    parser.add_argument("--dataset",              default="")
    parser.add_argument("--calib_method",         default="")
    parser.add_argument("--feature_set",          default="v1", choices=list(_FEATURE_REGISTRY))
    parser.add_argument("--out_dir",              default="logreg_results")
    parser.add_argument("--gene_universe",        default=None, help="CSV with gene_name column; restrict all pairs to this set.")
    parser.add_argument("--dropped_calib_perts",  default=None, help="JSON keyed by dataset; perts to drop from pred_de_calib.")
    parser.add_argument("--dropped_test_perts",   default=None, help="JSON keyed by dataset; perts to drop from pred_de_test and truth_test.")
    args = parser.parse_args()

    print(f"Model={args.model}  Dataset={args.dataset}  "
          f"CalibMethod={args.calib_method}  FeatureSet={args.feature_set}")
    print(f"train_splits={args.train_splits}  "
          f"truth files={[os.path.basename(t) for t in args.truth]}")

    required_cols = {"pert", "gene", "label", "p_value", "log2FC"}

    ## take k562 as the held-out test cell line as an example ##
    ## ====================================================== ##

    pred_calib_df = pd.read_csv(args.pred_de_calib) # predictions on calibration perts of rpe1, hepg2, jurkat
    pred_test_df  = pd.read_csv(args.pred_de_test)  # predictions on calibration+test perts of k562
    for name, df in [("calib", pred_calib_df), ("test", pred_test_df)]:
        if missing := required_cols - set(df.columns):
            raise ValueError(f"{name} DE CSV missing columns: {missing}")

    truth_dfs     = [pd.read_csv(p) for p in args.truth]
    # k562 ground truth
    truth_test_df = pd.read_csv(args.truth_test)
    # rpe1 + hepg2 + jurkat ground truth CSVs concatenated
    truth_pool_df = pd.concat(truth_dfs, ignore_index=True)

    # Filter to gene universe and drop out-of-coverage perts
    if args.gene_universe:
        gene_set      = set(pd.read_csv(args.gene_universe, index_col=0)["gene_name"])
        pred_calib_df = pred_calib_df[pred_calib_df["gene"].isin(gene_set)]
        pred_test_df  = pred_test_df[pred_test_df["gene"].isin(gene_set)]
        truth_pool_df = truth_pool_df[truth_pool_df["gene"].isin(gene_set)]
        truth_test_df = truth_test_df[truth_test_df["gene"].isin(gene_set)]
        print(f"  Gene universe: {len(gene_set)} genes")

    if args.dropped_calib_perts:
        with open(args.dropped_calib_perts) as f:
            dropped_calib = set(json.load(f)[args.dataset])
        pred_calib_df = pred_calib_df[~pred_calib_df["pert"].isin(dropped_calib)]
        print(f"  Dropped {len(dropped_calib)} calib perts from pred_calib_df")

    if args.dropped_test_perts:
        with open(args.dropped_test_perts) as f:
            dropped_test = set(json.load(f)[args.dataset])
        pred_test_df  = pred_test_df[~pred_test_df["pert"].isin(dropped_test)]
        truth_test_df = truth_test_df[~truth_test_df["pert"].isin(dropped_test)]
        print(f"  Dropped {len(dropped_test)} test perts from pred_test_df and truth_test_df")

    # Build rows for freq_de_train: this is the exact same training set used for model training.
    # For STATE, this is 3 training cell lines filtered to train+test splits plus held-out cell line's train split.
    # For TranScouter, this is 3 training cell lines filtered to train+test splits.
    rows_for_gene_feats = truth_pool_df[truth_pool_df["split_calib"].isin(args.train_splits)]
    if args.truth_gene_features:
        extra_splits = args.train_splits_extra or ['train']
        extra_rows   = pd.concat(
            [pd.read_csv(p) for p in args.truth_gene_features],
            ignore_index=True,
        )
        extra_rows = extra_rows[extra_rows["split_calib"].isin(extra_splits)]
        rows_for_gene_feats = pd.concat([rows_for_gene_feats, extra_rows], ignore_index=True)

    print("\nComputing gene-level frequency features ...")
    gene_feats = build_gene_features(rows_for_gene_feats)
    print(f"  Genes with training stats: {gene_feats.shape[0]}")

    # freq_de_pred_calib: per-gene fraction of (cell_line, pert) pairs predicted as DE
    pred_calib_df = pred_calib_df.assign(_pair=pred_calib_df["cell_line"] + "__" + pred_calib_df["pert"])
    n_calib_perts = pred_calib_df["_pair"].nunique()
    freq_de_pred_calib = (
        pred_calib_df[pred_calib_df["label"] != 0]
        .groupby("gene")["_pair"].nunique()
        .rename("freq_de_pred_calib") / n_calib_perts
    )
    pred_calib_df = pred_calib_df.drop(columns=["_pair"])
    gene_feats = gene_feats.join(freq_de_pred_calib, how="left").fillna(0.0)

    def build_design_matrix(pred_df, truth_source, split_label):
        """
        Merge predicted DE with gene features and ground-truth labels.
        split_label may be a string or a list of strings.
        """
        labels = [split_label] if isinstance(split_label, str) else split_label
        truth_split = (truth_source[truth_source["split_calib"].isin(labels)]
                       [["pert", "gene", "label"]].rename(columns={"label": "y_true"}))
        pred_feats  = _add_pair_features(pred_df).merge(
            gene_feats.reset_index(), on="gene", how="left"
        ).fillna(0.0)
        merged = truth_split.merge(pred_feats, on=["pert", "gene"], how="inner")
        X, feat_names = _FEATURE_REGISTRY[args.feature_set](merged)
        return merged, X, merged["y_true"].values, feat_names

    # Fit on calibration split (3 training cell lines)
    calib_merged, X_calib, y_calib, feat_names = build_design_matrix(
        pred_calib_df, truth_pool_df, "calibration"
    )
    print(f"\nCalibration split: {len(calib_merged)} pairs  |  features: {feat_names}")
    print("Fitting logistic regression ...")
    clf = LogisticRegression(C=np.inf, class_weight="balanced",
                             solver="lbfgs", max_iter=1000)
    clf.fit(X_calib, y_calib)
    if max(clf.n_iter_) >= 1000:
        warnings.warn("Logistic regression did not converge!")

    param_rows = []
    for cls_idx, cls_name in enumerate(["Non-DE", "Up", "Down"]):
        coef_str = "  ".join(f"{f}={v:.4f}" for f, v in zip(feat_names, clf.coef_[cls_idx]))
        print(f"  {cls_name}: {coef_str}  bias={clf.intercept_[cls_idx]:.4f}")
        row = {"class": cls_name}
        row.update(dict(zip(feat_names, clf.coef_[cls_idx])))
        row["bias"] = clf.intercept_[cls_idx]
        param_rows.append(row)

    # Evaluate
    print(f"\n{'='*55}\n  Evaluation\n{'='*55}")
    rows = []
    y_pred_calib = clf.predict(X_calib)
    prob_calib   = clf.predict_proba(X_calib)
    rows.append(evaluate(y_calib, y_pred_calib, "calibration (in-sample)", len(calib_merged)))

    test_merged, X_test, y_test, _ = build_design_matrix(
        pred_test_df, truth_test_df, ["calibration", "test"]
    )
    print(f"\nTest split: {len(test_merged)} pairs")
    y_pred_test = clf.predict(X_test)
    prob_test   = clf.predict_proba(X_test)
    rows.append(evaluate(y_test, y_pred_test, "test", len(test_merged)))

    # Save outputs
    os.makedirs(args.out_dir, exist_ok=True)
    tag  = "_".join(filter(None, [args.calib_method, args.model, args.dataset, args.feature_set]))
    meta = {"model": args.model, "dataset": args.dataset,
            "calib_method": args.calib_method, "feature_set": args.feature_set}

    pd.DataFrame([{**meta, **r} for r in rows]).to_csv(
        os.path.join(args.out_dir, f"{tag}_metrics.csv"), index=False)
    pd.DataFrame([{**meta, **r} for r in param_rows]).to_csv(
        os.path.join(args.out_dir, f"{tag}_model_params.csv"), index=False)

    pred_cols = ["pert", "gene", "y_true", "y_pred_before", "y_pred",
                 "prob_non_de", "prob_up", "prob_down"]

    calib_merged["y_pred_before"] = calib_merged["label"].values
    calib_merged["y_pred"]        = y_pred_calib
    calib_merged[["prob_non_de", "prob_up", "prob_down"]] = prob_calib
    calib_merged[pred_cols].to_csv(
        os.path.join(args.out_dir, f"{tag}_calib_predictions.csv"), index=False)

    test_merged["y_pred_before"] = test_merged["label"].values
    test_merged["y_pred"]        = y_pred_test
    test_merged[["prob_non_de", "prob_up", "prob_down"]] = prob_test
    test_merged[pred_cols].to_csv(
        os.path.join(args.out_dir, f"{tag}_test_predictions.csv"), index=False)

    print(f"\nSaved to {args.out_dir}/  (tag: {tag})")
