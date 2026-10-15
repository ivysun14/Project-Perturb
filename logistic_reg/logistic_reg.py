"""
logistic_reg.py
===============
Multinomial logistic regression for 3-class perturbation-gene prediction.
The first model: Logistic regression using GenePT (GenePert with logistic head)

Model: logit(Y) = X @ B
  X : (n x m)  — n unique training perturbations, m=3072 GenePT embedding dim
  Y : (n x p)  — label matrix; n perturbations x p genes; each cell in {0,1,2}
  B : (m x p)  — one coefficient vector of length m per gene (fit separately)

One multinomial logistic regression is fit per gene using the perturbation
embeddings of all training perturbations that have a label for that gene.
Test prediction for pair (pert, gene): apply that gene's model to embedding of pert.

Classes:
  0 = Non-DE
  1 = Up-regulated
  2 = Down-regulated

Modes:
  Within-cell-line:  --de_doc_file <single CSV with split_calib column>
  Cross-cell-line:   --train_de_doc_files <csv1,csv2,csv3> --test_de_doc_file <csv4>
                     Training uses train+test splits (80%) of source cell lines.
                     Evaluation on held-out cell line's calibration, test, and combined.

Input CSV format (from DE_DOC_stats.py):
  pert, gene, label, split_calib
"""

import os
import sys
import json
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import classification_report
from sklearn.exceptions import ConvergenceWarning

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics.class_metrics import compute_3class_metrics
from majority_vote.analysis_utility import plot_data_summary
from majority_vote.error_utility import run_error_analysis_from_results


def load_embeddings(primary_path: str, secondary_path: str = None) -> dict:
    """
    Load GenePT embedding pkl file(s) into a single dict {gene: np.ndarray}.
    """
    with open(primary_path, 'rb') as f:
        emb = pickle.load(f)
    emb = {k: np.array(v, dtype=np.float32) for k, v in emb.items()}
    if secondary_path is not None:
        with open(secondary_path, 'rb') as f:
            secondary = pickle.load(f)
        secondary = {k: np.array(v, dtype=np.float32) for k, v in secondary.items() if k not in emb}
        emb.update(secondary)
        print(f"Merged secondary embeddings: {len(secondary)} new genes added")
    print(f"Total embeddings loaded: {len(emb)} genes")
    return emb


def build_feature_matrix(pert_names: list, embedding_dict: dict):
    """
    Build X of shape (n, 3072) for a list of perturbation names.

    Args:
        pert_names: ordered list of perturbation names
        embedding_dict: dict mapping name -> np.ndarray of shape (3072,)

    Returns:
        X:          np.ndarray of shape (n_valid, 3072)
        valid_mask: bool array of shape (n,); True where embedding exists
        valid_perts: list of pert names retained (in order)
    """
    missing = set()
    rows, valid = [], []
    for p in pert_names:
        if p not in embedding_dict:
            missing.add(p)
            valid.append(False)
        else:
            rows.append(embedding_dict[p])
            valid.append(True)
    if missing:
        warnings.warn(
            f"{len(missing)} perturbations not found in embedding dict and "
            f"will be dropped: {sorted(missing)}"
        )
    valid_mask = np.array(valid, dtype=bool)
    X = np.stack(rows, axis=0)  # (n_valid, 3072)
    valid_perts = [p for p, v in zip(pert_names, valid) if v]
    return X, valid_mask, valid_perts


def build_label_matrix(df: pd.DataFrame, pert_names: list, gene_names: list) -> np.ndarray:
    """
    Build Y of shape (n_perts, n_genes) from a pairs DataFrame.
    Cells with no observed label are NaN.

    Args:
        df:         DataFrame with columns [pert, gene, label]
        pert_names: ordered list of perturbation names (rows of Y)
        gene_names: ordered list of gene names (columns of Y)

    Returns:
        Y: np.ndarray of shape (n_perts, n_genes), dtype float64
    """
    pert_to_idx = {p: i for i, p in enumerate(pert_names)}
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    Y = np.full((len(pert_names), len(gene_names)), np.nan, dtype=np.float64)
    for _, row in df.iterrows():
        p, g = row['pert'], row['gene']
        if p in pert_to_idx and g in gene_to_idx:
            Y[pert_to_idx[p], gene_to_idx[g]] = row['label']
    return Y


def merge_train_sets(train_dfs: list, random_seed: int) -> pd.DataFrame:
    """
    Merge training DataFrames from multiple cell lines into a single train set.

    For each (pert, gene) pair that appears in more than one cell line, the label
    is decided by majority vote.  When two or more labels tie, a warning is printed
    and the tie is broken by drawing uniformly at random from the tied classes.

    Args:
        train_dfs:   list of DataFrames, each with columns [pert, gene, label]
        random_seed: seed for the random tie-breaker

    Returns:
        merged DataFrame with columns [pert, gene, label] — one row per unique pair
    """
    rng = np.random.default_rng(random_seed)
    combined = pd.concat(
        [df[['pert', 'gene', 'label']] for df in train_dfs],
        ignore_index=True,
    )

    records = []
    tie_pairs = []

    for (pert, gene), group in combined.groupby(['pert', 'gene'], sort=False):
        counts = group['label'].value_counts()
        max_count = counts.max()
        top_labels = counts[counts == max_count].index.tolist()
        if len(top_labels) > 1:
            tie_pairs.append((pert, gene, sorted(int(l) for l in top_labels)))
            chosen = int(rng.choice(top_labels))
        else:
            chosen = int(top_labels[0])
        records.append({'pert': pert, 'gene': gene, 'label': chosen})

    if tie_pairs:
        print(f"\n[WARNING] {len(tie_pairs)} (pert, gene) pair(s) had label ties — broken randomly:")
        for pert, gene, labels in tie_pairs[:20]:
            print(f"  ({pert}, {gene}): tied classes {labels}")
        if len(tie_pairs) > 20:
            print(f"  ... and {len(tie_pairs) - 20} more ties not shown")

    merged = pd.DataFrame(records)
    print(f"Merged train set: {len(merged)} unique (pert, gene) pairs "
          f"from {len(train_dfs)} cell lines  |  {len(tie_pairs)} ties broken")
    return merged


def fit_per_gene_models(X_train: np.ndarray, Y_train: np.ndarray,
                        gene_names: list, Cs: list, cv: int,
                        max_iter: int, random_seed: int) -> dict:
    """
    Fit one multinomial logistic regression per gene column of Y_train,
    Genes with <2 training observations are skipped. Genes with only one
    distinct class in training are stored as a constant predictor ('constant', class_int)
    and always predict that class at test time.

    Args:
        X_train:    (n_perts, m) perturbation embedding matrix
        Y_train:    (n_perts, p) label matrix; NaN = no observation
        gene_names: ordered list of gene names, length p
        Cs:         list of C values to search over (C = 1/lambda)
        cv:         number of cross-validation folds
        max_iter, random_seed: solver hyperparameters

    Returns:
        models: dict mapping gene_name -> fitted LogisticRegressionCV, or
                                          ('constant', int) for single-class genes
    """
    models = {}
    n_skipped      = 0
    n_constant     = 0
    n_nonconverged = 0
    for gi, gene in enumerate(gene_names):
        y_g = Y_train[:, gi]
        observed = ~np.isnan(y_g)
        if observed.sum() < 2:
            n_skipped += 1
            continue
        if len(np.unique(y_g[observed])) < 2:
            # Only one class seen in training — store as constant predictor
            models[gene] = ('constant', int(y_g[observed][0]))
            n_constant += 1
            continue
        if len(Cs) == 1:
            # Single C provided — skip CV entirely
            clf = LogisticRegression(
                C=Cs[0],
                l1_ratio=0,         # l2 penalty
                solver='lbfgs',
                class_weight='balanced',
                max_iter=max_iter,
                random_state=random_seed,
            )
        else:
            # Multiple C's - conduct cross validation
            clf = LogisticRegressionCV(
                Cs=Cs,
                cv=cv,
                l1_ratios=(0,),     # l2 penalty
                scoring='balanced_accuracy',
                solver='lbfgs',
                class_weight='balanced',
                max_iter=max_iter,
                random_state=random_seed,
                refit=True,
            )
        clf.fit(X_train[observed], y_g[observed].astype(int))
        models[gene] = clf
        if np.any(clf.n_iter_ >= max_iter):
            n_nonconverged += 1

    n_fitted = len(models) - n_constant
    print(f"  Fitted models for {n_fitted}/{len(gene_names)} genes  |  "
          f"constant predictor: {n_constant}  |  skipped (<2 obs): {n_skipped}")
    if n_nonconverged > 0:
        warnings.warn(
            f"{n_nonconverged}/{n_fitted} gene models may not have converged "
            f"(hit max_iter={max_iter}). Consider increasing --max_iter.",
            ConvergenceWarning,
            stacklevel=2,
        )
    return models


def predict_pairs(test_df: pd.DataFrame, models: dict,
                  pert_to_row: dict, X_test: np.ndarray):
    """
    Predict label and probabilities for each (pert, gene) pair in test_df.

    - Pairs whose pert has no embedding are dropped (excluded from output).
    - Pairs whose gene has no fitted model default to class 0 with uniform
      probabilities [1/3, 1/3, 1/3]; the gene is recorded in no_model_genes.

    Args:
        test_df:     DataFrame with columns [pert, gene, label]
        models:      dict mapping gene_name -> fitted LogisticRegression
        pert_to_row: dict mapping pert_name -> row index in X_test
        X_test:      (n_test_perts, m) embedding matrix for test perturbations

    Returns:
        y_pred:        int array of shape (n_valid,)
        y_proba:       float array of shape (n_valid, 3)
        valid_mask:    bool array of shape (len(test_df),); False only for missing-embedding perts
        no_model_genes: set of gene names that had no fitted model (defaulted to class 0)
    """
    n_no_emb       = 0
    no_model_genes = set()
    y_pred_list, y_proba_list, valid = [], [], []

    for _, row in test_df.iterrows():
        p, g = row['pert'], row['gene']
        if p not in pert_to_row:
            n_no_emb += 1
            valid.append(False)
            continue
        if g not in models:
            no_model_genes.add(g)
            y_pred_list.append(0)
            y_proba_list.append(np.array([1/3, 1/3, 1/3]))
            valid.append(True)
            continue
        x   = X_test[pert_to_row[p]].reshape(1, -1)
        clf = models[g]
        proba_full = np.zeros(3, dtype=np.float64)
        if isinstance(clf, tuple) and clf[0] == 'constant':
            pred = clf[1]
            proba_full[pred] = 1.0
        else:
            pred  = clf.predict(x)[0]
            proba = clf.predict_proba(x)[0]
            # ensure proba has 3 entries even if model only saw 2 classes
            for ci, c in enumerate(clf.classes_):
                proba_full[c] = proba[ci]
        y_pred_list.append(pred)
        y_proba_list.append(proba_full)
        valid.append(True)

    if n_no_emb > 0:
        warnings.warn(f"{n_no_emb} test pairs dropped: perturbation not in embedding dict")
    if no_model_genes:
        warnings.warn(
            f"{len(no_model_genes)} gene(s) had no fitted model and defaulted to class 0: "
            f"{sorted(no_model_genes)}"
        )

    valid_mask = np.array(valid, dtype=bool)
    y_pred  = np.array(y_pred_list, dtype=int)
    y_proba = np.array(y_proba_list, dtype=np.float64)
    return y_pred, y_proba, valid_mask, no_model_genes


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Per-gene logistic regression for 3-class perturbation-gene prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Modes:\n"
            "  Within-cell-line:  --de_doc_file <csv>\n"
            "  Cross-cell-line:   --train_de_doc_files <csv1,csv2,csv3> --test_de_doc_file <csv4>"
        ),
    )
    # Within-cell-line mode
    parser.add_argument('--de_doc_file', type=str, default=None,
                        help='(within-cell-line) DE_DOC CSV with split_calib column')
    # Cross-cell-line mode
    parser.add_argument('--train_de_doc_files', type=str, default=None,
                        help='(cross-cell-line) Comma-separated DE_DOC CSVs for training cell lines')
    parser.add_argument('--test_de_doc_file', type=str, default=None,
                        help='(cross-cell-line) DE_DOC CSV for the held-out test cell line')
    parser.add_argument('--embedding_file', type=str, required=True, help='Path to primary GenePT embedding pkl')
    parser.add_argument('--embedding_file2', type=str, default=None, help='Path to secondary GenePT embedding pkl (merged into primary)')
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name used for output file naming')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory for predictions and metrics')
    parser.add_argument('--Cs', type=str, default='0.1,1.0,10.0', help='Comma-separated C values to search over in CV (C=1/lambda)')
    parser.add_argument('--cv', type=int, default=5, help='Number of cross-validation folds for C selection')
    parser.add_argument('--max_iter', type=int, default=1000, help='Max solver iterations')
    parser.add_argument('--random_seed', type=int, default=305, help='Random seed')

    args = parser.parse_args()

    # Validate mode
    within_mode = args.de_doc_file is not None
    cross_mode  = args.train_de_doc_files is not None or args.test_de_doc_file is not None
    if within_mode and cross_mode:
        parser.error("Specify either --de_doc_file (within-cell-line) OR "
                     "--train_de_doc_files/--test_de_doc_file (cross-cell-line), not both.")
    if not within_mode and not cross_mode:
        parser.error("Must specify either --de_doc_file or "
                     "--train_de_doc_files + --test_de_doc_file.")
    if cross_mode and (args.train_de_doc_files is None or args.test_de_doc_file is None):
        parser.error("Cross-cell-line mode requires both --train_de_doc_files and --test_de_doc_file.")

    print("=" * 70)
    print("Per-Gene Logistic Regression: Perturbation-Gene Prediction")
    print("=" * 70)
    print(f"\nDataset:  {args.dataset_name}")
    print(f"Mode:     {'within-cell-line' if within_mode else 'cross-cell-line'}")
    print(f"Cs (CV):    {args.Cs}")
    print(f"CV folds:   {args.cv}")
    print(f"Max iter:   {args.max_iter}")
    print(f"Output dir: {args.output_dir}")

    error_dir = os.path.join(os.path.dirname(args.output_dir), 'error_analysis')
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(error_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Data loading                                                         #
    # ------------------------------------------------------------------ #
    if within_mode:
        df = pd.read_csv(args.de_doc_file)
        if 'split_calib' not in df.columns:
            raise ValueError("CSV must have a 'split_calib' column")
        train_df = df[df['split_calib'] == 'train'].reset_index(drop=True)
        calib_df = df[df['split_calib'] == 'calibration'].reset_index(drop=True)
        test_df  = df[df['split_calib'] == 'test'].reset_index(drop=True)
        print(f"\nTrain: {len(train_df)} pairs  |  Calib: {len(calib_df)}  |  Test: {len(test_df)}")
    else:
        # Cross-cell-line: use train+test (80%) from each source cell line
        train_paths = [p.strip() for p in args.train_de_doc_files.split(',')]
        print(f"\nLoading {len(train_paths)} training cell line file(s):")
        train_dfs = []
        for path in train_paths:
            cell_df = pd.read_csv(path)
            if 'split_calib' not in cell_df.columns:
                raise ValueError(f"CSV '{path}' must have a 'split_calib' column")
            cell_train = cell_df[cell_df['split_calib'].isin(['train', 'test'])].reset_index(drop=True)
            print(f"  {path}: {len(cell_train)} train+test pairs")
            train_dfs.append(cell_train)

        print("\nMerging training sets via majority vote ...")
        train_df = merge_train_sets(train_dfs, args.random_seed)

        test_cell_df = pd.read_csv(args.test_de_doc_file)
        if 'split_calib' not in test_cell_df.columns:
            raise ValueError(f"CSV '{args.test_de_doc_file}' must have a 'split_calib' column")
        calib_df = test_cell_df[test_cell_df['split_calib'] == 'calibration'].reset_index(drop=True)
        test_df  = test_cell_df[test_cell_df['split_calib'] == 'test'].reset_index(drop=True)
        print(f"\nTarget calib: {len(calib_df)} pairs  |  Target test: {len(test_df)} pairs")

        # Save unseen genes JSON
        train_genes_set = set(train_df['gene'].unique())
        eval_genes_set  = set(pd.concat([calib_df, test_df])['gene'].unique())
        unseen_genes    = sorted(eval_genes_set - train_genes_set)
        unseen_path = os.path.join(args.output_dir, f'{args.dataset_name}_transfer_unseen.json')
        with open(unseen_path, 'w') as f:
            json.dump({'unseen_genes': unseen_genes, 'n_unseen_genes': len(unseen_genes)}, f, indent=2)
        print(f"\nUnseen genes (eval only, default class 0): {len(unseen_genes)} — saved to {unseen_path}")

    eval_combined_df = pd.concat([calib_df, test_df], ignore_index=True)
    plot_data_summary(train_df, eval_combined_df, args.dataset_name, output_dir=error_dir)

    # ------------------------------------------------------------------ #
    # Build training matrices and fit models                               #
    # ------------------------------------------------------------------ #
    embedding_dict = load_embeddings(args.embedding_file, args.embedding_file2)

    train_perts      = sorted(train_df['pert'].unique())
    train_genes_list = sorted(train_df['gene'].unique())
    print(f"\nBuilding feature matrix X ({len(train_perts)} train perts x 3072) ...")
    X_train, _, valid_train_perts = build_feature_matrix(train_perts, embedding_dict)
    print(f"  X_train shape: {X_train.shape}")

    print(f"Building label matrix Y ({len(valid_train_perts)} perts x {len(train_genes_list)} genes) ...")
    Y_train = build_label_matrix(train_df, valid_train_perts, train_genes_list)
    n_observed = (~np.isnan(Y_train)).sum()
    print(f"  Y_train shape: {Y_train.shape}  ({n_observed} observed labels, "
          f"{np.isnan(Y_train).sum()} NaN)")

    Cs = [float(c) for c in args.Cs.split(',')]
    print(f"\nFitting per-gene models (Cs={Cs}, cv={args.cv}, max_iter={args.max_iter}) ...")
    models = fit_per_gene_models(X_train, Y_train, train_genes_list, Cs, args.cv,
                                 args.max_iter, args.random_seed)

    # ------------------------------------------------------------------ #
    # Build a single eval embedding matrix covering all calib+test perts  #
    # ------------------------------------------------------------------ #
    all_eval_perts = sorted(eval_combined_df['pert'].unique())
    print(f"\nBuilding eval embedding matrix ({len(all_eval_perts)} perts) ...")
    X_eval, _, valid_eval_perts = build_feature_matrix(all_eval_perts, embedding_dict)
    pert_to_row = {p: i for i, p in enumerate(valid_eval_perts)}

    # ------------------------------------------------------------------ #
    # Predict on calib, test, combined                                     #
    # ------------------------------------------------------------------ #
    def predict_split(df, split_label):
        print(f"\nPredicting on {split_label} set ({len(df)} pairs) ...")
        y_pred, y_proba, valid_mask, no_model = predict_pairs(df, models, pert_to_row, X_eval)
        df_clean = df[valid_mask].reset_index(drop=True)
        y_true   = df_clean['label'].values
        print(f"  Valid predictions: {valid_mask.sum()}/{len(df)}  "
              f"(defaulted genes: {len(no_model)})")
        return y_pred, y_proba, y_true, df_clean, no_model

    y_pred_calib, y_proba_calib, y_true_calib, calib_clean, no_model_calib = predict_split(calib_df, "calibration")
    y_pred_test,  y_proba_test,  y_true_test,  test_clean,  no_model_test  = predict_split(test_df,  "test")

    # Save defaulted genes JSON (union across both eval splits)
    no_model_genes = sorted(no_model_calib | no_model_test)
    no_model_path  = os.path.join(args.output_dir, f'{args.dataset_name}_logreg_no_model_genes.json')
    with open(no_model_path, 'w') as f:
        json.dump({'no_model_genes': no_model_genes, 'n_no_model_genes': len(no_model_genes)}, f, indent=2)
    print(f"\nGenes with no fitted model (defaulted to class 0): {len(no_model_genes)} — saved to {no_model_path}")

    y_pred_combined  = np.concatenate([y_pred_calib,  y_pred_test])
    y_proba_combined = np.concatenate([y_proba_calib, y_proba_test], axis=0)
    y_true_combined  = np.concatenate([y_true_calib,  y_true_test])
    combined_clean   = pd.concat([calib_clean, test_clean], ignore_index=True)

    # ------------------------------------------------------------------ #
    # Metrics and reporting                                                #
    # ------------------------------------------------------------------ #
    metrics_calib    = compute_3class_metrics(y_true_calib,    y_pred_calib,    y_proba_calib)
    metrics_test     = compute_3class_metrics(y_true_test,     y_pred_test,     y_proba_test)
    metrics_combined = compute_3class_metrics(y_true_combined, y_pred_combined, y_proba_combined)

    def print_metrics(metrics, header):
        print(f"\n{'='*70}")
        print(header)
        print(f"{'='*70}")
        print(f"  Accuracy:          {metrics['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"  Precision (macro): {metrics['precision_macro']:.4f}")
        print(f"  Recall (macro):    {metrics['recall_macro']:.4f}")
        print(f"  F1 (macro):        {metrics['f1_macro']:.4f}")
        if 'auroc_ovr_macro' in metrics:
            print(f"  AUROC OvR (macro): {metrics['auroc_ovr_macro']:.4f}")
        print(f"\n  Per-class metrics:")
        for class_idx in range(3):
            class_name = ['Non-DE', 'Up-regulated', 'Down-regulated'][class_idx]
            print(f"  {class_name} (class {class_idx}):")
            print(f"    Precision: {metrics[f'precision_class_{class_idx}']:.4f}")
            print(f"    Recall:    {metrics[f'recall_class_{class_idx}']:.4f}")
            print(f"    F1:        {metrics[f'f1_class_{class_idx}']:.4f}")
            if f'auroc_ovr_class_{class_idx}' in metrics:
                print(f"    AUROC OvR: {metrics[f'auroc_ovr_class_{class_idx}']:.4f}")

    print_metrics(metrics_calib,    "RESULTS — Calibration Set (20%)")
    print_metrics(metrics_test,     "RESULTS — Test Set (20%)")
    print_metrics(metrics_combined, "RESULTS — Combined Calib + Test Set (40%)")

    for split_label, y_true, y_pred in [
        ("Calibration", y_true_calib,    y_pred_calib),
        ("Test",        y_true_test,     y_pred_test),
        ("Combined",    y_true_combined, y_pred_combined),
    ]:
        print(f"\n{'='*70}")
        print(f"Detailed Classification Report — {split_label} Set")
        print(f"{'='*70}")
        print(classification_report(
            y_true, y_pred,
            target_names=['Non-DE', 'Up-regulated', 'Down-regulated'],
            zero_division=0
        ))

    # ------------------------------------------------------------------ #
    # Save 3 metrics CSVs and 3 predictions CSVs (with probabilities)     #
    # ------------------------------------------------------------------ #
    gene_to_best_C = {
        gene: float(np.mean(m.C_)) if hasattr(m, 'C_') else np.nan
        for gene, m in models.items()
    }

    for split_label, df_clean, metrics, y_pred, y_true, y_proba in [
        ("calib",    calib_clean,    metrics_calib,    y_pred_calib,    y_true_calib,    y_proba_calib),
        ("test",     test_clean,     metrics_test,     y_pred_test,     y_true_test,     y_proba_test),
        ("combined", combined_clean, metrics_combined, y_pred_combined, y_true_combined, y_proba_combined),
    ]:
        metrics_df = pd.DataFrame([metrics])
        metrics_df['dataset'] = args.dataset_name
        metrics_path = os.path.join(args.output_dir, f'{args.dataset_name}_logreg_metrics_{split_label}.csv')
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Metrics ({split_label}) saved to {metrics_path}")

        pred_df = df_clean.copy()
        pred_df['prediction']   = y_pred
        pred_df['correct']      = (y_pred == y_true)
        pred_df['prob_class_0'] = y_proba[:, 0]
        pred_df['prob_class_1'] = y_proba[:, 1]
        pred_df['prob_class_2'] = y_proba[:, 2]
        pred_df['best_C']       = pred_df['gene'].map(gene_to_best_C)
        pred_path = os.path.join(args.output_dir, f'{args.dataset_name}_logreg_predictions_{split_label}.csv')
        pred_df.to_csv(pred_path, index=False)
        print(f"Predictions ({split_label}) saved to {pred_path}")

    # ------------------------------------------------------------------ #
    # Error analysis on combined calib+test                                #
    # ------------------------------------------------------------------ #
    print(f"\n{'='*70}")
    print("ERROR ANALYSIS (Combined Calib + Test)")
    print(f"{'='*70}")
    combined_keys = list(zip(combined_clean['pert'], combined_clean['gene']))
    run_error_analysis_from_results(
        dataset=args.dataset_name,
        test_keys=combined_keys,
        predictions=y_pred_combined,
        true_labels=y_true_combined,
        adata=None,
        task_name="3class_logreg",
        class_names=['Non-DE', 'Up-regulated', 'Down-regulated'],
        output_dir=error_dir,
    )

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")