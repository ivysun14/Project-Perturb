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

Input CSV format (from DE_DOC_stats.py):
  pert, gene, label, split
"""

import os
import sys
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import classification_report

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baseline.analysis_utility import compute_3class_metrics

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
    n_skipped  = 0
    n_constant = 0
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

    n_fitted = len(models) - n_constant
    print(f"  Fitted models for {n_fitted}/{len(gene_names)} genes  |  "
          f"constant predictor: {n_constant}  |  skipped (<2 obs): {n_skipped}")
    return models


def predict_pairs(test_df: pd.DataFrame, models: dict,
                  pert_to_row: dict, X_test: np.ndarray):
    """
    Predict label and probabilities for each (pert, gene) pair in test_df.
    Pairs whose gene has no fitted model, or whose pert has no embedding,
    are excluded and flagged.

    Args:
        test_df:     DataFrame with columns [pert, gene, label]
        models:      dict mapping gene_name -> fitted LogisticRegression
        pert_to_row: dict mapping pert_name -> row index in X_test
        X_test:      (n_test_perts, m) embedding matrix for test perturbations

    Returns:
        y_pred:     int array of shape (n_valid,)
        y_proba:    float array of shape (n_valid, 3)
        valid_mask: bool array of shape (len(test_df),)
    """
    n_no_model = 0
    n_no_emb   = 0
    y_pred_list, y_proba_list, valid = [], [], []

    for _, row in test_df.iterrows():
        p, g = row['pert'], row['gene']
        if p not in pert_to_row:
            n_no_emb += 1
            valid.append(False)
            continue
        if g not in models:
            n_no_model += 1
            valid.append(False)
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

    # These are flagged but none of which should happen (at least for within cell line)
    if n_no_emb > 0:
        warnings.warn(f"{n_no_emb} test pairs dropped: perturbation not in embedding dict")
    if n_no_model > 0:
        warnings.warn(f"{n_no_model} test pairs dropped: gene had no fitted model (single class or too few training obs)")

    valid_mask = np.array(valid, dtype=bool)
    y_pred  = np.array(y_pred_list, dtype=int)
    y_proba = np.array(y_proba_list, dtype=np.float64)
    return y_pred, y_proba, valid_mask


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Per-gene logistic regression for 3-class perturbation-gene prediction')
    parser.add_argument('--de_doc_file', type=str, required=True, help='Path to DE_DOC CSV file with columns: pert, gene, label, split')
    parser.add_argument('--embedding_file', type=str, required=True, help='Path to primary GenePT embedding pkl')
    parser.add_argument('--embedding_file2', type=str, default=None, help='Path to secondary GenePT embedding pkl (merged into primary)')
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name used for output file naming')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory for predictions and metrics')
    parser.add_argument('--Cs', type=str, default='0.1,1.0,10.0', help='Comma-separated C values to search over in CV (C=1/lambda)')
    parser.add_argument('--cv', type=int, default=5, help='Number of cross-validation folds for C selection')
    parser.add_argument('--max_iter', type=int, default=1000, help='Max solver iterations')
    parser.add_argument('--random_seed', type=int, default=305, help='Random seed')

    args = parser.parse_args()

    print("=" * 70)
    print("Per-Gene Logistic Regression: Perturbation-Gene Prediction")
    print("=" * 70)
    print(f"\nDataset:  {args.dataset_name}")
    print(f"Cs (CV):    {args.Cs}")
    print(f"CV folds:   {args.cv}")
    print(f"Max iter:   {args.max_iter}")
    print(f"Output dir: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.de_doc_file)
    if 'split' not in df.columns:
        raise ValueError("CSV must have a 'split' column ('train'/'test')")
    
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    test_df  = df[df['split'] == 'test'].reset_index(drop=True)
    embedding_dict = load_embeddings(args.embedding_file, args.embedding_file2)

    # Build X_train: (n_valid_train_perts, 3072)
    train_perts = sorted(train_df['pert'].unique())   # fixed order → rows of X/Y
    train_genes = sorted(train_df['gene'].unique())   # fixed order → columns of Y
    print(f"\nBuilding feature matrix X ({len(train_perts)} train perts x 3072) ...")
    X_train, train_valid_mask, valid_train_perts = build_feature_matrix(train_perts, embedding_dict)
    print(f"  X_train shape: {X_train.shape}  ({train_valid_mask.sum()} perts with embeddings)")

    # Build Y_train: (n_valid_train_perts, n_genes)
    print(f"Building label matrix Y ({len(valid_train_perts)} perts x {len(train_genes)} genes) ...")
    Y_train = build_label_matrix(train_df, valid_train_perts, train_genes)
    n_observed = (~np.isnan(Y_train)).sum()
    print(f"  Y_train shape: {Y_train.shape}  ({n_observed} observed labels, "
          f"{np.isnan(Y_train).sum()} NaN)")

    # Fit one logistic regression per gene
    Cs = [float(c) for c in args.Cs.split(',')]
    print(f"\nFitting per-gene models (Cs={Cs}, cv={args.cv}, max_iter={args.max_iter}) ...")
    models = fit_per_gene_models(X_train, Y_train, train_genes, Cs, args.cv, args.max_iter, args.random_seed)

    # Build X_test: (n_test_perts, 3072)
    test_perts = sorted(test_df['pert'].unique())
    print(f"\nBuilding test embedding matrix ({len(test_perts)} test perts) ...")
    X_test, test_valid_mask, valid_test_perts = build_feature_matrix(test_perts, embedding_dict)
    pert_to_row = {p: i for i, p in enumerate(valid_test_perts)}

    print(f"\nPredicting {len(test_df)} test pairs ...")
    y_pred, y_proba, pred_valid_mask = predict_pairs(test_df, models, pert_to_row, X_test)
    test_df_clean = test_df[pred_valid_mask].reset_index(drop=True)
    y_true = test_df_clean['label'].values
    print(f"  Valid predictions: {pred_valid_mask.sum()}/{len(test_df)}")

    metrics = compute_3class_metrics(y_true, y_pred, y_proba)

    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"\nOverall metrics:")
    print(f"  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro):    {metrics['recall_macro']:.4f}")
    print(f"  F1 (macro):        {metrics['f1_macro']:.4f}")
    if 'auroc_ovr_macro' in metrics:
        print(f"  AUROC OvR (macro): {metrics['auroc_ovr_macro']:.4f}")

    print(f"\nPer-class metrics:")
    for class_idx in range(3):
        class_name = ['Non-DE', 'Up-regulated', 'Down-regulated'][class_idx]
        print(f"  {class_name} (class {class_idx}):")
        print(f"    Precision: {metrics[f'precision_class_{class_idx}']:.4f}")
        print(f"    Recall:    {metrics[f'recall_class_{class_idx}']:.4f}")
        print(f"    F1:        {metrics[f'f1_class_{class_idx}']:.4f}")
        if f'auroc_ovr_class_{class_idx}' in metrics:
            print(f"    AUROC OvR: {metrics[f'auroc_ovr_class_{class_idx}']:.4f}")

    print(f"\n{'='*70}")
    print("Detailed Classification Report")
    print(f"{'='*70}")
    print(classification_report(
        y_true, y_pred,
        target_names=['Non-DE', 'Up-regulated', 'Down-regulated'],
        zero_division=0
    ))

    # Map each gene to the C chosen by CV (NaN for constant/skipped genes)
    # If CV is not performed, this will just assign nan to all genes
    gene_to_best_C = {
        gene: float(np.mean(m.C_)) if hasattr(m, 'C_') else np.nan
        for gene, m in models.items()
    }

    predictions_df = test_df_clean.copy()
    predictions_df['prediction']   = y_pred
    predictions_df['correct']      = (y_pred == y_true)
    predictions_df['prob_class_0'] = y_proba[:, 0]
    predictions_df['prob_class_1'] = y_proba[:, 1]
    predictions_df['prob_class_2'] = y_proba[:, 2]
    predictions_df['best_C']       = predictions_df['gene'].map(gene_to_best_C)
    predictions_path = os.path.join(args.output_dir, f'{args.dataset_name}_logreg_predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)
    print(f"\nPredictions saved to {predictions_path}")

    metrics_df = pd.DataFrame([metrics])
    metrics_df['dataset'] = args.dataset_name
    metrics_path = os.path.join(args.output_dir, f'{args.dataset_name}_logreg_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
