"""
perturb_metrics.py
==================
Metric functions for evaluating perturbation prediction models.
"""

import ot
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.stats import pearsonr
import sys
sys.path.insert(0, '/home/jsun140/scratchyiqunchen/jsun140/Project-Perturb/majority_vote')
from analysis_utility import compute_3class_metrics


## Expression-level Metrics

def mean_expr(ad, pert, pert_col='gene'):
    """
    For a specific perturbation, compute per-gene expression mean.
    """
    X = ad[ad.obs[pert_col] == pert].X          # subset to all cells of a specific perturbation
    if sp.issparse(X):
        X = X.toarray()
    return np.asarray(X.mean(axis=0)).squeeze() # (G, )


def wasserstein_distances(pred_dist: np.ndarray, real_dist: np.ndarray):
    """
    Based on documentation: https://pythonot.github.io/all.html#ot.wasserstein_1d
    
    Compute the W1 and W2 distance between the predicted expression distribution
    and the real perturbed expression distribution. W2 is more sensitive to shifts
    that are uneven across quantiles.

    NOTE: Wasserstein computation can be slow.
    """
    w1 = ot.wasserstein_1d(pred_dist, real_dist, p=1)
    w2 = np.sqrt(ot.wasserstein_1d(pred_dist, real_dist, p=2))
    return w1, w2


def expression_metrics(pred_ad, real_ad, pert_col='gene', skip_wasserstein=False):
    """
    Compute expression-level metrics. These includes:
        MAE, MSE       -> following formulation in STATE
        pearson_global —> one correlation over all (pert, gene) entries flattened
        pearson_delta  -> following formulation in STATE
        pearson_per_pert —> for each pert, correlate its gene profile (n_genes,);
                         average over perts
        pearson_per_gene —> for each gene, correlate its response across perts (n_perts,);
                         average over genes
        w1_mean, w2_mean —> mean W1, W2 per gene
    """
    # Compute average gene expression per perturbation for predicted and real objects
    # Mean matrices shape: (n_perts, n_genes) -> each row is the per-gene mean of a perturbation
    perts = [p for p in pred_ad.obs[pert_col].unique() if p != 'ctrl']
    pred_means = np.vstack([mean_expr(pred_ad, p, pert_col) for p in perts])
    real_means = np.vstack([mean_expr(real_ad, p, pert_col) for p in perts])
    n_perts, n_genes = pred_means.shape

    # MAE: mean over perts of per-pert L1 norm (sum of |diff| over genes)
    # MSE: mean over perts of per-pert squared L2 norm (sum of diff² over genes)
    diff = pred_means - real_means
    mae = float(np.mean(np.sum(np.abs(diff), axis=1)))
    mse = float(np.mean(np.sum(diff ** 2,    axis=1)))

    # Global pearson correlation: single correlation over all (pert, gene) entries
    # ravel() flattens 2D array into 1D
    pearson_global, _ = pearsonr(pred_means.ravel(), real_means.ravel())

    # Pearson Delta Correlation: correlate expression deltas Δ = |p̄_t - c̄|
    ctrl_real = mean_expr(real_ad, 'ctrl', pert_col)
    delta_pred = np.abs(pred_means - ctrl_real)   # (n_perts, n_genes)
    delta_real = np.abs(real_means - ctrl_real)   # (n_perts, n_genes)
    pearson_delta, _ = pearsonr(delta_pred.ravel(), delta_real.ravel())

    # Mean perturbation pearson correlation
    # for each perturbation, obtain pearson r across gene expression profiles then average
    pert_pearsons = []
    for i in range(n_perts):
        if np.std(pred_means[i]) == 0 or np.std(real_means[i]) == 0:
            continue
        corr, _ = pearsonr(pred_means[i], real_means[i])
        pert_pearsons.append(corr)
    pearson_per_pert = float(np.mean(pert_pearsons)) if pert_pearsons else float('nan')

    # Mean gene pearson correlation
    # for each gene, obtain pearson r across perturbations then average
    gene_pearsons = []
    for g in range(n_genes):
        if np.std(pred_means[:, g]) == 0 or np.std(real_means[:, g]) == 0:
            continue
        corr, _ = pearsonr(pred_means[:, g], real_means[:, g])
        gene_pearsons.append(corr)
    pearson_per_gene = float(np.mean(gene_pearsons)) if gene_pearsons else float('nan')

    # Per-gene W1 and W2: for each gene, compare its mean expression distribution
    # across perturbations between predicted and real
    if skip_wasserstein:
        w1_mean = float('nan')
        w2_mean = float('nan')
    else:
        w1s, w2s = zip(*[wasserstein_distances(pred_means[:, g], real_means[:, g])
                         for g in range(n_genes)])
        w1_mean = float(np.mean(w1s))   # take mean over gene wasserstein
        w2_mean = float(np.mean(w2s))   # take mean over gene wasserstein

    return {
        'mae':              mae,
        'mse':              mse,
        'pearson_global':   float(pearson_global),
        'pearson_delta':    float(pearson_delta),
        'pearson_per_pert': pearson_per_pert,
        'pearson_per_gene': pearson_per_gene,
        'w1_mean':          w1_mean,
        'w2_mean':          w2_mean,
    }


### DE Overlap Metrics (Pooling up/down directions together)

def _top_k_degs(de_df, pert, k=None):
    """
    Get top-k DE gene names for a perturbation: filter DE (label != 0),
    rank by |log2FC| descending, take top k (all if k is None or k > n_de).
    Returns a ranked list of gene names.
    """
    sub = de_df[(de_df['pert'] == pert) & (de_df['label'] != 0)].copy()
    sub = sub.reindex(sub['log2FC'].abs().sort_values(ascending=False).index)
    if k is not None:
        sub = sub.head(k)
    return sub['gene'].tolist()


def de_overlap(de_pred_df, de_real_df, k=None):
    """
    Calculate DE overlap accuracy. When k=None, k is set to N where N is the
    number of true DEGs per-perturbation. For each perturbation that have at
    least 1 true DEG, an overlap score is calculated, and a final metric averaged 
    over perturbations is reported.

    Overlap_{t,k} = |G_t_true^(k) ∩ G_t_pred^(k)| / k
    """
    pred_perts = set(de_pred_df['pert'].unique())
    # fixed a behavioral bug here: previously if we loop over every perturbation in
    # the true_de_df, a perturbation never predicted in the cross-cell-line tasks will
    # simply get overlap = 0, artifically deflate the de_overlap metric.
    perts = [p for p in de_real_df['pert'].unique() if p in pred_perts]
    n_excluded = de_real_df['pert'].nunique() - len(perts)
    if n_excluded > 0:
        print(f'  de_overlap: {n_excluded} perts in real not in pred, excluded from overlap computation')
    overlaps = []
    for pert in perts:
        # get all real DEGs
        true_degs = _top_k_degs(de_real_df, pert, k=None)
        if len(true_degs) == 0:
            continue
        effective_k = len(true_degs) if k is None else k
        true_top = true_degs[:effective_k]
        pred_top = _top_k_degs(de_pred_df, pert, k=effective_k)
        overlaps.append(len(set(true_top) & set(pred_top)) / effective_k)
    return float(np.mean(overlaps)) if overlaps else float('nan')


## Combined Metric Computation

def compute_all_metrics_from_dfs(pred_de_df, real_de_df, pred_ad, real_ad, skip_wasserstein=True):
    """
    Core metric computation.

    Args:
        pred_de_df : DataFrame with predicted DE (columns: pert, gene, label, log2FC)
        real_de_df : DataFrame with real DE labels (columns: pert, gene, label, log2FC)
        pred_ad    : AnnData of predicted expression
        real_ad    : AnnData of real expression

    Returns:
        dict of metric_name to value
    """
    # 3-class DE classification metrics
    merged = real_de_df.merge(pred_de_df[['pert', 'gene', 'label']], on=['pert', 'gene'],
                              how='inner', suffixes=('_real', '_pred'))
    cls_metrics = compute_3class_metrics(merged['label_real'].values, merged['label_pred'].values)

    # DE overlap metrics
    overlap_N  = de_overlap(pred_de_df, real_de_df, k=None)
    overlap_50 = de_overlap(pred_de_df, real_de_df, k=50)

    # Expression-level metrics (might need to set 'pert_col' argument)
    expr_m = expression_metrics(pred_ad, real_ad, skip_wasserstein=skip_wasserstein)

    return {
        **cls_metrics,
        'de_overlap_N':    overlap_N,
        'de_overlap_50':   overlap_50,
        **expr_m,
    }
