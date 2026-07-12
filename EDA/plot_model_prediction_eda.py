"""
plot_model_prediction_eda.py
============================
EDA for prediction outputs from Scouter (within-cell-line), TranScouter, and STATE
(both cross-cell-line). H5ad paths are resolved automatically from config.py.

Analyses produced:
  1. Per-gene mean scatter (pooled): real vs predicted mean across all perturbed cells,
     with Pearson r and Spearman ρ.
  2. Per-gene Pearson r histogram: for each gene, Pearson r is computed by correlating
     its real vs predicted per-perturbation mean expression vector across all
     perturbations, then the distribution of these per-gene r values is plotted.
     The mean of this distribution matches the pearson_per_gene metric in
     perturb_metrics.py.
  3. Control vs perturbed per-gene mean scatters: ctrl mean (x) vs real perturbed
     mean (y), and ctrl mean (x) vs predicted perturbed mean (y).
  4. Overlaid distribution comparison (real vs pred): mean, variance, zero proportion.
  5. Per-gene summary CSV saved to results/.

Data loading per model:
  Scouter     — within-cell-line; calib + filtered test h5ads concatenated via
                load_scouter_pred(); obs['gene'] == 'ctrl' marks control cells.
  TranScouter — cross-cell-line; single testcalib h5ad from cross_cond/prediction/;
                var_names are already gene names; obs['gene'] == 'ctrl' for controls.
                Subsetted to dropped_test_perts.json exclusion list and common gene
                universe before analysis.
  STATE       — cross-cell-line; eval_task1_merged.ckpt h5ads; var_names are integer
                indices positionally aligned to common_gene_universe.csv — replaced
                with universe gene names after an exact count assertion.
                obs['gene'] == 'non-targeting' for control cells.
                Subsetted to dropped_test_perts.json exclusion list before analysis.

Usage:
    python plot_model_prediction_eda.py --model Scouter     --dataset k562
    python plot_model_prediction_eda.py --model TranScouter --dataset rpe1
    python plot_model_prediction_eda.py --model STATE       --dataset hepg2
    python plot_model_prediction_eda.py --model Scouter     --dataset k562 --calibration sparsity
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(__file__))
from config import DATASETS, RESULTS_DIR, FIGURES_DIR, PRED_PATHS, CALIB_PATHS, GENE_UNIVERSE_CSV, DROPPED_PERTS_JSON, load_scouter_pred

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_h5ad(path: str) -> ad.AnnData:
    return ad.read_h5ad(path)


def dense(X) -> np.ndarray:
    return X.toarray() if sparse.issparse(X) else np.asarray(X)


def gene_stats(X: np.ndarray) -> dict:
    return {
        'mean':      X.mean(axis=0),
        'var':       X.var(axis=0),
        'zero_prop': (X == 0).mean(axis=0),
    }


def resolve_gene_names(adata: ad.AnnData) -> np.ndarray:
    """
    Return gene name array.
    - Scouter / TranScouter: var has a 'gene_name' column; use it directly.
    - STATE: var_names are already set to universe gene names by subset_cross_cell_line;
      fall through to return var_names.
    """
    if 'gene_name' in adata.var.columns:
        return np.array(adata.var['gene_name'].tolist())
    return np.array(adata.var_names.tolist())


def subset_cross_cell_line(
    adata_real: ad.AnnData,
    adata_pred: ad.AnnData,
    ds: str,
    model: str,
    universe_genes: np.ndarray,
) -> tuple[ad.AnnData, ad.AnnData]:
    """
    For cross-cell-line models (TranScouter, STATE):
      1. Drop perturbations listed in dropped_test_perts.json (control cells are kept).
      2. Subset genes to the common gene universe.

    TranScouter: var['gene_name'] is present; filter by name.
    STATE: var_names are integer indices aligned positionally to the universe; take first
           len(universe_genes) columns (gene count should already match, but guard anyway).
    """
    with open(DROPPED_PERTS_JSON) as f:
        dropped = set(json.load(f)[ds])

    # Drop flagged perturbations from both objects (keep control cells regardless).
    keep_cells = adata_real.obs['gene'].apply(lambda g: g not in dropped).values
    adata_real = adata_real[keep_cells].copy()
    adata_pred = adata_pred[keep_cells].copy()
    print(f"  After dropping {len(dropped)} perts: {adata_real.n_obs:,} cells remain")

    # Subset genes to common universe.
    if model == 'TranScouter':
        # var_names are already human-readable gene names; reorder to universe order.
        name_to_idx = {name: i for i, name in enumerate(adata_real.var_names)}
        col_idx = [name_to_idx[g] for g in universe_genes if g in name_to_idx]
        adata_real = adata_real[:, col_idx].copy()
        adata_pred = adata_pred[:, col_idx].copy()
    else:
        # STATE: integer var_names are positionally aligned to the universe.
        n = len(universe_genes)
        assert adata_real.n_vars == n, (
            f"STATE {ds}: expected {n} genes (universe) but h5ad has {adata_real.n_vars}"
        )
        adata_real = adata_real.copy()
        adata_pred = adata_pred.copy()
        adata_real.var_names = universe_genes
        adata_pred.var_names = universe_genes

    print(f"  After gene subsetting: {adata_real.n_vars:,} genes")
    return adata_real, adata_pred


def compute_pert_means_matrix(X_pert: np.ndarray, pert_labels: np.ndarray) -> np.ndarray:
    """
    For each unique perturbation, compute per-gene mean across its cells.
    Returns shape (n_perts, n_genes).
    """
    perts = np.unique(pert_labels)
    return np.stack([X_pert[pert_labels == p].mean(axis=0) for p in perts])


def compute_per_gene_pearsonr(real_means: np.ndarray, pred_means: np.ndarray) -> np.ndarray:
    """
    For each gene, compute Pearson r between its real and predicted per-perturbation
    mean expression vector (length = n_perts). Genes with zero variance in either
    vector are skipped (assigned np.nan), matching the pearson_per_gene metric in
    perturb_metrics.py.
    Returns shape (n_genes,) with np.nan for skipped genes.
    """
    n_genes = real_means.shape[1]
    rs = np.full(n_genes, np.nan)
    for g in range(n_genes):
        if np.std(real_means[:, g]) == 0 or np.std(pred_means[:, g]) == 0:
            continue
        rs[g], _ = pearsonr(real_means[:, g], pred_means[:, g])
    return rs


def plot_real_vs_pred_mean(real_mean, pred_mean, dataset, model, tag, out_dir):
    """
    Scatter of per-gene mean: real (x) vs predicted (y), coloured by density proxy.
    """
    r, _  = pearsonr(real_mean, pred_mean)
    rho,_ = spearmanr(real_mean, pred_mean)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(real_mean, pred_mean, s=3, alpha=0.3, color='steelblue', rasterized=True)
    lim = max(real_mean.max(), pred_mean.max()) * 1.02
    ax.plot([0, lim], [0, lim], 'k--', lw=1, alpha=0.6, label='y = x')
    ax.set_xlabel('Per-gene mean — real expression', fontsize=12, fontweight='bold')
    ax.set_ylabel('Per-gene mean — predicted expression', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Real vs Predicted Per-gene Mean\n{model} — {dataset} {tag}\n'
        f'Pearson r = {r:.3f}   Spearman ρ = {rho:.3f}',
        fontsize=12, fontweight='bold',
    )
    ax.legend(fontsize=9, frameon=False)
    plt.tight_layout()
    out = os.path.join(out_dir, f'real_vs_pred_mean.pdf')
    plt.savefig(out, bbox_inches='tight', dpi=400)
    print(f"  Saved: {out}")
    plt.close()
    return r, rho


def plot_per_gene_pearsonr_hist(per_gene_r: np.ndarray, dataset, model, tag, out_dir):
    """
    Histogram of per-gene Pearson r values, where each gene's r is computed by
    correlating its real vs predicted per-perturbation mean expression vector across
    all perturbations. This directly visualises the pearson_per_gene metric from
    perturb_metrics.py.
    """
    valid = per_gene_r[~np.isnan(per_gene_r)]
    mean_r = float(np.mean(valid))
    median_r = float(np.median(valid))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(valid, bins=80, color='steelblue', alpha=0.7, histtype='stepfilled', linewidth=0.5)
    ax.axvline(mean_r,   color='firebrick', lw=1.5, linestyle='--', label=f'Mean = {mean_r:.3f}')
    ax.axvline(median_r, color='darkorange', lw=1.5, linestyle=':',  label=f'Median = {median_r:.3f}')
    ax.axvline(0, color='k', lw=1, alpha=0.4)
    ax.set_xlabel('Per-gene Pearson r (across perturbations)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of genes', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Per-gene Pearson r Distribution\n{model} — {dataset} {tag}\n'
        f'n = {len(valid):,} genes  |  mean r = {mean_r:.3f}',
        fontsize=12, fontweight='bold',
    )
    ax.legend(fontsize=9, frameon=False)
    plt.tight_layout()
    out = os.path.join(out_dir, 'per_gene_pearsonr_hist.pdf')
    plt.savefig(out, bbox_inches='tight', dpi=400)
    print(f"  Saved: {out}")
    plt.close()
    return mean_r


def plot_ctrl_vs_pert_mean(ctrl_mean, pert_mean, label_x, label_y, dataset, model, tag, out_dir):
    """
    Scatter of per-gene control mean (x) vs perturbed mean (y).
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(ctrl_mean, pert_mean, s=3, alpha=0.3, color='steelblue', rasterized=True)
    lim = max(ctrl_mean.max(), pert_mean.max()) * 1.02
    ax.plot([0, lim], [0, lim], 'k--', lw=1, alpha=0.6, label='y = x')
    ax.set_xlabel(label_x, fontsize=12, fontweight='bold')
    ax.set_ylabel(label_y, fontsize=12, fontweight='bold')
    ax.set_title(f'Control vs Perturbed Per-gene Mean\n{model} — {dataset} {tag}',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, frameon=False)
    plt.tight_layout()
    safe_y = label_y.replace(' ', '_').replace('/', '')[:20]
    out = os.path.join(out_dir, f'ctrl_vs_{safe_y}.pdf')
    plt.savefig(out, bbox_inches='tight', dpi=400)
    print(f"  Saved: {out}")
    plt.close()


def plot_distribution_comparison(stats_a, stats_b, label_a, label_b,
                                 color_a, dataset, model, tag, out_dir):
    """
    Overlaid histograms of mean, variance, zero proportion for two conditions.
    """
    metrics = [
        ('mean',      'Mean expression'),
        ('var',       'Variance'),
        ('zero_prop', 'Zero proportion'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    kw = dict(density=True, alpha=0.6, histtype='stepfilled', linewidth=0.5)

    for ax, (key, xlabel) in zip(axes, metrics):
        bins = 60 if key == 'zero_prop' else 100
        ax.hist(stats_a[key], bins=bins, color=color_a, label=label_a, **kw)
        ax.hist(stats_b[key], bins=bins, color='grey',  label=label_b, **kw)
        ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
        ax.set_ylabel('Density', fontsize=14, fontweight='bold')
        ax.set_title(f'{model} — {dataset} {tag}\n{xlabel}', fontsize=14, fontweight='bold')
        ax.tick_params(axis='both', labelsize=13)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight('bold')
        ax.legend(fontsize=12, frameon=False)

    plt.tight_layout()
    out = os.path.join(out_dir, f'distributions.pdf')
    plt.savefig(out, bbox_inches='tight', dpi=400)
    print(f"  Saved: {out}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['Scouter', 'TranScouter', 'STATE'],
                        help='Model whose predictions to visualise')
    parser.add_argument('--dataset', required=True, choices=list(DATASETS.keys()))
    parser.add_argument('--calibration', default='none', choices=['none', 'sparsity'],
                        help='Calibration applied to predictions (default: none)')
    parser.add_argument('--tag', default=None,
                        help='Tag override for plot titles (default: within for Scouter, cross for others)')
    args = parser.parse_args()

    model       = args.model
    ds          = args.dataset
    calibration = args.calibration
    tag         = args.tag or ('within' if model == 'Scouter' else 'cross')

    # All outputs go into a dedicated subdirectory named {model}_{dataset}_{calibration}
    run_name   = f'{model}_{ds}_{calibration}'
    fig_dir    = os.path.join(FIGURES_DIR, run_name)
    result_dir = os.path.join(RESULTS_DIR, run_name)
    os.makedirs(fig_dir,    exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    # Per-model defaults for control-cell detection
    # Scouter / TranScouter: obs['gene'] == 'ctrl'
    # STATE: obs['gene'] == 'non-targeting'
    target_col  = 'gene'
    control_key = 'ctrl' if model in ('Scouter', 'TranScouter') else 'non-targeting'

    print(f"\n=== {model} — {ds} [{tag}] | calibration={calibration} ===")

    universe_genes = pd.read_csv(GENE_UNIVERSE_CSV)['gene_name'].values

    # Load real h5ad (same regardless of calibration; calibration only changes the pred)
    if model == 'Scouter':
        print("  Loading Scouter real (calib + filtered test concat)...")
        adata_real = load_scouter_pred(ds, 'real')
    else:
        paths = PRED_PATHS[model][ds]
        print(f"  Loading {model} real h5ad...")
        adata_real = ad.read_h5ad(paths['real'])

    # Load pred h5ad: raw predictions or sparsity-calibrated
    if calibration == 'none':
        if model == 'Scouter':
            print("  Loading Scouter pred (calib + filtered test concat)...")
            adata_pred = load_scouter_pred(ds, 'pred')
        else:
            print(f"  Loading {model} pred h5ad...")
            adata_pred = ad.read_h5ad(PRED_PATHS[model][ds]['pred'])
    else:
        print(f"  Loading {model} sparsity-calibrated pred h5ad...")
        adata_pred = ad.read_h5ad(CALIB_PATHS[model][ds])

    print(f"    real: {adata_real.n_obs:,} cells × {adata_real.n_vars:,} genes")
    print(f"    pred: {adata_pred.n_obs:,} cells × {adata_pred.n_vars:,} genes")

    # For cross-cell-line models, drop flagged perts and subset to common gene universe.
    if model in ('TranScouter', 'STATE'):
        print("  Subsetting to common perturbations and gene universe...")
        adata_real, adata_pred = subset_cross_cell_line(
            adata_real, adata_pred, ds, model, universe_genes,
        )

    gene_names = resolve_gene_names(adata_real)

    # ── separate control / perturbed cells ──
    ctrl_mask = (adata_real.obs[target_col] == control_key).values
    assert ctrl_mask.any(), (
        f"No control cells found for {model} {ds} "
        f"(obs[{target_col!r}] == {control_key!r} matched 0 cells)"
    )
    pert_mask = ~ctrl_mask
    print(f"  Control cells (gene=={control_key!r}): {ctrl_mask.sum():,}  |  Perturbed cells: {pert_mask.sum():,}")

    pert_labels = adata_real.obs[target_col].values[pert_mask]
    X_real = dense(adata_real.X)
    X_pred = dense(adata_pred.X)
    del adata_real, adata_pred

    X_ctrl      = X_real[ctrl_mask]
    X_real_pert = X_real[pert_mask]
    X_pred_pert = X_pred[pert_mask]
    ctrl_stats      = gene_stats(X_ctrl)
    real_pert_stats = gene_stats(X_real_pert)
    pred_pert_stats = gene_stats(X_pred_pert)
    real_means_matrix = compute_pert_means_matrix(X_real_pert, pert_labels)
    pred_means_matrix = compute_pert_means_matrix(X_pred_pert, pert_labels)
    real_mom = real_means_matrix.mean(axis=0)
    pred_mom = pred_means_matrix.mean(axis=0)
    per_gene_r = compute_per_gene_pearsonr(real_means_matrix, pred_means_matrix)

    # ── 1. real vs pred mean correlation (pooled) ──
    print("  Plotting real vs predicted mean (pooled)...")
    r, rho = plot_real_vs_pred_mean(
        real_pert_stats['mean'], pred_pert_stats['mean'],
        ds, model, tag, fig_dir,
    )
    print(f"    Pearson r = {r:.4f}   Spearman ρ = {rho:.4f}")

    # ── 1b. per-gene Pearson r histogram (matches pearson_per_gene metric) ──
    print("  Plotting per-gene Pearson r distribution...")
    mean_per_gene_r = plot_per_gene_pearsonr_hist(per_gene_r, ds, model, tag, fig_dir)
    print(f"    Mean per-gene Pearson r = {mean_per_gene_r:.4f}")

    # ── 2. ctrl vs real/pred pert mean ──
    print("  Plotting ctrl vs real perturbed mean...")
    plot_ctrl_vs_pert_mean(
        ctrl_stats['mean'], real_pert_stats['mean'],
        'Per-gene mean — control cells', 'Per-gene mean — real perturbed cells',
        ds, model, tag, fig_dir,
    )
    print("  Plotting ctrl vs predicted mean...")
    plot_ctrl_vs_pert_mean(
        ctrl_stats['mean'], pred_pert_stats['mean'],
        'Per-gene mean — control cells', 'Per-gene mean — predicted cells',
        ds, model, tag, fig_dir,
    )

    # ── 3. distribution comparison: real vs pred ──
    print("  Plotting distribution comparison...")
    plot_distribution_comparison(
        real_pert_stats, pred_pert_stats,
        label_a='Real', label_b='Predicted',
        color_a='steelblue',
        dataset=ds, model=model, tag=tag, out_dir=fig_dir,
    )

    # ── 4. save per-gene summary ──
    out_csv = os.path.join(result_dir, 'gene_summary.csv')
    summary = pd.DataFrame({
        'gene':              gene_names,
        'real_mean':         real_pert_stats['mean'],
        'pred_mean':         pred_pert_stats['mean'],
        'real_mean_of_means': real_mom,
        'pred_mean_of_means': pred_mom,
        'per_gene_pearsonr': per_gene_r,
        'real_var':          real_pert_stats['var'],
        'pred_var':          pred_pert_stats['var'],
        'real_zero_prop':    real_pert_stats['zero_prop'],
        'pred_zero_prop':    pred_pert_stats['zero_prop'],
        'ctrl_mean':         ctrl_stats['mean'],
        'ctrl_var':          ctrl_stats['var'],
        'ctrl_zero_prop':    ctrl_stats['zero_prop'],
    })
    summary.to_csv(out_csv, index=False)
    print(f"  Saved summary: {out_csv}")
