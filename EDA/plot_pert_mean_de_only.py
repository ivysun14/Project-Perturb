"""
plot_pert_mean_de_only.py
=========================
For each gene, compute mean expression in perturbed cells restricted to
perturbations where that gene is differentially expressed. Genes never DE
in any perturbation are excluded.

This contrasts with gene_ctrl_vs_pert_mean.pdf, where the perturbed mean
pools all perturbations regardless of DE status. Here, only the perturbations
that actually move a gene show up — so the y-axis reflects genuine signal
rather than a diluted average dominated by null perturbations.

Output:
    results/pert_mean_de_only_{dataset}.csv  — per-gene ctrl_mean and pert_mean_de
    figures/gene_ctrl_vs_pert_mean_de_only.pdf
"""

import os
import sys
from collections import defaultdict
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
from scipy import sparse

sys.path.insert(0, os.path.dirname(__file__))
from config import DATASETS, RESULTS_DIR, FIGURES_DIR, DATASET_COLORS

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_adata(cfg: dict) -> ad.AnnData:
    adata = ad.read_h5ad(cfg['h5ad'])
    sc.pp.normalize_total(adata, target_sum=1e4, key_added='size_factors')
    sc.pp.log1p(adata)
    return adata


def compute_pert_mean_de_only(cfg: dict) -> pd.DataFrame:
    """
    Returns a DataFrame (gene, ctrl_mean, pert_mean_de) for genes that are
    DE in at least one perturbation.
    """
    name = cfg['display_name']

    # --- load expression ---
    print(f"  Loading {name} h5ad...")
    adata = load_adata(cfg)
    gene_names = np.array(adata.var[cfg['gene_name_col']].tolist())
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    pert_labels = adata.obs[cfg['target_col']].values
    ctrl_mask   = (pert_labels == cfg['control_key'])

    X = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
    del adata

    # Control mean (per gene) — recompute here to keep this script self-contained
    ctrl_mean = X[ctrl_mask].mean(axis=0)

    # --- build pert → cell-index mapping (perturbed cells only) ---
    pert_to_cells = defaultdict(list)
    for i, p in enumerate(pert_labels):
        if p != cfg['control_key']:
            pert_to_cells[p].append(i)
    pert_to_cells = {p: np.array(idxs) for p, idxs in pert_to_cells.items()}

    # --- build gene → DE perturbation set from DE CSV ---
    print(f"  Loading DE CSV for {name}...")
    de_df = pd.read_csv(cfg['de_csv'])
    # Keep only DE pairs (label != 0 means up or down)
    de_pairs = de_df[de_df['label'] != 0][['gene', 'pert']]
    gene_to_de_perts = defaultdict(set)
    for gene, pert in zip(de_pairs['gene'], de_pairs['pert']):
        gene_to_de_perts[gene].add(pert)

    print(f"  Computing DE-restricted perturbed mean for {len(gene_to_de_perts):,} DE genes...")
    records = []
    for gene, de_perts in gene_to_de_perts.items():
        if gene not in gene_to_idx:
            continue
        j = gene_to_idx[gene]

        # Gather cells from perturbations where this gene is DE
        cell_idx = np.concatenate(
            [pert_to_cells[p] for p in de_perts if p in pert_to_cells]
        )
        if len(cell_idx) == 0:
            continue

        records.append({
            'gene':         gene,
            'ctrl_mean':    ctrl_mean[j],
            'pert_mean_de': X[cell_idx, j].mean(),
            'n_de_perts':   len(de_perts),
        })

    return pd.DataFrame(records)


# --- compute and save per dataset ---
results = {}
for ds, cfg in DATASETS.items():
    print(f"\n=== {ds} ===")
    df = compute_pert_mean_de_only(cfg)
    out = os.path.join(RESULTS_DIR, f'pert_mean_de_only_{ds}.csv')
    df.to_csv(out, index=False)
    print(f"  Saved {len(df):,} genes → {out}")
    results[ds] = df

# --- plot ---
n = len(results)
fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
if n == 1:
    axes = [axes]

for ax, (ds, df) in zip(axes, results.items()):
    name  = DATASETS[ds]['display_name']
    color = DATASET_COLORS[ds]

    sc_ = ax.scatter(
        df['ctrl_mean'], df['pert_mean_de'],
        c=np.log1p(df['n_de_perts']),
        cmap='YlOrRd', s=3, alpha=0.4, rasterized=True,
    )
    lim = max(df['ctrl_mean'].max(), df['pert_mean_de'].max()) * 1.02
    ax.plot([0, lim], [0, lim], 'k--', lw=1, alpha=0.6, label='y = x')

    plt.colorbar(sc_, ax=ax, label='log(1 + n DE perturbations)', pad=0.02)

    ax.set_xlabel('Per-gene mean in control cells', fontsize=12, fontweight='bold')
    ax.set_ylabel('Per-gene mean in DE-perturbation cells only', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Control vs DE-restricted Perturbed Mean\n{name}  '
        f'(n={len(df):,} DE genes)',
        fontsize=13, fontweight='bold',
    )
    ax.legend(fontsize=9, frameon=False)

plt.tight_layout()
out = os.path.join(FIGURES_DIR, 'gene_ctrl_vs_pert_mean_de_only.pdf')
plt.savefig(out, bbox_inches='tight', dpi=300)
print(f"\nSaved: {out}")
plt.close()
