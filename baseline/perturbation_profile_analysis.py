"""
perturbation_profile_analysis.py
================================

Analyze perturbation effect profiles across datasets:
1. Nearest neighbor analysis - do same perturbations cluster across datasets?
2. Effect consistency score - how consistent is a perturbation's effect across datasets?
3. Transfer prediction - can we predict effects in one dataset from another?

Uses shared (pert, gene) pairs from concordance analysis.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cosine, cdist
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error, r2_score
from collections import defaultdict
from tqdm import tqdm


OUTPUT_DIR = "/Users/yiquntchen/Desktop/chen-lab/Project-Perturb/output/profile_analysis"


# =============================================================================
# 2. NEAREST NEIGHBOR ANALYSIS
# =============================================================================
#
# Idea: For each perturbation in dataset A, find its k-nearest neighbors in
# dataset B based on logFC profile similarity (cosine distance).
#
# Key question: Is the same perturbation (same gene knocked out) among the
# top-k neighbors? This tells us whether perturbation effects are reproducible.
#
# Metrics:
# - Recall@k: Fraction of perturbations where the same pert is in top-k
# - Mean reciprocal rank (MRR): Average of 1/rank for the same pert
# - Median rank: Where does the same pert typically rank?
#
# Output:
# - Distribution of ranks for same-pert matches
# - Recall@k curves
# - Examples of perturbations with high/low reproducibility
# =============================================================================

def build_perturbation_profiles(shared_pairs_df, name1, name2):
    """
    Build perturbation profiles (logFC vectors) for each dataset.

    Args:
        shared_pairs_df: DataFrame with columns [perturbation, gene, logfc_{name1}, logfc_{name2}]

    Returns:
        profiles1: dict {pert -> {gene -> logfc}}
        profiles2: dict {pert -> {gene -> logfc}}
        common_genes: set of genes present for building aligned vectors
    """
    profiles1 = defaultdict(dict)
    profiles2 = defaultdict(dict)

    for _, row in shared_pairs_df.iterrows():
        pert, gene = row['perturbation'], row['gene']
        profiles1[pert][gene] = row[f'logfc_{name1}']
        profiles2[pert][gene] = row[f'logfc_{name2}']

    return dict(profiles1), dict(profiles2)


def align_profiles_to_matrix(profiles, gene_list):
    """
    Convert profiles dict to matrix aligned to gene_list.

    Returns:
        matrix: (n_perts, n_genes) array
        pert_list: list of perturbation names in order
    """
    pert_list = list(profiles.keys())
    matrix = np.zeros((len(pert_list), len(gene_list)))

    for i, pert in enumerate(pert_list):
        for j, gene in enumerate(gene_list):
            if gene in profiles[pert]:
                matrix[i, j] = profiles[pert][gene]

    return matrix, pert_list


def nearest_neighbor_analysis(shared_pairs_df, name1, name2, k_values=[1, 5, 10, 50]):
    """
    For each perturbation in dataset1, find nearest neighbors in dataset2.
    Check if the same perturbation is among top-k.

    Args:
        shared_pairs_df: DataFrame from concordance analysis
        name1, name2: dataset names
        k_values: list of k values for recall@k

    Returns:
        results: dict with metrics and detailed results
    """
    print(f"  Building perturbation profiles...")
    profiles1, profiles2 = build_perturbation_profiles(shared_pairs_df, name1, name2)

    # Find perturbations present in both
    common_perts = set(profiles1.keys()) & set(profiles2.keys())
    print(f"  {len(common_perts)} perturbations in both datasets")

    if len(common_perts) < 10:
        print("  Not enough common perturbations for NN analysis")
        return None

    # For each common pert, get the genes it was measured on in both datasets
    # Build per-perturbation gene lists (only genes measured in BOTH for that pert)
    pert_gene_lists = {}
    for pert in common_perts:
        genes1 = set(profiles1[pert].keys())
        genes2 = set(profiles2[pert].keys())
        pert_gene_lists[pert] = genes1 & genes2

    # Use union of all genes for global alignment
    all_genes = set()
    for pert in common_perts:
        all_genes.update(pert_gene_lists[pert])
    gene_list = sorted(all_genes)
    print(f"  {len(gene_list)} total genes across common perturbations")

    # Build aligned matrices
    print(f"  Aligning profiles to matrix...")
    mat1, perts1 = align_profiles_to_matrix(profiles1, gene_list)
    mat2, perts2 = align_profiles_to_matrix(profiles2, gene_list)

    # Only keep common perturbations for fair comparison
    common_perts_list = sorted(common_perts)
    idx1 = [perts1.index(p) for p in common_perts_list]
    idx2 = [perts2.index(p) for p in common_perts_list]

    mat1_common = mat1[idx1]
    mat2_common = mat2[idx2]

    # Compute pairwise distances (cosine)
    print(f"  Computing pairwise distances...")
    # Distance from each pert in ds1 to each pert in ds2
    distances = cdist(mat1_common, mat2_common, metric='cosine')

    # For each perturbation in ds1, find rank of same pert in ds2
    ranks = []
    for i, pert in enumerate(common_perts_list):
        # Sort by distance, find where same pert (index i in ds2) ranks
        sorted_indices = np.argsort(distances[i])
        rank = np.where(sorted_indices == i)[0][0] + 1  # 1-indexed
        ranks.append(rank)

    ranks = np.array(ranks)

    # Compute metrics
    results = {
        'n_perturbations': len(common_perts_list),
        'n_genes': len(gene_list),
        'median_rank': np.median(ranks),
        'mean_rank': np.mean(ranks),
        'mrr': np.mean(1.0 / ranks),  # Mean reciprocal rank
        'ranks': ranks,
        'perturbations': common_perts_list,
    }

    for k in k_values:
        results[f'recall@{k}'] = np.mean(ranks <= k)

    return results


def plot_nearest_neighbor_results(results, name1, name2, output_path):
    """Plot nearest neighbor analysis results."""
    if results is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Rank distribution
    ax = axes[0]
    ax.hist(results['ranks'], bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(results['median_rank'], color='red', linestyle='--',
               label=f"Median: {results['median_rank']:.0f}")
    ax.axvline(1, color='green', linestyle='--', label='Perfect match')
    ax.set_xlabel('Rank of same perturbation')
    ax.set_ylabel('Count')
    ax.set_title(f'Rank Distribution\nMRR={results["mrr"]:.3f}')
    ax.legend()

    # Recall@k curve
    ax = axes[1]
    k_values = [1, 5, 10, 20, 50, 100, 200]
    recalls = [np.mean(results['ranks'] <= k) for k in k_values]
    ax.plot(k_values, recalls, 'bo-', linewidth=2, markersize=8)
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('k')
    ax.set_ylabel('Recall@k')
    ax.set_title('Recall@k Curve')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    # Cumulative distribution
    ax = axes[2]
    sorted_ranks = np.sort(results['ranks'])
    cumulative = np.arange(1, len(sorted_ranks) + 1) / len(sorted_ranks)
    ax.plot(sorted_ranks, cumulative, linewidth=2)
    ax.axvline(results['median_rank'], color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('Rank')
    ax.set_ylabel('Cumulative fraction')
    ax.set_title('Cumulative Rank Distribution')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Nearest Neighbor Analysis: {name1} vs {name2}\n'
                 f'{results["n_perturbations"]} perturbations, {results["n_genes"]} genes',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# =============================================================================
# 5. EFFECT CONSISTENCY SCORE
# =============================================================================
#
# Idea: For each perturbation present in multiple datasets, compute how
# consistent its effect is across datasets.
#
# Metrics per perturbation:
# - Pairwise logFC correlation between datasets
# - Direction agreement (what fraction of genes have same up/down/not_de)
# - Effect magnitude consistency (correlation of |logFC|)
#
# Output:
# - Distribution of consistency scores across perturbations
# - Identify highly consistent vs inconsistent perturbations
# - Factors associated with consistency (e.g., effect size, # genes affected)
# =============================================================================

def compute_effect_consistency(shared_pairs_df, name1, name2):
    """
    Compute effect consistency for each perturbation across two datasets.

    Returns:
        DataFrame with consistency metrics per perturbation
    """
    results = []

    for pert, group in shared_pairs_df.groupby('perturbation'):
        if len(group) < 10:  # Need enough genes
            continue

        logfc1 = group[f'logfc_{name1}'].values
        logfc2 = group[f'logfc_{name2}'].values
        dir1 = group[f'direction_{name1}'].values
        dir2 = group[f'direction_{name2}'].values

        # LogFC correlation
        logfc_r, _ = pearsonr(logfc1, logfc2)
        logfc_rho, _ = spearmanr(logfc1, logfc2)

        # Direction agreement
        dir_agreement = np.mean(dir1 == dir2)

        # Per-class agreement
        up_mask = dir1 == 'up'
        down_mask = dir1 == 'down'
        notde_mask = dir1 == 'not_de'

        up_agreement = np.mean(dir1[up_mask] == dir2[up_mask]) if up_mask.sum() > 0 else np.nan
        down_agreement = np.mean(dir1[down_mask] == dir2[down_mask]) if down_mask.sum() > 0 else np.nan
        notde_agreement = np.mean(dir1[notde_mask] == dir2[notde_mask]) if notde_mask.sum() > 0 else np.nan

        # Effect magnitude consistency
        mag_r, _ = pearsonr(np.abs(logfc1), np.abs(logfc2))

        # Effect size metrics
        n_de1 = np.sum(dir1 != 'not_de')
        n_de2 = np.sum(dir2 != 'not_de')
        mean_abs_logfc1 = np.mean(np.abs(logfc1))
        mean_abs_logfc2 = np.mean(np.abs(logfc2))

        results.append({
            'perturbation': pert,
            'n_genes': len(group),
            'logfc_pearson_r': logfc_r,
            'logfc_spearman_rho': logfc_rho,
            'direction_agreement': dir_agreement,
            'up_agreement': up_agreement,
            'down_agreement': down_agreement,
            'notde_agreement': notde_agreement,
            'magnitude_r': mag_r,
            f'n_de_{name1}': n_de1,
            f'n_de_{name2}': n_de2,
            f'mean_abs_logfc_{name1}': mean_abs_logfc1,
            f'mean_abs_logfc_{name2}': mean_abs_logfc2,
        })

    return pd.DataFrame(results)


def plot_effect_consistency(consistency_df, name1, name2, output_path):
    """Plot effect consistency analysis results."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # LogFC correlation distribution
    ax = axes[0, 0]
    ax.hist(consistency_df['logfc_pearson_r'].dropna(), bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(consistency_df['logfc_pearson_r'].median(), color='red', linestyle='--',
               label=f"Median: {consistency_df['logfc_pearson_r'].median():.3f}")
    ax.set_xlabel('LogFC Pearson r')
    ax.set_ylabel('Count')
    ax.set_title('LogFC Correlation Distribution')
    ax.legend()

    # Direction agreement distribution
    ax = axes[0, 1]
    ax.hist(consistency_df['direction_agreement'], bins=30, edgecolor='black', alpha=0.7, color='green')
    ax.axvline(consistency_df['direction_agreement'].median(), color='red', linestyle='--',
               label=f"Median: {consistency_df['direction_agreement'].median():.3f}")
    ax.set_xlabel('Direction Agreement')
    ax.set_ylabel('Count')
    ax.set_title('Direction Agreement Distribution')
    ax.legend()

    # LogFC r vs direction agreement
    ax = axes[0, 2]
    ax.scatter(consistency_df['logfc_pearson_r'], consistency_df['direction_agreement'],
               alpha=0.5, s=20)
    ax.set_xlabel('LogFC Pearson r')
    ax.set_ylabel('Direction Agreement')
    ax.set_title('LogFC Correlation vs Direction Agreement')

    # Consistency vs effect size
    ax = axes[1, 0]
    ax.scatter(consistency_df[f'mean_abs_logfc_{name1}'], consistency_df['logfc_pearson_r'],
               alpha=0.5, s=20)
    ax.set_xlabel(f'Mean |logFC| in {name1}')
    ax.set_ylabel('LogFC Pearson r')
    ax.set_title('Consistency vs Effect Size')

    # Consistency vs number of DE genes
    ax = axes[1, 1]
    ax.scatter(consistency_df[f'n_de_{name1}'], consistency_df['logfc_pearson_r'],
               alpha=0.5, s=20)
    ax.set_xlabel(f'# DE genes in {name1}')
    ax.set_ylabel('LogFC Pearson r')
    ax.set_title('Consistency vs # DE Genes')

    # Per-class agreement comparison
    ax = axes[1, 2]
    class_cols = ['up_agreement', 'down_agreement', 'notde_agreement']
    class_names = ['Up', 'Down', 'Not DE']
    means = [consistency_df[c].mean() for c in class_cols]
    stds = [consistency_df[c].std() for c in class_cols]
    colors = ['green', 'red', 'gray']
    ax.bar(class_names, means, yerr=stds, color=colors, alpha=0.7, capsize=5)
    ax.set_ylabel('Mean Agreement')
    ax.set_title('Per-Class Agreement')
    ax.set_ylim(0, 1)

    plt.suptitle(f'Effect Consistency: {name1} vs {name2}\n'
                 f'{len(consistency_df)} perturbations with 10+ genes', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# =============================================================================
# 6. TRANSFER PREDICTION
# =============================================================================
#
# Idea: Can we predict perturbation effects in dataset B given effects in A?
#
# Approaches:
# a) Direct transfer: Use logFC from A to predict logFC in B (same pert, same gene)
#    - Baseline: Just use logFC_A as prediction for logFC_B
#    - This measures how well effects transfer directly
#
# b) kNN transfer: For a perturbation in B, find similar perturbations in A,
#    use their effects to predict B's effects
#    - Split perturbations: some in both (train), some only in B (test)
#    - For test perts, find kNN among train perts, average their effects
#
# c) Gene-wise transfer: For each gene, learn a simple model (linear) mapping
#    logFC_A -> logFC_B across perturbations
#
# Metrics:
# - MSE, R^2, Pearson r between predicted and actual logFC
# - Direction accuracy (predict up/down/not_de)
# =============================================================================

def direct_transfer_analysis(shared_pairs_df, name1, name2):
    """
    Analyze direct transfer: use logFC from dataset1 to predict dataset2.

    This is the simplest baseline - just use the same perturbation's effect
    from one dataset to predict the other.
    """
    logfc1 = shared_pairs_df[f'logfc_{name1}'].values
    logfc2 = shared_pairs_df[f'logfc_{name2}'].values
    dir1 = shared_pairs_df[f'direction_{name1}'].values
    dir2 = shared_pairs_df[f'direction_{name2}'].values

    # Regression metrics
    mse = mean_squared_error(logfc2, logfc1)
    r2 = r2_score(logfc2, logfc1)
    pearson_r, _ = pearsonr(logfc1, logfc2)

    # Classification metrics (using dir1 to predict dir2)
    dir_accuracy = np.mean(dir1 == dir2)

    return {
        'mse': mse,
        'rmse': np.sqrt(mse),
        'r2': r2,
        'pearson_r': pearson_r,
        'direction_accuracy': dir_accuracy,
        'n_pairs': len(logfc1),
    }


def knn_transfer_analysis(shared_pairs_df, name1, name2, k=5, test_frac=0.2):
    """
    kNN transfer: predict effects in dataset2 using similar perturbations from dataset1.

    Method:
    1. Split perturbations into train (80%) and test (20%)
    2. For test perturbations, find k nearest train perturbations (by logFC profile in ds1)
    3. Predict test pert's logFC in ds2 as weighted average of neighbors' logFC in ds2
    4. Compare predictions to actual logFC in ds2
    """
    # Build perturbation profiles
    profiles1, profiles2 = build_perturbation_profiles(shared_pairs_df, name1, name2)
    common_perts = sorted(set(profiles1.keys()) & set(profiles2.keys()))

    if len(common_perts) < 20:
        return None

    # Get common genes
    all_genes = set()
    for pert in common_perts:
        genes = set(profiles1[pert].keys()) & set(profiles2[pert].keys())
        all_genes.update(genes)
    gene_list = sorted(all_genes)

    # Build matrices
    mat1, _ = align_profiles_to_matrix({p: profiles1[p] for p in common_perts}, gene_list)
    mat2, _ = align_profiles_to_matrix({p: profiles2[p] for p in common_perts}, gene_list)

    # Train/test split
    np.random.seed(42)
    n_test = max(5, int(len(common_perts) * test_frac))
    test_idx = np.random.choice(len(common_perts), n_test, replace=False)
    train_idx = np.array([i for i in range(len(common_perts)) if i not in test_idx])

    mat1_train, mat2_train = mat1[train_idx], mat2[train_idx]
    mat1_test, mat2_test = mat1[test_idx], mat2[test_idx]

    # For each test perturbation, find kNN in train set (based on ds1 profiles)
    nn = NearestNeighbors(n_neighbors=min(k, len(train_idx)), metric='cosine')
    nn.fit(mat1_train)

    distances, indices = nn.kneighbors(mat1_test)

    # Predict: weighted average of neighbors' ds2 profiles
    # Weight by inverse distance
    weights = 1 / (distances + 1e-6)
    weights = weights / weights.sum(axis=1, keepdims=True)

    predictions = np.zeros_like(mat2_test)
    for i in range(len(test_idx)):
        for j, neighbor_idx in enumerate(indices[i]):
            predictions[i] += weights[i, j] * mat2_train[neighbor_idx]

    # Evaluate
    actual = mat2_test.flatten()
    predicted = predictions.flatten()

    # Remove zeros (genes not measured)
    mask = (actual != 0) | (predicted != 0)
    actual = actual[mask]
    predicted = predicted[mask]

    mse = mean_squared_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    pearson_r, _ = pearsonr(actual, predicted)

    return {
        'mse': mse,
        'rmse': np.sqrt(mse),
        'r2': r2,
        'pearson_r': pearson_r,
        'n_train_perts': len(train_idx),
        'n_test_perts': len(test_idx),
        'k': k,
    }


def plot_transfer_analysis(direct_results, knn_results, name1, name2,
                           shared_pairs_df, output_path):
    """Plot transfer analysis results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Direct transfer scatter
    ax = axes[0]
    plot_df = shared_pairs_df.sample(min(10000, len(shared_pairs_df)), random_state=42)
    ax.scatter(plot_df[f'logfc_{name1}'], plot_df[f'logfc_{name2}'], alpha=0.1, s=5)

    # Add regression line
    x = plot_df[f'logfc_{name1}'].values
    y = plot_df[f'logfc_{name2}'].values
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'y={z[0]:.2f}x+{z[1]:.2f}')
    ax.plot(x_line, x_line, 'k--', alpha=0.5, label='y=x')

    ax.set_xlabel(f'{name1} logFC')
    ax.set_ylabel(f'{name2} logFC')
    ax.set_title(f'Direct Transfer\nr={direct_results["pearson_r"]:.3f}, '
                 f'R²={direct_results["r2"]:.3f}')
    ax.legend()

    # Comparison bar chart
    ax = axes[1]
    methods = ['Direct', 'kNN']
    r_values = [direct_results['pearson_r'],
                knn_results['pearson_r'] if knn_results else 0]
    colors = ['steelblue', 'coral']
    ax.bar(methods, r_values, color=colors, alpha=0.7)
    ax.set_ylabel('Pearson r')
    ax.set_title('Transfer Method Comparison')
    ax.set_ylim(0, 1)

    # Metrics table
    ax = axes[2]
    ax.axis('off')

    table_data = [
        ['Metric', 'Direct', 'kNN'],
        ['Pearson r', f"{direct_results['pearson_r']:.3f}",
         f"{knn_results['pearson_r']:.3f}" if knn_results else 'N/A'],
        ['R²', f"{direct_results['r2']:.3f}",
         f"{knn_results['r2']:.3f}" if knn_results else 'N/A'],
        ['RMSE', f"{direct_results['rmse']:.3f}",
         f"{knn_results['rmse']:.3f}" if knn_results else 'N/A'],
        ['Dir Acc', f"{direct_results['direction_accuracy']:.3f}", 'N/A'],
    ]

    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    ax.set_title('Transfer Metrics Summary')

    plt.suptitle(f'Transfer Prediction: {name1} → {name2}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def run_profile_analysis(shared_pairs_csv, name1, name2):
    """
    Run all profile analyses on a pair of datasets.

    Args:
        shared_pairs_csv: Path to CSV from concordance analysis
        name1, name2: Dataset names
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Profile Analysis: {name1} vs {name2}")
    print('='*60)

    # Load data
    df = pd.read_csv(shared_pairs_csv)
    print(f"Loaded {len(df):,} (pert,gene) pairs")

    # 2. Nearest neighbor analysis
    print("\n--- Nearest Neighbor Analysis ---")
    nn_results = nearest_neighbor_analysis(df, name1, name2)
    if nn_results:
        print(f"  Recall@1: {nn_results['recall@1']:.3f}")
        print(f"  Recall@10: {nn_results['recall@10']:.3f}")
        print(f"  MRR: {nn_results['mrr']:.3f}")
        print(f"  Median rank: {nn_results['median_rank']:.0f}")
        plot_nearest_neighbor_results(nn_results, name1, name2,
            os.path.join(OUTPUT_DIR, f"nn_analysis_{name1}_vs_{name2}.pdf"))

    # 5. Effect consistency
    print("\n--- Effect Consistency Analysis ---")
    consistency_df = compute_effect_consistency(df, name1, name2)
    print(f"  {len(consistency_df)} perturbations analyzed")
    print(f"  Median logFC r: {consistency_df['logfc_pearson_r'].median():.3f}")
    print(f"  Median dir agreement: {consistency_df['direction_agreement'].median():.3f}")
    consistency_df.to_csv(os.path.join(OUTPUT_DIR, f"consistency_{name1}_vs_{name2}.csv"), index=False)
    plot_effect_consistency(consistency_df, name1, name2,
        os.path.join(OUTPUT_DIR, f"consistency_{name1}_vs_{name2}.pdf"))

    # 6. Transfer prediction
    print("\n--- Transfer Prediction Analysis ---")
    direct_results = direct_transfer_analysis(df, name1, name2)
    print(f"  Direct transfer r: {direct_results['pearson_r']:.3f}")
    print(f"  Direct transfer R²: {direct_results['r2']:.3f}")

    knn_results = knn_transfer_analysis(df, name1, name2, k=5)
    if knn_results:
        print(f"  kNN transfer r: {knn_results['pearson_r']:.3f}")
        print(f"  kNN transfer R²: {knn_results['r2']:.3f}")

    plot_transfer_analysis(direct_results, knn_results, name1, name2, df,
        os.path.join(OUTPUT_DIR, f"transfer_{name1}_vs_{name2}.pdf"))

    return {
        'nn_results': nn_results,
        'consistency_df': consistency_df,
        'direct_results': direct_results,
        'knn_results': knn_results,
    }


def main():
    """Run profile analysis on all dataset pairs with shared data."""
    concordance_dir = "/Users/yiquntchen/Desktop/chen-lab/Project-Perturb/output/concordance"

    # Find all shared_pairs CSV files
    import glob
    csv_files = glob.glob(os.path.join(concordance_dir, "shared_pairs_*.csv"))

    if not csv_files:
        print("No shared_pairs CSV files found. Run concordance_analysis.py first.")
        return

    print(f"Found {len(csv_files)} dataset pairs to analyze")

    all_results = {}
    for csv_path in tqdm(csv_files, desc="Analyzing pairs"):
        # Extract dataset names from filename
        basename = os.path.basename(csv_path)
        # Format: shared_pairs_{name1}_vs_{name2}.csv
        parts = basename.replace('shared_pairs_', '').replace('.csv', '').split('_vs_')
        if len(parts) == 2:
            name1, name2 = parts
            all_results[(name1, name2)] = run_profile_analysis(csv_path, name1, name2)

    print(f"\nDone! Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
