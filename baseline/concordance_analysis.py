"""
concordance_analysis.py
=======================

Analyze concordance of DE classes (up/down/not_de) and logFC agreement
across different cell line datasets.

ONLY compares (perturbation, gene) pairs that exist in BOTH datasets.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
from tqdm import tqdm


DATA_DIR = "/Users/yiquntchen/Desktop/chen-lab/perturbevovle/problems/perturbation_prediction/data"
OUTPUT_DIR = "/Users/yiquntchen/Desktop/chen-lab/Project-Perturb/output/concordance"

DATASETS = ['adamson', 'hepg2', 'jurkat', 'replogle_k562', 'replogle_rpe1', 'tcell_rest']


def convert_dict_format_to_list(data):
    """Convert dict format to list format."""
    records = []
    keys = data['keys']
    labels = data['labels']
    logfc = data['logfc']

    for i in range(len(keys)):
        records.append({
            'perturbation': keys[i][0],
            'target_gene': keys[i][1],
            'direction': labels[i],
            'logfc': logfc[i]
        })
    return records


def load_dataset(dataset_name):
    """Load ground truth data from JSON files."""
    data_path = os.path.join(DATA_DIR, dataset_name)
    all_data = []

    train_path = os.path.join(data_path, "train.json")
    is_dict_format = False
    if os.path.exists(train_path):
        with open(train_path, 'r') as f:
            sample = json.load(f)
            is_dict_format = isinstance(sample, dict) and 'keys' in sample

    if is_dict_format:
        splits = [
            ('train', 'train.json', None),
            ('val', 'val.json', 'val_ground_truth.json'),
            ('test', 'test.json', 'test_ground_truth.json')
        ]
        for split_name, keys_file, gt_file in splits:
            keys_path = os.path.join(data_path, keys_file)
            if not os.path.exists(keys_path):
                continue
            with open(keys_path, 'r') as f:
                keys_data = json.load(f)
            if gt_file:
                gt_path = os.path.join(data_path, gt_file)
                with open(gt_path, 'r') as f:
                    gt_data = json.load(f)
                merged = {'keys': keys_data['keys'], 'labels': gt_data['labels'], 'logfc': gt_data['logfc']}
                data = convert_dict_format_to_list(merged)
            else:
                data = convert_dict_format_to_list(keys_data)
            all_data.extend(data)
    else:
        for split in ['train', 'val_ground_truth', 'test_ground_truth']:
            filepath = os.path.join(data_path, f"{split}.json")
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    all_data.extend(json.load(f))

    return all_data


def build_pair_index(data, dataset_name=""):
    """Build dict mapping (perturbation, gene) -> {direction, logfc}"""
    pair_index = {}
    for record in tqdm(data, desc=f"  Indexing {dataset_name}", leave=False):
        key = (record['perturbation'], record['target_gene'])
        pair_index[key] = {'direction': record['direction'], 'logfc': record['logfc']}
    return pair_index


def compare_datasets(idx1, idx2, name1, name2, output_dir):
    """
    Compare two datasets on their shared (pert, gene) pairs.
    Returns stats and saves data + plots.
    """
    # Find common pairs
    common_pairs = set(idx1.keys()) & set(idx2.keys())

    if len(common_pairs) == 0:
        return None

    # Build merged dataframe
    rows = []
    for (pert, gene) in common_pairs:
        rows.append({
            'perturbation': pert,
            'gene': gene,
            f'direction_{name1}': idx1[(pert, gene)]['direction'],
            f'direction_{name2}': idx2[(pert, gene)]['direction'],
            f'logfc_{name1}': idx1[(pert, gene)]['logfc'],
            f'logfc_{name2}': idx2[(pert, gene)]['logfc'],
        })

    df = pd.DataFrame(rows)

    # Save clean data
    df.to_csv(os.path.join(output_dir, f"shared_pairs_{name1}_vs_{name2}.csv"), index=False)

    # Compute metrics
    dir1, dir2 = df[f'direction_{name1}'], df[f'direction_{name2}']
    logfc1, logfc2 = df[f'logfc_{name1}'], df[f'logfc_{name2}']

    accuracy = accuracy_score(dir1, dir2)
    kappa = cohen_kappa_score(dir1, dir2)

    classes = ['up', 'down', 'not_de']
    cm = confusion_matrix(dir1, dir2, labels=classes)

    # Per-class agreement (recall for each class)
    class_agreement = {}
    for cls in classes:
        mask = dir1 == cls
        if mask.sum() > 0:
            class_agreement[cls] = (dir1[mask] == dir2[mask]).mean()
        else:
            class_agreement[cls] = np.nan

    # LogFC correlation
    logfc_r, _ = pearsonr(logfc1, logfc2)
    logfc_rho, _ = spearmanr(logfc1, logfc2)

    # Count common perturbations
    common_perts = df['perturbation'].nunique()

    # Create plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Confusion matrix
    ax = axes[0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel(f'{name2}')
    ax.set_ylabel(f'{name1}')
    ax.set_title(f'Direction Confusion Matrix\nAcc={accuracy:.3f}, Kappa={kappa:.3f}')

    # LogFC scatter
    ax = axes[1]
    plot_df = df.sample(min(10000, len(df)), random_state=42)
    ax.scatter(plot_df[f'logfc_{name1}'], plot_df[f'logfc_{name2}'], alpha=0.1, s=5)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    # Add diagonal
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'r--', alpha=0.5)
    ax.set_xlabel(f'{name1} logFC')
    ax.set_ylabel(f'{name2} logFC')
    ax.set_title(f'LogFC Agreement\nr={logfc_r:.3f}, rho={logfc_rho:.3f}')

    # Per-class agreement bars
    ax = axes[2]
    colors = {'up': 'green', 'down': 'red', 'not_de': 'gray'}
    bars = ax.bar(classes, [class_agreement.get(c, 0) for c in classes],
                  color=[colors[c] for c in classes], alpha=0.7)
    ax.axhline(accuracy, color='blue', linestyle='--', label=f'Overall: {accuracy:.3f}')
    ax.set_ylabel('Agreement rate')
    ax.set_title('Per-class Agreement')
    ax.set_ylim(0, 1)
    ax.legend()

    plt.suptitle(f'{name1} vs {name2}\n{common_perts} common perts, {len(df):,} (pert,gene) pairs', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"concordance_{name1}_vs_{name2}.pdf"), dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'n_common_perts': common_perts,
        'n_pairs': len(df),
        'accuracy': accuracy,
        'kappa': kappa,
        'up_agreement': class_agreement.get('up', np.nan),
        'down_agreement': class_agreement.get('down', np.nan),
        'not_de_agreement': class_agreement.get('not_de', np.nan),
        'logfc_pearson_r': logfc_r,
        'logfc_spearman_rho': logfc_rho,
    }


def create_summary_heatmap(results, dataset_names, output_path):
    """Create summary heatmaps."""
    n = len(dataset_names)

    acc_mat = np.eye(n)
    logfc_mat = np.eye(n)

    for (n1, n2), stats in results.items():
        if stats is None:
            continue
        i, j = dataset_names.index(n1), dataset_names.index(n2)
        acc_mat[i, j] = acc_mat[j, i] = stats['accuracy']
        logfc_mat[i, j] = logfc_mat[j, i] = stats['logfc_pearson_r']

    short_names = [n.replace('replogle_', '').replace('tcell_rest', 'tcell') for n in dataset_names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.heatmap(acc_mat, annot=True, fmt='.2f', cmap='RdYlGn',
                xticklabels=short_names, yticklabels=short_names, ax=axes[0], vmin=0, vmax=1)
    axes[0].set_title('Direction Agreement (Accuracy)')

    sns.heatmap(logfc_mat, annot=True, fmt='.2f', cmap='RdYlBu_r',
                xticklabels=short_names, yticklabels=short_names, ax=axes[1], vmin=-1, vmax=1)
    axes[1].set_title('LogFC Correlation (Pearson r)')

    plt.suptitle('Cross-Dataset Concordance (shared pert-gene pairs only)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_markdown_summary(results, dataset_names, output_path):
    """Generate markdown summary."""
    with open(output_path, 'w') as f:
        f.write("# Cross-Dataset Concordance Analysis\n\n")
        f.write("**Method**: Only comparing (perturbation, gene) pairs that exist in BOTH datasets.\n\n")
        f.write(f"**Datasets**: {', '.join(dataset_names)}\n\n")

        f.write("## Summary Table\n\n")
        f.write("| Dataset 1 | Dataset 2 | Common Perts | Pairs | Accuracy | Kappa | Up Agr | Down Agr | NotDE Agr | LogFC r | LogFC rho |\n")
        f.write("|-----------|-----------|--------------|-------|----------|-------|--------|----------|-----------|---------|----------|\n")

        for (n1, n2), stats in sorted(results.items()):
            if stats:
                f.write(f"| {n1} | {n2} | {stats['n_common_perts']} | {stats['n_pairs']:,} | "
                        f"{stats['accuracy']:.3f} | {stats['kappa']:.3f} | "
                        f"{stats['up_agreement']:.3f} | {stats['down_agreement']:.3f} | "
                        f"{stats['not_de_agreement']:.3f} | {stats['logfc_pearson_r']:.3f} | "
                        f"{stats['logfc_spearman_rho']:.3f} |\n")
            else:
                f.write(f"| {n1} | {n2} | 0 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |\n")

        f.write("\n## Interpretation\n\n")
        f.write("- **Accuracy**: Fraction of (pert,gene) pairs with same direction label\n")
        f.write("- **Kappa**: Cohen's kappa (agreement beyond chance)\n")
        f.write("- **Per-class Agreement**: Given class X in dataset1, P(class X in dataset2)\n")
        f.write("- **LogFC r/rho**: Correlation of logFC values for same (pert,gene) pairs\n")

        f.write("\n## Files\n\n")
        f.write("- `concordance_*.pdf`: Confusion matrix + logFC scatter + per-class bars\n")
        f.write("- `shared_pairs_*.csv`: Merged data with columns: perturbation, gene, direction_*, logfc_*\n")
        f.write("- `summary_heatmap.pdf`: Overview heatmaps\n")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find available datasets
    available = [name for name in DATASETS if os.path.exists(os.path.join(DATA_DIR, name))]
    print(f"Available datasets: {available}")
    print(f"Pairwise comparisons: {len(list(combinations(available, 2)))}\n")

    # Load and index all datasets
    print("=" * 60)
    print("STEP 1: Loading datasets")
    print("=" * 60)

    indices = {}
    for name in tqdm(available, desc="Loading"):
        print(f"\n  {name}...")
        data = load_dataset(name)
        print(f"    {len(data):,} records")
        indices[name] = build_pair_index(data, name)
        print(f"    {len(indices[name]):,} unique (pert,gene) pairs")

    # Pairwise comparisons
    print("\n" + "=" * 60)
    print("STEP 2: Pairwise comparisons")
    print("=" * 60)

    results = {}
    pairs = list(combinations(available, 2))

    for n1, n2 in tqdm(pairs, desc="Comparing"):
        results[(n1, n2)] = compare_datasets(indices[n1], indices[n2], n1, n2, OUTPUT_DIR)

    # Summary
    print("\n" + "=" * 60)
    print("STEP 3: Summary")
    print("=" * 60)

    create_summary_heatmap(results, available, os.path.join(OUTPUT_DIR, "summary_heatmap.pdf"))
    generate_markdown_summary(results, available, os.path.join(OUTPUT_DIR, "concordance_summary.md"))

    print(f"\nDone! Results in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
