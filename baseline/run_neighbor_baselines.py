"""
Test neighbor-based prediction approaches for perturbation effect prediction.

Approaches:
1. Majority vote (baseline)
2. Perturbation-neighbor voting: find similar perturbations, vote on target gene's response
3. Target gene-neighbor voting: find similar genes, vote on their response to this perturbation
4. Combined: ensemble of multiple neighbor sources

Usage:
    python run_neighbor_baselines.py --dataset adamson
    python run_neighbor_baselines.py --dataset adamson --methods all
    python run_neighbor_baselines.py --dataset adamson --methods majority pert_neighbor
"""

import os
import json
import argparse
import numpy as np
from collections import defaultdict, Counter
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix
)
from tqdm import tqdm

PROBLEM_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROBLEM_DIR, 'data')
RESULTS_DIR = os.path.join(PROBLEM_DIR, 'neighbor_baseline_results')

ALL_DATASETS = ['adamson', 'replogle_k562', 'replogle_rpe1', 'hepg2', 'jurkat', 'tcell_rest']

# Load neighbor files
def load_neighbors():
    """Load pre-computed neighbor files."""
    neighbors = {}

    esm2_path = os.path.join(PROBLEM_DIR, 'gene_neighbors_esm2.json')
    if os.path.exists(esm2_path):
        with open(esm2_path) as f:
            neighbors['esm2'] = json.load(f)
        print(f"  Loaded ESM2 neighbors: {len(neighbors['esm2'])} genes")

    genept_path = os.path.join(PROBLEM_DIR, 'gene_neighbors_genept.json')
    if os.path.exists(genept_path):
        with open(genept_path) as f:
            neighbors['genept'] = json.load(f)
        print(f"  Loaded GenePT neighbors: {len(neighbors['genept'])} genes")

    copath_path = os.path.join(PROBLEM_DIR, 'gene_copath_neighbors.json')
    if os.path.exists(copath_path):
        with open(copath_path) as f:
            neighbors['copath'] = json.load(f)
        print(f"  Loaded co-pathway neighbors: {len(neighbors['copath'])} genes")

    return neighbors


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def normalize_data(data, is_ground_truth=False):
    """Convert dict format to list-of-dicts format."""
    if isinstance(data, list):
        return data
    keys = data.get('keys', [])
    labels = data.get('labels', [])
    logfcs = data.get('logfc', [0.0] * max(len(keys), len(labels)))

    if is_ground_truth and not keys and labels:
        return [{'direction': labels[i], 'logfc': logfcs[i] if i < len(logfcs) else 0.0}
                for i in range(len(labels))]

    result = []
    for i, key in enumerate(keys):
        result.append({
            'perturbation': key[0],
            'target_gene': key[1],
            'direction': labels[i] if i < len(labels) else 'not_de',
            'logfc': logfcs[i] if i < len(logfcs) else 0.0
        })
    return result


def build_indices(train_data):
    """Build lookup indices for fast neighbor queries."""
    # (perturbation, target_gene) -> label, logfc
    pair_to_label = {}
    pair_to_logfc = {}

    # perturbation -> {target_gene -> label}
    pert_to_genes = defaultdict(dict)

    # target_gene -> {perturbation -> label}
    gene_to_perts = defaultdict(dict)

    # Per-gene stats for majority vote fallback
    gene_stats = defaultdict(lambda: {'up': 0, 'down': 0, 'not_de': 0, 'logfc_sum': 0, 'count': 0})

    # Per-perturbation stats
    pert_stats = defaultdict(lambda: {'up': 0, 'down': 0, 'not_de': 0, 'logfc_sum': 0, 'count': 0})

    for r in tqdm(train_data, desc="Building indices"):
        p, g, label, logfc = r['perturbation'], r['target_gene'], r['direction'], r['logfc']

        pair_to_label[(p, g)] = label
        pair_to_logfc[(p, g)] = logfc
        pert_to_genes[p][g] = label
        gene_to_perts[g][p] = label

        gene_stats[g][label] += 1
        gene_stats[g]['logfc_sum'] += logfc
        gene_stats[g]['count'] += 1

        pert_stats[p][label] += 1
        pert_stats[p]['logfc_sum'] += logfc
        pert_stats[p]['count'] += 1

    # Global stats
    label_counts = Counter(r['direction'] for r in train_data)
    global_majority = label_counts.most_common(1)[0][0]
    global_mean_logfc = sum(r['logfc'] for r in train_data) / len(train_data)

    return {
        'pair_to_label': pair_to_label,
        'pair_to_logfc': pair_to_logfc,
        'pert_to_genes': dict(pert_to_genes),
        'gene_to_perts': dict(gene_to_perts),
        'gene_stats': dict(gene_stats),
        'pert_stats': dict(pert_stats),
        'global_majority': global_majority,
        'global_mean_logfc': global_mean_logfc,
        'all_perts': set(pert_to_genes.keys()),
        'all_genes': set(gene_to_perts.keys()),
    }


# ============== PREDICTION METHODS ==============

def predict_majority_gene(test_data, indices, neighbors=None, **kwargs):
    """Baseline: per-gene majority vote. Ignores perturbation."""
    predictions = []
    logfcs = []

    for r in tqdm(test_data, desc="Majority (gene)"):
        g = r['target_gene']
        if g in indices['gene_stats']:
            stats = indices['gene_stats'][g]
            pred = max(['up', 'down', 'not_de'], key=lambda x: stats[x])
            logfc = stats['logfc_sum'] / stats['count'] if stats['count'] > 0 else indices['global_mean_logfc']
        else:
            pred = indices['global_majority']
            logfc = indices['global_mean_logfc']
        predictions.append(pred)
        logfcs.append(logfc)

    return predictions, logfcs


def predict_majority_pert(test_data, indices, neighbors=None, **kwargs):
    """Baseline: per-perturbation majority vote. Ignores target gene."""
    predictions = []
    logfcs = []

    for r in tqdm(test_data, desc="Majority (pert)"):
        p = r['perturbation']
        if p in indices['pert_stats']:
            stats = indices['pert_stats'][p]
            pred = max(['up', 'down', 'not_de'], key=lambda x: stats[x])
            logfc = stats['logfc_sum'] / stats['count'] if stats['count'] > 0 else indices['global_mean_logfc']
        else:
            pred = indices['global_majority']
            logfc = indices['global_mean_logfc']
        predictions.append(pred)
        logfcs.append(logfc)

    return predictions, logfcs


def predict_pert_neighbor(test_data, indices, neighbors, neighbor_type='esm2', k=10, **kwargs):
    """
    Perturbation-neighbor voting:
    Find similar perturbations P', look up what happened to target gene G when P' was knocked out.
    """
    neighbor_dict = neighbors.get(neighbor_type, {})
    predictions = []
    logfcs = []

    for r in tqdm(test_data, desc=f"Pert-neighbor ({neighbor_type})"):
        p, g = r['perturbation'], r['target_gene']

        # Get neighbors of the perturbation gene
        pert_neighbors = neighbor_dict.get(p, [])[:k]

        # Vote: what happened to G when similar perturbations were applied?
        votes = {'up': 0, 'down': 0, 'not_de': 0}
        logfc_vals = []

        for i, p_neighbor in enumerate(pert_neighbors):
            weight = 1.0 / (i + 1)  # Rank-based weighting (closer = higher weight)
            if (p_neighbor, g) in indices['pair_to_label']:
                label = indices['pair_to_label'][(p_neighbor, g)]
                votes[label] += weight
                logfc_vals.append(indices['pair_to_logfc'][(p_neighbor, g)] * weight)

        # If we got votes, use them; otherwise fallback to gene majority
        if sum(votes.values()) > 0:
            pred = max(votes.keys(), key=lambda x: votes[x])
            logfc = sum(logfc_vals) / sum(1/(i+1) for i in range(len(logfc_vals))) if logfc_vals else indices['global_mean_logfc']
        else:
            # Fallback to gene-level majority
            if g in indices['gene_stats']:
                stats = indices['gene_stats'][g]
                pred = max(['up', 'down', 'not_de'], key=lambda x: stats[x])
                logfc = stats['logfc_sum'] / stats['count'] if stats['count'] > 0 else indices['global_mean_logfc']
            else:
                pred = indices['global_majority']
                logfc = indices['global_mean_logfc']

        predictions.append(pred)
        logfcs.append(logfc)

    return predictions, logfcs


def predict_gene_neighbor(test_data, indices, neighbors, neighbor_type='esm2', k=10, **kwargs):
    """
    Target gene-neighbor voting:
    Find similar target genes G', look up what happened to G' when perturbation P was applied.
    """
    neighbor_dict = neighbors.get(neighbor_type, {})
    predictions = []
    logfcs = []

    for r in tqdm(test_data, desc=f"Gene-neighbor ({neighbor_type})"):
        p, g = r['perturbation'], r['target_gene']

        # Get neighbors of the target gene
        gene_neighbors = neighbor_dict.get(g, [])[:k]

        # Vote: what happened to similar genes when P was applied?
        votes = {'up': 0, 'down': 0, 'not_de': 0}
        logfc_vals = []

        for i, g_neighbor in enumerate(gene_neighbors):
            weight = 1.0 / (i + 1)
            if (p, g_neighbor) in indices['pair_to_label']:
                label = indices['pair_to_label'][(p, g_neighbor)]
                votes[label] += weight
                logfc_vals.append(indices['pair_to_logfc'][(p, g_neighbor)] * weight)

        if sum(votes.values()) > 0:
            pred = max(votes.keys(), key=lambda x: votes[x])
            logfc = sum(logfc_vals) / sum(1/(i+1) for i in range(len(logfc_vals))) if logfc_vals else indices['global_mean_logfc']
        else:
            if g in indices['gene_stats']:
                stats = indices['gene_stats'][g]
                pred = max(['up', 'down', 'not_de'], key=lambda x: stats[x])
                logfc = stats['logfc_sum'] / stats['count'] if stats['count'] > 0 else indices['global_mean_logfc']
            else:
                pred = indices['global_majority']
                logfc = indices['global_mean_logfc']

        predictions.append(pred)
        logfcs.append(logfc)

    return predictions, logfcs


def predict_both_neighbors(test_data, indices, neighbors, neighbor_type='esm2', k=10, **kwargs):
    """
    Combined: vote using both perturbation neighbors AND target gene neighbors.
    """
    neighbor_dict = neighbors.get(neighbor_type, {})
    predictions = []
    logfcs = []

    for r in tqdm(test_data, desc=f"Both-neighbors ({neighbor_type})"):
        p, g = r['perturbation'], r['target_gene']

        votes = {'up': 0, 'down': 0, 'not_de': 0}
        logfc_vals = []

        # Perturbation neighbors
        pert_neighbors = neighbor_dict.get(p, [])[:k]
        for i, p_neighbor in enumerate(pert_neighbors):
            weight = 1.0 / (i + 1)
            if (p_neighbor, g) in indices['pair_to_label']:
                label = indices['pair_to_label'][(p_neighbor, g)]
                votes[label] += weight
                logfc_vals.append(indices['pair_to_logfc'][(p_neighbor, g)] * weight)

        # Target gene neighbors
        gene_neighbors = neighbor_dict.get(g, [])[:k]
        for i, g_neighbor in enumerate(gene_neighbors):
            weight = 1.0 / (i + 1)
            if (p, g_neighbor) in indices['pair_to_label']:
                label = indices['pair_to_label'][(p, g_neighbor)]
                votes[label] += weight
                logfc_vals.append(indices['pair_to_logfc'][(p, g_neighbor)] * weight)

        if sum(votes.values()) > 0:
            pred = max(votes.keys(), key=lambda x: votes[x])
            logfc = np.mean(logfc_vals) if logfc_vals else indices['global_mean_logfc']
        else:
            if g in indices['gene_stats']:
                stats = indices['gene_stats'][g]
                pred = max(['up', 'down', 'not_de'], key=lambda x: stats[x])
                logfc = stats['logfc_sum'] / stats['count'] if stats['count'] > 0 else indices['global_mean_logfc']
            else:
                pred = indices['global_majority']
                logfc = indices['global_mean_logfc']

        predictions.append(pred)
        logfcs.append(logfc)

    return predictions, logfcs


def predict_ensemble(test_data, indices, neighbors, k=10, **kwargs):
    """
    Ensemble: combine votes from all neighbor sources (ESM2, GenePT, co-pathway).
    """
    predictions = []
    logfcs = []

    # Weights for each source (can be tuned)
    source_weights = {'esm2': 1.0, 'genept': 1.0, 'copath': 1.0}

    for r in tqdm(test_data, desc="Ensemble"):
        p, g = r['perturbation'], r['target_gene']

        votes = {'up': 0, 'down': 0, 'not_de': 0}
        logfc_vals = []

        for source, neighbor_dict in neighbors.items():
            source_weight = source_weights.get(source, 1.0)

            # Perturbation neighbors
            pert_neighbors = neighbor_dict.get(p, [])[:k]
            for i, p_neighbor in enumerate(pert_neighbors):
                weight = source_weight / (i + 1)
                if (p_neighbor, g) in indices['pair_to_label']:
                    label = indices['pair_to_label'][(p_neighbor, g)]
                    votes[label] += weight
                    logfc_vals.append(indices['pair_to_logfc'][(p_neighbor, g)])

            # Target gene neighbors
            gene_neighbors = neighbor_dict.get(g, [])[:k]
            for i, g_neighbor in enumerate(gene_neighbors):
                weight = source_weight / (i + 1)
                if (p, g_neighbor) in indices['pair_to_label']:
                    label = indices['pair_to_label'][(p, g_neighbor)]
                    votes[label] += weight
                    logfc_vals.append(indices['pair_to_logfc'][(p, g_neighbor)])

        if sum(votes.values()) > 0:
            pred = max(votes.keys(), key=lambda x: votes[x])
            logfc = np.mean(logfc_vals) if logfc_vals else indices['global_mean_logfc']
        else:
            if g in indices['gene_stats']:
                stats = indices['gene_stats'][g]
                pred = max(['up', 'down', 'not_de'], key=lambda x: stats[x])
                logfc = stats['logfc_sum'] / stats['count'] if stats['count'] > 0 else indices['global_mean_logfc']
            else:
                pred = indices['global_majority']
                logfc = indices['global_mean_logfc']

        predictions.append(pred)
        logfcs.append(logfc)

    return predictions, logfcs


def predict_two_stage(test_data, indices, neighbors, neighbor_type='esm2', k=10, **kwargs):
    """
    Two-stage prediction (perturbation neighbors only):
    1. First predict: DE or not_de (binary)
    2. If DE, predict: up or down

    Only uses perturbation neighbors - more realistic scenario.
    """
    neighbor_dict = neighbors.get(neighbor_type, {})
    predictions = []
    logfcs = []

    for r in tqdm(test_data, desc=f"Two-stage ({neighbor_type})"):
        p, g = r['perturbation'], r['target_gene']

        # Collect neighbor evidence from similar perturbations only
        de_votes = 0  # up + down
        notde_votes = 0
        up_votes = 0
        down_votes = 0
        logfc_vals = []

        # Check perturbation neighbors only
        pert_neighbors = neighbor_dict.get(p, [])[:k]
        for i, p_neighbor in enumerate(pert_neighbors):
            weight = 1.0 / (i + 1)
            if (p_neighbor, g) in indices['pair_to_label']:
                label = indices['pair_to_label'][(p_neighbor, g)]
                if label == 'not_de':
                    notde_votes += weight
                else:
                    de_votes += weight
                    if label == 'up':
                        up_votes += weight
                    else:
                        down_votes += weight
                logfc_vals.append(indices['pair_to_logfc'][(p_neighbor, g)])

        # Stage 1: DE or not?
        if de_votes + notde_votes > 0:
            # Bias towards DE if we see any strong DE signal
            de_ratio = de_votes / (de_votes + notde_votes)
            if de_ratio > 0.3:  # Lower threshold to catch more DE
                # Stage 2: up or down?
                pred = 'up' if up_votes >= down_votes else 'down'
            else:
                pred = 'not_de'
            logfc = np.mean(logfc_vals) if logfc_vals else indices['global_mean_logfc']
        else:
            # Fallback to gene-level majority
            if g in indices['gene_stats']:
                stats = indices['gene_stats'][g]
                pred = max(['up', 'down', 'not_de'], key=lambda x: stats[x])
                logfc = stats['logfc_sum'] / stats['count'] if stats['count'] > 0 else indices['global_mean_logfc']
            else:
                pred = indices['global_majority']
                logfc = indices['global_mean_logfc']

        predictions.append(pred)
        logfcs.append(logfc)

    return predictions, logfcs


# ============== EVALUATION ==============

def compute_metrics(pred_labels, true_labels, pred_logfcs, true_logfcs):
    """Compute all metrics including per-class accuracy."""
    labels_order = ['up', 'down', 'not_de']

    metrics = {
        'accuracy': accuracy_score(true_labels, pred_labels),
        'macro_f1': f1_score(true_labels, pred_labels, labels=labels_order, average='macro', zero_division=0),
        'macro_precision': precision_score(true_labels, pred_labels, labels=labels_order, average='macro', zero_division=0),
        'macro_recall': recall_score(true_labels, pred_labels, labels=labels_order, average='macro', zero_division=0),
        'per_class': {}
    }

    # Per-class metrics
    per_class_f1 = f1_score(true_labels, pred_labels, labels=labels_order, average=None, zero_division=0)
    per_class_prec = precision_score(true_labels, pred_labels, labels=labels_order, average=None, zero_division=0)
    per_class_rec = recall_score(true_labels, pred_labels, labels=labels_order, average=None, zero_division=0)

    # Per-class accuracy (= recall, but let's compute explicitly for clarity)
    cm = confusion_matrix(true_labels, pred_labels, labels=labels_order)

    for i, label in enumerate(labels_order):
        class_total = cm[i].sum()
        class_correct = cm[i][i]
        class_acc = class_correct / class_total if class_total > 0 else 0

        metrics['per_class'][label] = {
            'f1': per_class_f1[i],
            'precision': per_class_prec[i],
            'recall': per_class_rec[i],
            'accuracy': class_acc,  # Same as recall, but explicit
            'support': int(class_total),  # Number of true samples
            'correct': int(class_correct),
        }

    # LogFC metrics
    pred_logfcs = np.array(pred_logfcs)
    true_logfcs = np.array(true_logfcs)
    metrics['logfc_mae'] = float(np.mean(np.abs(pred_logfcs - true_logfcs)))
    std = np.std(true_logfcs)
    metrics['logfc_nmae'] = metrics['logfc_mae'] / std if std > 0 else 0

    metrics['confusion_matrix'] = cm.tolist()

    return metrics


def print_metrics(method_name, metrics):
    """Pretty print metrics."""
    print(f"\n{'='*60}")
    print(f"Method: {method_name}")
    print(f"{'='*60}")

    print(f"\n--- AGGREGATE METRICS ---")
    print(f"  Accuracy:        {metrics['accuracy']:.4f}")
    print(f"  Macro F1:        {metrics['macro_f1']:.4f}")
    print(f"  Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:    {metrics['macro_recall']:.4f}")

    print(f"\n--- PER-CLASS METRICS ---")
    print(f"  {'Class':<8} {'Acc':>8} {'F1':>8} {'Prec':>8} {'Recall':>8} {'Support':>10}")
    print(f"  {'-'*52}")
    for label in ['up', 'down', 'not_de']:
        pc = metrics['per_class'][label]
        print(f"  {label:<8} {pc['accuracy']:>8.4f} {pc['f1']:>8.4f} {pc['precision']:>8.4f} {pc['recall']:>8.4f} {pc['support']:>10,}")


def save_results(dataset, method_name, metrics, output_dir):
    """Save results to JSON and markdown."""
    os.makedirs(output_dir, exist_ok=True)

    # JSON
    json_path = os.path.join(output_dir, f'{dataset}_{method_name}.json')
    with open(json_path, 'w') as f:
        json.dump({'dataset': dataset, 'method': method_name, **metrics}, f, indent=2)

    # Markdown
    md_path = os.path.join(output_dir, f'{dataset}_{method_name}.md')
    with open(md_path, 'w') as f:
        f.write(f"# {method_name} Results: {dataset.upper()}\n\n")

        f.write(f"## Aggregate Metrics\n")
        f.write(f"| Metric | Value |\n|--------|-------|\n")
        f.write(f"| Accuracy | {metrics['accuracy']:.4f} |\n")
        f.write(f"| Macro F1 | {metrics['macro_f1']:.4f} |\n")
        f.write(f"| Macro Precision | {metrics['macro_precision']:.4f} |\n")
        f.write(f"| Macro Recall | {metrics['macro_recall']:.4f} |\n\n")

        f.write(f"## Per-Class Metrics\n")
        f.write(f"| Class | Accuracy | F1 | Precision | Recall | Support |\n")
        f.write(f"|-------|----------|-----|-----------|--------|--------|\n")
        for label in ['up', 'down', 'not_de']:
            pc = metrics['per_class'][label]
            f.write(f"| {label} | {pc['accuracy']:.4f} | {pc['f1']:.4f} | {pc['precision']:.4f} | {pc['recall']:.4f} | {pc['support']:,} |\n")

        f.write(f"\n## Confusion Matrix\n")
        f.write(f"| | Pred: up | Pred: down | Pred: not_de |\n")
        f.write(f"|---|---|---|---|\n")
        cm = metrics['confusion_matrix']
        for i, label in enumerate(['True: up', 'True: down', 'True: not_de']):
            f.write(f"| {label} | {cm[i][0]:,} | {cm[i][1]:,} | {cm[i][2]:,} |\n")

    print(f"  Saved: {json_path}")
    print(f"  Saved: {md_path}")


# ============== MAIN ==============

METHODS = {
    'majority_gene': (predict_majority_gene, {}),
    'majority_pert': (predict_majority_pert, {}),
    'pert_neighbor_esm2': (predict_pert_neighbor, {'neighbor_type': 'esm2'}),
    'pert_neighbor_genept': (predict_pert_neighbor, {'neighbor_type': 'genept'}),
    'pert_neighbor_copath': (predict_pert_neighbor, {'neighbor_type': 'copath'}),
    'gene_neighbor_esm2': (predict_gene_neighbor, {'neighbor_type': 'esm2'}),
    'gene_neighbor_genept': (predict_gene_neighbor, {'neighbor_type': 'genept'}),
    'gene_neighbor_copath': (predict_gene_neighbor, {'neighbor_type': 'copath'}),
    'both_neighbor_esm2': (predict_both_neighbors, {'neighbor_type': 'esm2'}),
    'both_neighbor_genept': (predict_both_neighbors, {'neighbor_type': 'genept'}),
    'ensemble': (predict_ensemble, {}),
    'two_stage_esm2': (predict_two_stage, {'neighbor_type': 'esm2'}),
    'two_stage_genept': (predict_two_stage, {'neighbor_type': 'genept'}),
}


def run_single_dataset(dataset, methods, k, eval_on_test, neighbors):
    """Run all methods on a single dataset."""
    data_path = os.path.join(DATA_DIR, dataset)
    print(f"\nLoading data from {data_path}...")

    train_data = normalize_data(load_json(os.path.join(data_path, 'train.json')))
    if eval_on_test:
        test_data = normalize_data(load_json(os.path.join(data_path, 'test.json')))
        gt_data = normalize_data(load_json(os.path.join(data_path, 'test_ground_truth.json')), is_ground_truth=True)
    else:
        test_data = normalize_data(load_json(os.path.join(data_path, 'val.json')))
        gt_data = normalize_data(load_json(os.path.join(data_path, 'val_ground_truth.json')), is_ground_truth=True)

    print(f"  Train: {len(train_data):,} samples")
    print(f"  Test:  {len(test_data):,} samples")

    # Build indices
    print("\nBuilding lookup indices...")
    indices = build_indices(train_data)

    # Get ground truth
    true_labels = [r['direction'] for r in gt_data]
    true_logfcs = [r['logfc'] for r in gt_data]

    print(f"\nClass distribution (test): {Counter(true_labels)}")

    # Run methods
    all_results = {}

    for method_name in methods:
        if method_name not in METHODS:
            print(f"\nUnknown method: {method_name}, skipping")
            continue

        predict_fn, method_kwargs = METHODS[method_name]
        print(f"\n>>> Running {method_name}...")

        pred_labels, pred_logfcs = predict_fn(
            test_data, indices, neighbors, k=k, **method_kwargs
        )

        metrics = compute_metrics(pred_labels, true_labels, pred_logfcs, true_logfcs)
        all_results[method_name] = metrics

        print_metrics(method_name, metrics)
        save_results(dataset, method_name, metrics, RESULTS_DIR)

    # Summary comparison
    if len(all_results) > 1:
        print("\n\n" + "="*80)
        print(f"SUMMARY COMPARISON - {dataset.upper()}")
        print("="*80)
        print(f"{'Method':<25} {'Acc':>8} {'MacroF1':>8} {'up_F1':>8} {'down_F1':>8} {'notde_F1':>8}")
        print("-"*80)

        for method_name, m in all_results.items():
            print(f"{method_name:<25} {m['accuracy']:>8.4f} {m['macro_f1']:>8.4f} "
                  f"{m['per_class']['up']['f1']:>8.4f} {m['per_class']['down']['f1']:>8.4f} "
                  f"{m['per_class']['not_de']['f1']:>8.4f}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Test neighbor-based prediction methods')
    parser.add_argument('--dataset', '-d', required=True, nargs='+',
                        help=f'Dataset(s) to evaluate: {ALL_DATASETS} or "all"')
    parser.add_argument('--methods', '-m', nargs='+', default=['majority_gene', 'majority_pert', 'ensemble'],
                        help=f'Methods to test. Options: {list(METHODS.keys())} or "all"')
    parser.add_argument('--k', type=int, default=10, help='Number of neighbors to use')
    parser.add_argument('--val', action='store_true', help='Evaluate on validation set')
    parser.add_argument('--data-dir', default=None,
                        help='Custom data directory (e.g., for PerturbQA splits)')
    args = parser.parse_args()

    # Custom data directory
    global DATA_DIR, RESULTS_DIR
    if args.data_dir:
        DATA_DIR = args.data_dir
        RESULTS_DIR = os.path.join(args.data_dir, 'neighbor_baseline_results')

    # Handle "all" for datasets
    if 'all' in args.dataset:
        if args.data_dir:
            datasets = [d for d in os.listdir(DATA_DIR)
                       if os.path.isdir(os.path.join(DATA_DIR, d)) and d != '__pycache__'
                       and d != 'neighbor_baseline_results']
        else:
            datasets = ALL_DATASETS
    else:
        datasets = args.dataset

    methods = list(METHODS.keys()) if 'all' in args.methods else args.methods
    eval_on_test = not args.val

    print("="*60)
    print("NEIGHBOR-BASED PREDICTION METHODS")
    print(f"Data dir: {DATA_DIR}")
    print(f"Datasets: {datasets}")
    print(f"Methods: {methods}")
    print(f"k (neighbors): {args.k}")
    print("="*60)

    # Load neighbors (shared across datasets)
    print("\nLoading neighbor files...")
    neighbors = load_neighbors()

    # Run each dataset
    for dataset in datasets:
        data_path = os.path.join(DATA_DIR, dataset)
        if not os.path.exists(data_path):
            print(f"\n>>> Skipping {dataset} (not found)")
            continue

        print(f"\n\n{'#'*60}")
        print(f"# DATASET: {dataset.upper()}")
        print(f"{'#'*60}")

        try:
            run_single_dataset(dataset, methods, args.k, eval_on_test, neighbors)
        except Exception as e:
            import traceback
            print(f"Error on {dataset}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
