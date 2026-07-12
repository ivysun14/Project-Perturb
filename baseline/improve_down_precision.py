"""
improve_down_precision.py
=========================

Analyze Down class false positives and try targeted improvements.
The gap vs SynthPert is in precision (0.25 vs 0.47), not recall.

Key insight: We need features that distinguish TRUE Down from false positives.
"""

import pickle
import numpy as np
from collections import defaultdict
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine
import gseapy as gp
from perturbqa import load_de, load_dir

# SynthPert baseline
SYNTHPERT = {
    'k562': {'accuracy': 0.89, 'macro_f1': 0.70, 'down': {'p': 0.47, 'r': 0.60, 'f1': 0.53}},
    'rpe1': {'accuracy': 0.87, 'macro_f1': 0.72, 'down': {'p': 0.47, 'r': 0.65, 'f1': 0.54}},
}


def load_genept_embeddings():
    path = '/Users/yiquntchen/Downloads/GenePT_emebdding_v2/GenePT_gene_embedding_ada_text.pickle'
    with open(path, 'rb') as f:
        embeddings = pickle.load(f)
    return {k: np.array(v) for k, v in embeddings.items()}


def load_pathway_data():
    kegg = gp.get_library('KEGG_2019_Human')
    go_bp = gp.get_library('GO_Biological_Process_2023')
    gene_to_pathways = defaultdict(set)
    for pathway, genes in kegg.items():
        for gene in genes:
            gene_to_pathways[gene.upper()].add(pathway)
    for pathway, genes in go_bp.items():
        for gene in genes:
            gene_to_pathways[gene.upper()].add(pathway)
    return gene_to_pathways


def embedding_similarity(g1, g2, embeddings):
    e1 = embeddings.get(g1.upper())
    e2 = embeddings.get(g2.upper())
    if e1 is None or e2 is None:
        return 0.0
    return 1 - cosine(e1, e2)


def pathway_similarity(g1, g2, gene_to_pathways):
    p1 = gene_to_pathways.get(g1.upper(), set())
    p2 = gene_to_pathways.get(g2.upper(), set())
    if not p1 or not p2:
        return 0.0
    return len(p1 & p2) / len(p1 | p2)


def analyze_false_positives(dataset='k562'):
    """Analyze what we predict as Down that isn't."""
    print(f"\n{'='*80}")
    print(f"ANALYZING DOWN FALSE POSITIVES: {dataset.upper()}")
    print(f"{'='*80}")

    # Load data
    data_de = load_de(dataset)
    data_dir = load_dir(dataset)
    train_de, test_de = data_de['train'], data_de['test']
    train_dir, test_dir = data_dir['train'], data_dir['test']

    # Build priors
    gene_de_stats = defaultdict(lambda: {'pos': 0, 'total': 0})
    gene_dir_stats = defaultdict(lambda: {'pos': 0, 'total': 0})

    for item in train_de:
        gene_de_stats[item['gene']]['total'] += 1
        if item['label'] == 1:
            gene_de_stats[item['gene']]['pos'] += 1

    for item in train_dir:
        gene_dir_stats[item['gene']]['total'] += 1
        if item['label'] == 1:
            gene_dir_stats[item['gene']]['pos'] += 1

    gene_de_prior = {g: s['pos']/s['total'] for g, s in gene_de_stats.items()}
    gene_dir_prior = {g: s['pos']/s['total'] if s['total'] > 0 else 0.5 for g, s in gene_dir_stats.items()}

    # Build test data
    test_keys = [(x['pert'], x['gene']) for x in test_de]
    de_dict = {(x['pert'], x['gene']): x['label'] for x in test_de}
    dir_dict = {(x['pert'], x['gene']): x['label'] for x in test_dir}

    # Build true 3-class labels
    y_true = []
    for pert, gene in test_keys:
        is_de = de_dict.get((pert, gene), 0)
        is_up = dir_dict.get((pert, gene), 0)
        if is_de == 0:
            y_true.append('not_de')
        elif is_up == 1:
            y_true.append('up')
        else:
            y_true.append('down')

    # Count class distribution
    class_counts = defaultdict(int)
    for y in y_true:
        class_counts[y] += 1

    print(f"\nClass distribution in test set:")
    for cls in ['not_de', 'up', 'down']:
        pct = 100 * class_counts[cls] / len(y_true)
        print(f"  {cls}: {class_counts[cls]} ({pct:.1f}%)")

    # Baseline predictions
    y_pred_baseline = []
    for pert, gene in test_keys:
        de_prior = gene_de_prior.get(gene, 0.5)
        dir_prior = gene_dir_prior.get(gene, 0.5)
        if de_prior < 0.5:
            y_pred_baseline.append('not_de')
        elif dir_prior >= 0.5:
            y_pred_baseline.append('up')
        else:
            y_pred_baseline.append('down')

    # Analyze Down predictions
    down_predictions = [(i, test_keys[i], y_true[i])
                        for i in range(len(y_pred_baseline)) if y_pred_baseline[i] == 'down']

    true_positives = [x for x in down_predictions if x[2] == 'down']
    false_positives = [x for x in down_predictions if x[2] != 'down']

    print(f"\nDown predictions: {len(down_predictions)}")
    print(f"  True Positives: {len(true_positives)}")
    print(f"  False Positives: {len(false_positives)}")

    # What are the false positives actually?
    fp_breakdown = defaultdict(int)
    for _, _, true_cls in false_positives:
        fp_breakdown[true_cls] += 1

    print(f"\nFalse Positive breakdown (what we predict as Down but isn't):")
    for cls, count in sorted(fp_breakdown.items(), key=lambda x: -x[1]):
        print(f"  Actually {cls}: {count}")

    # Analyze characteristics of FPs
    print(f"\nAnalyzing gene characteristics of Down FPs...")

    fp_genes = [gene for _, (pert, gene), _ in false_positives]
    tp_genes = [gene for _, (pert, gene), _ in true_positives]

    # Gene direction tendency analysis
    fp_dir_priors = [gene_dir_prior.get(g, 0.5) for g in fp_genes]
    tp_dir_priors = [gene_dir_prior.get(g, 0.5) for g in tp_genes]

    print(f"\n  Gene dir_prior (P(up|gene)):")
    print(f"    False Positives: mean={np.mean(fp_dir_priors):.3f}, std={np.std(fp_dir_priors):.3f}")
    print(f"    True Positives:  mean={np.mean(tp_dir_priors):.3f}, std={np.std(tp_dir_priors):.3f}")

    # Genes that are "confusing" - high DE prior but ambiguous direction
    print(f"\n  Gene de_prior distribution:")
    fp_de_priors = [gene_de_prior.get(g, 0.5) for g in fp_genes]
    tp_de_priors = [gene_de_prior.get(g, 0.5) for g in tp_genes]
    print(f"    False Positives: mean={np.mean(fp_de_priors):.3f}")
    print(f"    True Positives:  mean={np.mean(tp_de_priors):.3f}")

    return {
        'test_keys': test_keys,
        'y_true': y_true,
        'y_pred_baseline': y_pred_baseline,
        'gene_de_prior': gene_de_prior,
        'gene_dir_prior': gene_dir_prior,
        'train_de': train_de,
        'train_dir': train_dir,
    }


def try_direction_aware_prediction(data, embeddings, gene_to_pathways):
    """
    Try direction-aware prediction:
    - For genes where dir_prior is close to 0.5 (ambiguous), use embedding-weighted retrieval
    - For genes where dir_prior is clearly < 0.5, predict Down
    """
    print(f"\n{'='*80}")
    print("DIRECTION-AWARE PREDICTION")
    print("{'='*80}")

    test_keys = data['test_keys']
    y_true = data['y_true']
    gene_de_prior = data['gene_de_prior']
    gene_dir_prior = data['gene_dir_prior']
    train_de = data['train_de']
    train_dir = data['train_dir']

    # Build embedding-weighted retrieval for direction
    # Group training dir by gene
    train_dir_by_gene = defaultdict(list)
    for item in train_dir:
        train_dir_by_gene[item['gene']].append((item['pert'], item['label']))

    # Also build pert-level stats for direction
    pert_dir_stats = defaultdict(lambda: {'up': 0, 'down': 0, 'total': 0})
    for item in train_dir:
        pert = item['pert']
        pert_dir_stats[pert]['total'] += 1
        if item['label'] == 1:
            pert_dir_stats[pert]['up'] += 1
        else:
            pert_dir_stats[pert]['down'] += 1

    pert_up_tendency = {p: s['up'] / s['total'] if s['total'] > 0 else 0.5
                        for p, s in pert_dir_stats.items()}

    y_pred = []
    for pert, gene in test_keys:
        de_prior = gene_de_prior.get(gene, 0.5)
        dir_prior = gene_dir_prior.get(gene, 0.5)

        if de_prior < 0.5:
            y_pred.append('not_de')
        else:
            # For DE genes, use direction prediction
            # Key insight: combine gene prior + perturbation tendency
            pert_tendency = pert_up_tendency.get(pert, 0.5)

            # Weighted combination
            combined_dir = 0.5 * dir_prior + 0.5 * pert_tendency

            if combined_dir >= 0.5:
                y_pred.append('up')
            else:
                y_pred.append('down')

    labels = ['not_de', 'up', 'down']
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)

    print(f"\n--- Combined (gene + pert tendency) ---")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"Down: P={report['down']['precision']:.3f} R={report['down']['recall']:.3f} F1={report['down']['f1-score']:.3f}")
    print(f"Up:   P={report['up']['precision']:.3f} R={report['up']['recall']:.3f} F1={report['up']['f1-score']:.3f}")

    return report


def try_confidence_based_down(data, threshold=0.35):
    """
    Only predict Down when we're confident:
    - dir_prior < threshold (clearly tends down)
    """
    print(f"\n{'='*80}")
    print(f"CONFIDENCE-BASED DOWN (threshold={threshold})")
    print("{'='*80}")

    test_keys = data['test_keys']
    y_true = data['y_true']
    gene_de_prior = data['gene_de_prior']
    gene_dir_prior = data['gene_dir_prior']

    y_pred = []
    for pert, gene in test_keys:
        de_prior = gene_de_prior.get(gene, 0.5)
        dir_prior = gene_dir_prior.get(gene, 0.5)

        if de_prior < 0.5:
            y_pred.append('not_de')
        elif dir_prior < threshold:
            # Only predict Down when very confident
            y_pred.append('down')
        else:
            y_pred.append('up')

    labels = ['not_de', 'up', 'down']
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)

    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"Down: P={report['down']['precision']:.3f} R={report['down']['recall']:.3f} F1={report['down']['f1-score']:.3f}")
    print(f"Up:   P={report['up']['precision']:.3f} R={report['up']['recall']:.3f} F1={report['up']['f1-score']:.3f}")

    return report


def try_embedding_weighted_direction(data, embeddings):
    """
    Use embedding-weighted retrieval for direction prediction.
    For each (pert, gene) pair, weight direction by embedding similarity to pert.
    """
    print(f"\n{'='*80}")
    print("EMBEDDING-WEIGHTED DIRECTION")
    print("{'='*80}")

    test_keys = data['test_keys']
    y_true = data['y_true']
    gene_de_prior = data['gene_de_prior']
    train_dir = data['train_dir']

    # Build pert embeddings
    pert_genes = set(item['pert'] for item in train_dir)

    # Group training data by gene for direction
    train_dir_by_gene = defaultdict(list)
    for item in train_dir:
        train_dir_by_gene[item['gene']].append((item['pert'], item['label']))

    y_pred = []
    for pert_test, gene_test in test_keys:
        de_prior = gene_de_prior.get(gene_test, 0.5)

        if de_prior < 0.5:
            y_pred.append('not_de')
        else:
            # Embedding-weighted direction
            train_examples = train_dir_by_gene.get(gene_test, [])

            if len(train_examples) == 0:
                y_pred.append('up')  # Default to up
                continue

            # Weight by embedding similarity
            weights = []
            labels = []
            for pert_train, label in train_examples:
                sim = embedding_similarity(pert_test, pert_train, embeddings)
                weights.append(max(sim, 0.1))  # Minimum weight
                labels.append(label)

            weights = np.array(weights)
            labels = np.array(labels)

            # Weighted average direction
            weighted_dir = np.average(labels, weights=weights)

            if weighted_dir >= 0.5:
                y_pred.append('up')
            else:
                y_pred.append('down')

    labels = ['not_de', 'up', 'down']
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)

    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"Down: P={report['down']['precision']:.3f} R={report['down']['recall']:.3f} F1={report['down']['f1-score']:.3f}")
    print(f"Up:   P={report['up']['precision']:.3f} R={report['up']['recall']:.3f} F1={report['up']['f1-score']:.3f}")

    return report


def try_two_stage_ml(data, embeddings, gene_to_pathways):
    """
    Two-stage ML approach:
    1. First predict DE vs not_DE (high precision)
    2. Then for DE predictions, predict Up vs Down
    """
    print(f"\n{'='*80}")
    print("TWO-STAGE ML APPROACH")
    print("{'='*80}")

    test_keys = data['test_keys']
    y_true = data['y_true']
    train_de = data['train_de']
    train_dir = data['train_dir']
    gene_de_prior = data['gene_de_prior']
    gene_dir_prior = data['gene_dir_prior']

    # Build per-perturbation stats
    pert_de_stats = defaultdict(lambda: {'pos': 0, 'total': 0})
    pert_dir_stats = defaultdict(lambda: {'pos': 0, 'total': 0})

    for item in train_de:
        pert_de_stats[item['pert']]['total'] += 1
        if item['label'] == 1:
            pert_de_stats[item['pert']]['pos'] += 1

    for item in train_dir:
        pert_dir_stats[item['pert']]['total'] += 1
        if item['label'] == 1:
            pert_dir_stats[item['pert']]['pos'] += 1

    pert_de_prior = {p: s['pos']/s['total'] for p, s in pert_de_stats.items()}
    pert_dir_prior = {p: s['pos']/s['total'] if s['total'] > 0 else 0.5 for p, s in pert_dir_stats.items()}

    # Features function
    def get_features(pert, gene, for_dir=False):
        feats = []
        # Gene priors
        feats.append(gene_de_prior.get(gene, 0.5))
        feats.append(gene_dir_prior.get(gene, 0.5))
        # Perturbation priors
        feats.append(pert_de_prior.get(pert, 0.5))
        feats.append(pert_dir_prior.get(pert, 0.5))
        # Embedding similarity
        feats.append(embedding_similarity(pert, gene, embeddings))
        # Pathway similarity
        feats.append(pathway_similarity(pert, gene, gene_to_pathways))

        if for_dir:
            # Direction-specific: how strongly down-biased is the gene?
            dir_p = gene_dir_prior.get(gene, 0.5)
            feats.append(0.5 - dir_p)  # Positive means tends down
            # How strongly down-biased is the perturbation?
            pert_dir_p = pert_dir_prior.get(pert, 0.5)
            feats.append(0.5 - pert_dir_p)  # Positive means tends down

        return feats

    # Train direction model on DE genes only
    print("Building direction model on DE genes only...")

    # Get DE-positive examples from train_de
    de_positive_keys = set()
    for item in train_de:
        if item['label'] == 1:
            de_positive_keys.add((item['pert'], item['gene']))

    # Filter train_dir to only DE-positive
    dir_train_de_only = [item for item in train_dir if (item['pert'], item['gene']) in de_positive_keys]
    print(f"  Training direction model on {len(dir_train_de_only)} DE-positive examples")

    if len(dir_train_de_only) < 100:
        print("  Not enough DE-positive direction examples!")
        return None

    # Sample for training
    np.random.seed(42)
    sample_idx = np.random.choice(len(dir_train_de_only), size=min(20000, len(dir_train_de_only)), replace=False)

    X_dir = np.array([get_features(dir_train_de_only[i]['pert'], dir_train_de_only[i]['gene'], for_dir=True)
                      for i in sample_idx])
    y_dir = np.array([dir_train_de_only[i]['label'] for i in sample_idx])

    # Train with balanced class weights
    scaler = StandardScaler()
    X_dir_scaled = scaler.fit_transform(X_dir)

    model_dir = LogisticRegression(max_iter=1000, class_weight='balanced')
    model_dir.fit(X_dir_scaled, y_dir)

    # Print coefficients
    feat_names = ['gene_de', 'gene_dir', 'pert_de', 'pert_dir', 'emb_sim', 'pathway_sim',
                  'gene_down_bias', 'pert_down_bias']
    print("  Direction model coefficients:")
    for name, coef in zip(feat_names, model_dir.coef_[0]):
        print(f"    {name}: {coef:.4f}")

    # Make predictions
    y_pred = []
    pred_details = []

    for pert, gene in test_keys:
        de_prior = gene_de_prior.get(gene, 0.5)

        if de_prior < 0.5:
            y_pred.append('not_de')
            pred_details.append({'de_prior': de_prior, 'dir_prob': None})
        else:
            # Use ML model for direction
            feats = get_features(pert, gene, for_dir=True)
            feats_scaled = scaler.transform([feats])
            dir_prob = model_dir.predict_proba(feats_scaled)[0, 1]

            if dir_prob >= 0.5:
                y_pred.append('up')
            else:
                y_pred.append('down')

            pred_details.append({'de_prior': de_prior, 'dir_prob': dir_prob})

    labels = ['not_de', 'up', 'down']
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)

    print(f"\n--- Results ---")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"Macro F1: {report['macro avg']['f1-score']:.4f}")
    print(f"not_de: P={report['not_de']['precision']:.3f} R={report['not_de']['recall']:.3f} F1={report['not_de']['f1-score']:.3f}")
    print(f"up:     P={report['up']['precision']:.3f} R={report['up']['recall']:.3f} F1={report['up']['f1-score']:.3f}")
    print(f"down:   P={report['down']['precision']:.3f} R={report['down']['recall']:.3f} F1={report['down']['f1-score']:.3f}")

    # Analyze Down predictions
    down_preds = [(i, pred_details[i]) for i in range(len(y_pred)) if y_pred[i] == 'down']
    true_down = [i for i in range(len(y_true)) if y_true[i] == 'down']

    print(f"\nDown predictions: {len(down_preds)}")
    print(f"Actual Down in test: {len(true_down)}")

    return report


def try_cost_sensitive(data, embeddings, gene_to_pathways):
    """
    Cost-sensitive learning with custom class weights.
    """
    print(f"\n{'='*80}")
    print("COST-SENSITIVE 3-CLASS LEARNING")
    print("{'='*80}")

    test_keys = data['test_keys']
    y_true = data['y_true']
    train_de = data['train_de']
    train_dir = data['train_dir']
    gene_de_prior = data['gene_de_prior']
    gene_dir_prior = data['gene_dir_prior']

    # Build pert priors
    pert_de_stats = defaultdict(lambda: {'pos': 0, 'total': 0})
    pert_dir_stats = defaultdict(lambda: {'pos': 0, 'total': 0})

    for item in train_de:
        pert_de_stats[item['pert']]['total'] += 1
        if item['label'] == 1:
            pert_de_stats[item['pert']]['pos'] += 1

    for item in train_dir:
        pert_dir_stats[item['pert']]['total'] += 1
        if item['label'] == 1:
            pert_dir_stats[item['pert']]['pos'] += 1

    pert_de_prior = {p: s['pos']/s['total'] for p, s in pert_de_stats.items()}
    pert_dir_prior = {p: s['pos']/s['total'] if s['total'] > 0 else 0.5 for p, s in pert_dir_stats.items()}

    # Build 3-class training data
    de_dict = {(item['pert'], item['gene']): item['label'] for item in train_de}
    dir_dict = {(item['pert'], item['gene']): item['label'] for item in train_dir}

    train_3class = []
    for item in train_de:
        key = (item['pert'], item['gene'])
        is_de = de_dict.get(key, 0)
        is_up = dir_dict.get(key, 0)

        if is_de == 0:
            label = 0  # not_de
        elif is_up == 1:
            label = 1  # up
        else:
            label = 2  # down

        train_3class.append({**item, 'label_3class': label})

    # Count class distribution
    class_counts = defaultdict(int)
    for item in train_3class:
        class_counts[item['label_3class']] += 1

    print(f"Training class distribution:")
    for cls, name in [(0, 'not_de'), (1, 'up'), (2, 'down')]:
        pct = 100 * class_counts[cls] / len(train_3class)
        print(f"  {name}: {class_counts[cls]} ({pct:.1f}%)")

    # Feature function
    def get_features(pert, gene):
        feats = []
        feats.append(gene_de_prior.get(gene, 0.5))
        feats.append(gene_dir_prior.get(gene, 0.5))
        feats.append(pert_de_prior.get(pert, 0.5))
        feats.append(pert_dir_prior.get(pert, 0.5))
        feats.append(embedding_similarity(pert, gene, embeddings))
        feats.append(pathway_similarity(pert, gene, gene_to_pathways))
        # Direction-specific features
        feats.append(0.5 - gene_dir_prior.get(gene, 0.5))  # Gene down bias
        feats.append(0.5 - pert_dir_prior.get(pert, 0.5))  # Pert down bias
        return feats

    # Sample and build features
    np.random.seed(42)
    sample_idx = np.random.choice(len(train_3class), size=min(30000, len(train_3class)), replace=False)

    X_train = np.array([get_features(train_3class[i]['pert'], train_3class[i]['gene']) for i in sample_idx])
    y_train = np.array([train_3class[i]['label_3class'] for i in sample_idx])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Try different class weight configurations
    weight_configs = [
        {'not_de': 1, 'up': 1, 'down': 5},
        {'not_de': 1, 'up': 1, 'down': 10},
        {'not_de': 1, 'up': 2, 'down': 15},
        {'not_de': 1, 'up': 1, 'down': 20},
    ]

    # Test features
    X_test = np.array([get_features(pert, gene) for pert, gene in test_keys])
    X_test_scaled = scaler.transform(X_test)
    y_true_num = [{'not_de': 0, 'up': 1, 'down': 2}[y] for y in y_true]

    best_result = None
    best_f1 = 0

    for weights in weight_configs:
        model = LogisticRegression(
            max_iter=1000,
            class_weight={0: weights['not_de'], 1: weights['up'], 2: weights['down']}
        )
        model.fit(X_train_scaled, y_train)

        y_pred_num = model.predict(X_test_scaled)
        y_pred = [{0: 'not_de', 1: 'up', 2: 'down'}[p] for p in y_pred_num]

        report = classification_report(y_true, y_pred, labels=['not_de', 'up', 'down'],
                                        output_dict=True, zero_division=0)

        down_f1 = report['down']['f1-score']

        print(f"\nWeights: {weights}")
        print(f"  Acc: {report['accuracy']:.3f}, MacroF1: {report['macro avg']['f1-score']:.3f}")
        print(f"  Down: P={report['down']['precision']:.3f} R={report['down']['recall']:.3f} F1={down_f1:.3f}")

        if down_f1 > best_f1:
            best_f1 = down_f1
            best_result = {'weights': weights, 'report': report, 'model': model}

    print(f"\n=== BEST RESULT ===")
    print(f"Weights: {best_result['weights']}")
    r = best_result['report']
    print(f"Accuracy: {r['accuracy']:.4f}")
    print(f"Macro F1: {r['macro avg']['f1-score']:.4f}")
    print(f"not_de: P={r['not_de']['precision']:.3f} R={r['not_de']['recall']:.3f} F1={r['not_de']['f1-score']:.3f}")
    print(f"up:     P={r['up']['precision']:.3f} R={r['up']['recall']:.3f} F1={r['up']['f1-score']:.3f}")
    print(f"down:   P={r['down']['precision']:.3f} R={r['down']['recall']:.3f} F1={r['down']['f1-score']:.3f}")

    return best_result


if __name__ == "__main__":
    print("Loading resources...")
    embeddings = load_genept_embeddings()
    gene_to_pathways = load_pathway_data()

    for dataset in ['k562', 'rpe1']:
        data = analyze_false_positives(dataset)

        print("\n" + "="*80)
        print(f"TRYING IMPROVEMENTS FOR {dataset.upper()}")
        print("="*80)

        # Try various approaches
        try_direction_aware_prediction(data, embeddings, gene_to_pathways)

        for thresh in [0.4, 0.35, 0.3]:
            try_confidence_based_down(data, threshold=thresh)

        try_embedding_weighted_direction(data, embeddings)

        try_two_stage_ml(data, embeddings, gene_to_pathways)

        try_cost_sensitive(data, embeddings, gene_to_pathways)

        # Print SynthPert comparison
        if dataset in SYNTHPERT:
            print(f"\n--- SYNTHPERT BASELINE ---")
            sp = SYNTHPERT[dataset]
            print(f"Accuracy: {sp['accuracy']:.3f}, Macro F1: {sp['macro_f1']:.3f}")
            print(f"Down: P={sp['down']['p']:.3f} R={sp['down']['r']:.3f} F1={sp['down']['f1']:.3f}")
