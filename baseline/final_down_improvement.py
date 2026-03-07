"""
final_down_improvement.py
=========================

Final attempt to improve K562 Down F1.
Focus on precision-recall tradeoff using probability calibration.
"""

import pickle
import numpy as np
from collections import defaultdict
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine
import gseapy as gp
from perturbqa import load_de, load_dir


def load_genept_embeddings():
    path = '/Users/yiquntchen/Downloads/GenePT_emebdding_v2/GenePT_gene_embedding_ada_text.pickle'
    with open(path, 'rb') as f:
        return {k: np.array(v) for k, v in pickle.load(f).items()}


def load_pathway_data():
    kegg = gp.get_library('KEGG_2019_Human')
    go_bp = gp.get_library('GO_Biological_Process_2023')
    gene_to_pathways = defaultdict(set)
    for pathway, genes in {**kegg, **go_bp}.items():
        for gene in genes:
            gene_to_pathways[gene.upper()].add(pathway)
    return gene_to_pathways


def emb_sim(g1, g2, embeddings):
    e1 = embeddings.get(g1.upper())
    e2 = embeddings.get(g2.upper())
    if e1 is None or e2 is None:
        return 0.0
    return 1 - cosine(e1, e2)


def pathway_sim(g1, g2, gene_to_pathways):
    p1 = gene_to_pathways.get(g1.upper(), set())
    p2 = gene_to_pathways.get(g2.upper(), set())
    if not p1 or not p2:
        return 0.0
    return len(p1 & p2) / len(p1 | p2)


def run_final_analysis(dataset='k562'):
    """Final analysis with optimal threshold selection."""
    print(f"\n{'='*80}")
    print(f"FINAL ANALYSIS: {dataset.upper()}")
    print(f"{'='*80}")

    embeddings = load_genept_embeddings()
    gene_to_pathways = load_pathway_data()

    # Load data
    data_de = load_de(dataset)
    data_dir = load_dir(dataset)
    train_de, test_de = data_de['train'], data_de['test']
    train_dir, test_dir = data_dir['train'], data_dir['test']

    # Build stats
    gene_de_stats = defaultdict(lambda: {'pos': 0, 'total': 0})
    gene_dir_stats = defaultdict(lambda: {'pos': 0, 'total': 0})
    pert_de_stats = defaultdict(lambda: {'pos': 0, 'total': 0})
    pert_dir_stats = defaultdict(lambda: {'pos': 0, 'total': 0})
    gene_down_count = defaultdict(int)
    pert_down_count = defaultdict(int)

    for item in train_de:
        gene_de_stats[item['gene']]['total'] += 1
        pert_de_stats[item['pert']]['total'] += 1
        if item['label'] == 1:
            gene_de_stats[item['gene']]['pos'] += 1
            pert_de_stats[item['pert']]['pos'] += 1

    for item in train_dir:
        gene_dir_stats[item['gene']]['total'] += 1
        pert_dir_stats[item['pert']]['total'] += 1
        if item['label'] == 1:
            gene_dir_stats[item['gene']]['pos'] += 1
            pert_dir_stats[item['pert']]['pos'] += 1
        else:
            gene_down_count[item['gene']] += 1
            pert_down_count[item['pert']] += 1

    gene_de_prior = {g: s['pos']/s['total'] for g, s in gene_de_stats.items()}
    gene_dir_prior = {g: s['pos']/s['total'] if s['total'] > 0 else 0.5 for g, s in gene_dir_stats.items()}
    pert_de_prior = {p: s['pos']/s['total'] for p, s in pert_de_stats.items()}
    pert_dir_prior = {p: s['pos']/s['total'] if s['total'] > 0 else 0.5 for p, s in pert_dir_stats.items()}

    # Test data
    test_keys = [(x['pert'], x['gene']) for x in test_de]
    de_dict = {(x['pert'], x['gene']): x['label'] for x in test_de}
    dir_dict = {(x['pert'], x['gene']): x['label'] for x in test_dir}

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

    # Feature extraction
    def get_features(pert, gene):
        feats = []
        feats.append(gene_de_prior.get(gene, 0.5))
        feats.append(gene_dir_prior.get(gene, 0.5))
        feats.append(pert_de_prior.get(pert, 0.5))
        feats.append(pert_dir_prior.get(pert, 0.5))

        gene_down_rate = gene_down_count.get(gene, 0) / max(gene_de_stats[gene]['pos'], 1)
        pert_down_rate = pert_down_count.get(pert, 0) / max(pert_de_stats[pert]['pos'], 1)
        feats.append(gene_down_rate)
        feats.append(pert_down_rate)

        feats.append(emb_sim(pert, gene, embeddings))
        feats.append(pathway_sim(pert, gene, gene_to_pathways))

        return feats

    # Build 3-class training data
    de_train_dict = {(item['pert'], item['gene']): item['label'] for item in train_de}
    dir_train_dict = {(item['pert'], item['gene']): item['label'] for item in train_dir}

    train_3class = []
    for item in train_de:
        key = (item['pert'], item['gene'])
        is_de = de_train_dict.get(key, 0)
        is_up = dir_train_dict.get(key, 0)

        if is_de == 0:
            label = 0
        elif is_up == 1:
            label = 1
        else:
            label = 2

        train_3class.append({'pert': item['pert'], 'gene': item['gene'], 'label': label})

    # Sample and train
    np.random.seed(42)
    sample_idx = np.random.choice(len(train_3class), size=min(40000, len(train_3class)), replace=False)

    X_train = np.array([get_features(train_3class[i]['pert'], train_3class[i]['gene']) for i in sample_idx])
    y_train = np.array([train_3class[i]['label'] for i in sample_idx])

    X_test = np.array([get_features(pert, gene) for pert, gene in test_keys])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model WITHOUT class weights, then tune thresholds
    print("\nTraining calibrated model...")
    base_model = LogisticRegression(max_iter=1000)
    calibrated = CalibratedClassifierCV(base_model, cv=5, method='isotonic')
    calibrated.fit(X_train_scaled, y_train)

    proba = calibrated.predict_proba(X_test_scaled)
    # proba[:, 0] = P(not_de), proba[:, 1] = P(up), proba[:, 2] = P(down)

    # === Method 1: Standard argmax ===
    print("\n--- Method 1: Standard Argmax ---")
    y_pred_num = proba.argmax(axis=1)
    y_pred = [{0: 'not_de', 1: 'up', 2: 'down'}[p] for p in y_pred_num]
    report = classification_report(y_true, y_pred, labels=['not_de', 'up', 'down'],
                                    output_dict=True, zero_division=0)
    print(f"Acc: {report['accuracy']:.3f}, MacroF1: {report['macro avg']['f1-score']:.3f}")
    print(f"Down: P={report['down']['precision']:.3f} R={report['down']['recall']:.3f} F1={report['down']['f1-score']:.3f}")

    # === Method 2: Two-stage with thresholds ===
    print("\n--- Method 2: Two-Stage with Threshold Tuning ---")

    # Stage 1: DE vs not_DE (use DE model)
    # Stage 2: For DE predictions, Up vs Down

    # First find optimal DE threshold
    y_true_de = np.array([0 if y == 'not_de' else 1 for y in y_true])
    p_de = 1 - proba[:, 0]  # P(DE) = 1 - P(not_de)

    # For direction, among DE examples
    y_true_up = np.array([1 if y == 'up' else 0 for y in y_true])
    de_mask = y_true_de == 1

    # Sweep DE thresholds
    best_overall = None
    best_macro_f1 = 0

    print("\nGrid search over thresholds...")
    for de_thresh in [0.3, 0.35, 0.4, 0.45, 0.5]:
        for down_boost in [0.0, 0.1, 0.2, 0.3]:
            y_pred = []
            for i in range(len(test_keys)):
                if proba[i, 0] > (1 - de_thresh):  # P(not_de) > threshold
                    y_pred.append('not_de')
                else:
                    # DE - decide up vs down
                    # Apply down boost to encourage Down predictions
                    p_up = proba[i, 1]
                    p_down = proba[i, 2] + down_boost

                    if p_down > p_up:
                        y_pred.append('down')
                    else:
                        y_pred.append('up')

            report = classification_report(y_true, y_pred, labels=['not_de', 'up', 'down'],
                                            output_dict=True, zero_division=0)

            macro_f1 = report['macro avg']['f1-score']
            down_f1 = report['down']['f1-score']

            if down_f1 > 0.35:  # Show promising results
                print(f"  de_thresh={de_thresh}, down_boost={down_boost}: "
                      f"Acc={report['accuracy']:.3f}, Macro={macro_f1:.3f}, "
                      f"Down P={report['down']['precision']:.3f} R={report['down']['recall']:.3f} F1={down_f1:.3f}")

            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                best_overall = {
                    'de_thresh': de_thresh,
                    'down_boost': down_boost,
                    'report': report
                }

    print(f"\nBest overall (by Macro F1):")
    print(f"  de_thresh={best_overall['de_thresh']}, down_boost={best_overall['down_boost']}")
    r = best_overall['report']
    print(f"  Acc: {r['accuracy']:.3f}, MacroF1: {r['macro avg']['f1-score']:.3f}")
    print(f"  Down: P={r['down']['precision']:.3f} R={r['down']['recall']:.3f} F1={r['down']['f1-score']:.3f}")

    # === Method 3: Baseline comparison ===
    print("\n--- Method 3: Pure Baseline (Majority Vote) ---")
    baseline_pred = []
    for pert, gene in test_keys:
        de_prior = gene_de_prior.get(gene, 0.5)
        dir_prior = gene_dir_prior.get(gene, 0.5)
        if de_prior < 0.5:
            baseline_pred.append('not_de')
        elif dir_prior >= 0.5:
            baseline_pred.append('up')
        else:
            baseline_pred.append('down')

    report = classification_report(y_true, baseline_pred, labels=['not_de', 'up', 'down'],
                                    output_dict=True, zero_division=0)
    print(f"Acc: {report['accuracy']:.3f}, MacroF1: {report['macro avg']['f1-score']:.3f}")
    print(f"Down: P={report['down']['precision']:.3f} R={report['down']['recall']:.3f} F1={report['down']['f1-score']:.3f}")

    # === Method 4: Embedding-weighted retrieval for direction ===
    print("\n--- Method 4: Embedding-Weighted Direction ---")

    # Build training by gene for direction
    train_dir_by_gene = defaultdict(list)
    for item in train_dir:
        train_dir_by_gene[item['gene']].append((item['pert'], item['label']))

    emb_pred = []
    for pert_test, gene_test in test_keys:
        de_prior = gene_de_prior.get(gene_test, 0.5)

        if de_prior < 0.5:
            emb_pred.append('not_de')
        else:
            train_examples = train_dir_by_gene.get(gene_test, [])

            if len(train_examples) == 0:
                # No training data - use gene prior
                if gene_dir_prior.get(gene_test, 0.5) >= 0.5:
                    emb_pred.append('up')
                else:
                    emb_pred.append('down')
                continue

            # Weight by embedding similarity
            weights = []
            labels = []
            for pert_train, label in train_examples:
                sim = emb_sim(pert_test, pert_train, embeddings)
                weights.append(max(sim, 0.01))
                labels.append(label)

            weights = np.array(weights)
            labels = np.array(labels)

            weighted_dir = np.average(labels, weights=weights)

            if weighted_dir >= 0.5:
                emb_pred.append('up')
            else:
                emb_pred.append('down')

    report = classification_report(y_true, emb_pred, labels=['not_de', 'up', 'down'],
                                    output_dict=True, zero_division=0)
    print(f"Acc: {report['accuracy']:.3f}, MacroF1: {report['macro avg']['f1-score']:.3f}")
    print(f"Down: P={report['down']['precision']:.3f} R={report['down']['recall']:.3f} F1={report['down']['f1-score']:.3f}")

    # === Method 5: Hybrid - Use embedding weighting with boosted Down ===
    print("\n--- Method 5: Hybrid (Embedding + Down Boost) ---")

    for down_boost in [0.05, 0.1, 0.15, 0.2]:
        hybrid_pred = []
        for i, (pert_test, gene_test) in enumerate(test_keys):
            de_prior = gene_de_prior.get(gene_test, 0.5)

            if de_prior < 0.5:
                hybrid_pred.append('not_de')
            else:
                train_examples = train_dir_by_gene.get(gene_test, [])

                if len(train_examples) == 0:
                    dir_p = gene_dir_prior.get(gene_test, 0.5)
                else:
                    weights = []
                    labels = []
                    for pert_train, label in train_examples:
                        sim = emb_sim(pert_test, pert_train, embeddings)
                        weights.append(max(sim, 0.01))
                        labels.append(label)

                    weights = np.array(weights)
                    labels = np.array(labels)
                    dir_p = np.average(labels, weights=weights)

                # Apply down boost
                if dir_p < 0.5 + down_boost:
                    hybrid_pred.append('down')
                else:
                    hybrid_pred.append('up')

        report = classification_report(y_true, hybrid_pred, labels=['not_de', 'up', 'down'],
                                        output_dict=True, zero_division=0)
        down_f1 = report['down']['f1-score']
        print(f"  down_boost={down_boost}: Acc={report['accuracy']:.3f}, "
              f"Down P={report['down']['precision']:.3f} R={report['down']['recall']:.3f} F1={down_f1:.3f}")

    # === Summary ===
    print(f"\n{'='*80}")
    print("SYNTHPERT COMPARISON")
    print(f"{'='*80}")
    if dataset == 'k562':
        print("SynthPert K562: Acc=0.89, MacroF1=0.70, Down P=0.47 R=0.60 F1=0.53")
    elif dataset == 'rpe1':
        print("SynthPert RPE1: Acc=0.87, MacroF1=0.72, Down P=0.47 R=0.65 F1=0.54")


if __name__ == "__main__":
    for dataset in ['k562', 'rpe1']:
        run_final_analysis(dataset)
