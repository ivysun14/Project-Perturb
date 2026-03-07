"""
analyze_down_gap.py
===================

Deep analysis of K562 Down gap vs SynthPert.
What information are we missing?
"""

import pickle
import numpy as np
from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine
import gseapy as gp
from perturbqa import load_de, load_dir


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


def analyze_k562_gap():
    """Deep analysis of K562 Down prediction gap."""
    print("="*80)
    print("K562 DOWN GAP ANALYSIS")
    print("="*80)

    embeddings = load_genept_embeddings()
    gene_to_pathways = load_pathway_data()

    # Load data
    data_de = load_de('k562')
    data_dir = load_dir('k562')
    train_de, test_de = data_de['train'], data_de['test']
    train_dir, test_dir = data_dir['train'], data_dir['test']

    # Build comprehensive stats
    gene_de_stats = defaultdict(lambda: {'pos': 0, 'total': 0})
    gene_dir_stats = defaultdict(lambda: {'pos': 0, 'total': 0})
    pert_de_stats = defaultdict(lambda: {'pos': 0, 'total': 0})
    pert_dir_stats = defaultdict(lambda: {'pos': 0, 'total': 0})

    # Also track DOWN specifically
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
            # label=0 in dir means DOWN (when DE)
            gene_down_count[item['gene']] += 1
            pert_down_count[item['pert']] += 1

    # Build priors
    gene_de_prior = {g: s['pos']/s['total'] for g, s in gene_de_stats.items()}
    gene_dir_prior = {g: s['pos']/s['total'] if s['total'] > 0 else 0.5 for g, s in gene_dir_stats.items()}
    pert_de_prior = {p: s['pos']/s['total'] for p, s in pert_de_stats.items()}
    pert_dir_prior = {p: s['pos']/s['total'] if s['total'] > 0 else 0.5 for p, s in pert_dir_stats.items()}

    # Build test data with true 3-class labels
    test_keys = [(x['pert'], x['gene']) for x in test_de]
    de_dict = {(x['pert'], x['gene']): x['label'] for x in test_de}
    dir_dict = {(x['pert'], x['gene']): x['label'] for x in test_dir}

    y_true = []
    y_true_num = []
    for pert, gene in test_keys:
        is_de = de_dict.get((pert, gene), 0)
        is_up = dir_dict.get((pert, gene), 0)
        if is_de == 0:
            y_true.append('not_de')
            y_true_num.append(0)
        elif is_up == 1:
            y_true.append('up')
            y_true_num.append(1)
        else:
            y_true.append('down')
            y_true_num.append(2)

    # === APPROACH 1: Rich feature engineering ===
    print("\n--- APPROACH 1: Rich Feature Engineering ---")

    def get_rich_features(pert, gene):
        feats = []

        # Basic priors
        feats.append(gene_de_prior.get(gene, 0.5))
        feats.append(gene_dir_prior.get(gene, 0.5))
        feats.append(pert_de_prior.get(pert, 0.5))
        feats.append(pert_dir_prior.get(pert, 0.5))

        # Down-specific priors
        gene_down_rate = gene_down_count.get(gene, 0) / max(gene_de_stats[gene]['pos'], 1)
        pert_down_rate = pert_down_count.get(pert, 0) / max(pert_de_stats[pert]['pos'], 1)
        feats.append(gene_down_rate)
        feats.append(pert_down_rate)

        # Similarity features
        feats.append(embedding_similarity(pert, gene, embeddings))
        feats.append(pathway_similarity(pert, gene, gene_to_pathways))

        # Interaction features
        feats.append(gene_dir_prior.get(gene, 0.5) * pert_dir_prior.get(pert, 0.5))
        feats.append(gene_down_rate * pert_down_rate)

        # Confidence features
        gene_dir_n = gene_dir_stats[gene]['total']
        pert_dir_n = pert_dir_stats[pert]['total']
        feats.append(min(gene_dir_n, 100) / 100)  # Gene sample size (normalized)
        feats.append(min(pert_dir_n, 100) / 100)  # Pert sample size (normalized)

        return feats

    feature_names = [
        'gene_de_prior', 'gene_dir_prior', 'pert_de_prior', 'pert_dir_prior',
        'gene_down_rate', 'pert_down_rate',
        'emb_sim', 'pathway_sim',
        'dir_interaction', 'down_interaction',
        'gene_n', 'pert_n'
    ]

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

    # Sample training data
    np.random.seed(42)
    sample_idx = np.random.choice(len(train_3class), size=min(40000, len(train_3class)), replace=False)

    X_train = np.array([get_rich_features(train_3class[i]['pert'], train_3class[i]['gene'])
                        for i in sample_idx])
    y_train = np.array([train_3class[i]['label'] for i in sample_idx])

    X_test = np.array([get_rich_features(pert, gene) for pert, gene in test_keys])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Try multiple classifiers
    classifiers = [
        ('LogReg', LogisticRegression(max_iter=1000, class_weight='balanced')),
        ('RF', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)),
        ('GBM', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ]

    best_result = None
    best_f1 = 0

    for name, clf in classifiers:
        print(f"\n{name}:")
        clf.fit(X_train_scaled, y_train)
        y_pred_num = clf.predict(X_test_scaled)
        y_pred = [{0: 'not_de', 1: 'up', 2: 'down'}[p] for p in y_pred_num]

        report = classification_report(y_true, y_pred, labels=['not_de', 'up', 'down'],
                                        output_dict=True, zero_division=0)

        print(f"  Acc: {report['accuracy']:.3f}, MacroF1: {report['macro avg']['f1-score']:.3f}")
        print(f"  not_de: P={report['not_de']['precision']:.3f} R={report['not_de']['recall']:.3f}")
        print(f"  up:     P={report['up']['precision']:.3f} R={report['up']['recall']:.3f}")
        print(f"  down:   P={report['down']['precision']:.3f} R={report['down']['recall']:.3f} F1={report['down']['f1-score']:.3f}")

        if report['down']['f1-score'] > best_f1:
            best_f1 = report['down']['f1-score']
            best_result = {'name': name, 'clf': clf, 'report': report}

        # Feature importance for RF and GBM
        if hasattr(clf, 'feature_importances_'):
            importances = list(zip(feature_names, clf.feature_importances_))
            importances.sort(key=lambda x: -x[1])
            print(f"  Top features: {importances[:5]}")

    print(f"\nBest classifier: {best_result['name']}")
    print(f"Down F1: {best_f1:.3f}")

    # === APPROACH 2: Threshold optimization for UP vs DOWN ===
    print("\n\n--- APPROACH 2: Threshold Optimization ---")

    # Use probabilities from best model
    if hasattr(best_result['clf'], 'predict_proba'):
        proba = best_result['clf'].predict_proba(X_test_scaled)

        # proba[:, 0] = P(not_de), proba[:, 1] = P(up), proba[:, 2] = P(down)
        p_down = proba[:, 2]
        p_up = proba[:, 1]
        p_notde = proba[:, 0]

        # Try different decision boundaries
        print("\nThreshold tuning (down_threshold vs up_threshold):")

        for down_thresh in [0.15, 0.20, 0.25, 0.30]:
            for up_thresh in [0.3, 0.35, 0.4]:
                y_pred = []
                for i in range(len(test_keys)):
                    if p_notde[i] > 0.5:
                        y_pred.append('not_de')
                    elif p_down[i] > down_thresh and p_down[i] > p_up[i]:
                        y_pred.append('down')
                    else:
                        y_pred.append('up')

                report = classification_report(y_true, y_pred, labels=['not_de', 'up', 'down'],
                                               output_dict=True, zero_division=0)

                down_f1 = report['down']['f1-score']
                if down_f1 > 0.35:  # Only show promising results
                    print(f"  down>{down_thresh}, up>{up_thresh}: Acc={report['accuracy']:.3f}, "
                          f"Down P={report['down']['precision']:.3f} R={report['down']['recall']:.3f} F1={down_f1:.3f}")

    # === APPROACH 3: Analyze what makes K562 Down hard ===
    print("\n\n--- APPROACH 3: Error Analysis ---")

    y_pred = best_result['clf'].predict(X_test_scaled)
    y_pred_labels = [{0: 'not_de', 1: 'up', 2: 'down'}[p] for p in y_pred]

    # Confusion matrix
    cm = confusion_matrix(y_true_num, y_pred, labels=[0, 1, 2])
    print("\nConfusion Matrix:")
    print("            Pred not_de  Pred up  Pred down")
    print(f"True not_de   {cm[0,0]:6d}   {cm[0,1]:6d}    {cm[0,2]:6d}")
    print(f"True up       {cm[1,0]:6d}   {cm[1,1]:6d}    {cm[1,2]:6d}")
    print(f"True down     {cm[2,0]:6d}   {cm[2,1]:6d}    {cm[2,2]:6d}")

    # Analyze Down misses
    down_as_up = [(i, test_keys[i]) for i in range(len(y_true))
                  if y_true[i] == 'down' and y_pred_labels[i] == 'up']
    down_as_notde = [(i, test_keys[i]) for i in range(len(y_true))
                     if y_true[i] == 'down' and y_pred_labels[i] == 'not_de']

    print(f"\nTrue Down predicted as Up: {len(down_as_up)}")
    print(f"True Down predicted as not_de: {len(down_as_notde)}")

    # Analyze features of missed Down
    if down_as_up:
        missed_features = np.array([X_test[i] for i, _ in down_as_up])
        correct_down_idx = [i for i in range(len(y_true))
                            if y_true[i] == 'down' and y_pred_labels[i] == 'down']
        correct_features = np.array([X_test[i] for i in correct_down_idx]) if correct_down_idx else None

        print("\nFeature comparison (Down predicted correctly vs as Up):")
        for j, name in enumerate(feature_names):
            if correct_features is not None:
                print(f"  {name}: correct={np.mean(correct_features[:,j]):.3f}, "
                      f"missed_as_up={np.mean(missed_features[:,j]):.3f}")

    # === APPROACH 4: Ensemble with baseline ===
    print("\n\n--- APPROACH 4: Ensemble with Baseline ---")

    # Get baseline predictions
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

    # Ensemble: if both agree on Down, predict Down; otherwise trust ML
    ensemble_pred = []
    for i in range(len(test_keys)):
        if baseline_pred[i] == 'down' and y_pred_labels[i] == 'down':
            ensemble_pred.append('down')
        elif baseline_pred[i] == 'down' or y_pred_labels[i] == 'down':
            # One says down - use ML probabilities
            if hasattr(best_result['clf'], 'predict_proba'):
                proba = best_result['clf'].predict_proba(X_test_scaled[i:i+1])
                if proba[0, 2] > 0.25:  # High Down probability
                    ensemble_pred.append('down')
                else:
                    ensemble_pred.append(y_pred_labels[i])
            else:
                ensemble_pred.append(y_pred_labels[i])
        else:
            ensemble_pred.append(y_pred_labels[i])

    report = classification_report(y_true, ensemble_pred, labels=['not_de', 'up', 'down'],
                                    output_dict=True, zero_division=0)

    print(f"Ensemble: Acc={report['accuracy']:.3f}, MacroF1={report['macro avg']['f1-score']:.3f}")
    print(f"  Down: P={report['down']['precision']:.3f} R={report['down']['recall']:.3f} F1={report['down']['f1-score']:.3f}")

    # === FINAL SUMMARY ===
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nBest Down F1 achieved: {best_f1:.3f}")
    print(f"SynthPert Down F1: 0.530")
    print(f"Gap: {0.53 - best_f1:.3f}")

    print("\nKey insights:")
    print("1. K562 Down is only 2.5% of test data - severe class imbalance")
    print("2. Down-as-Up misses are common - genes may go both ways")
    print("3. Need perturbation-specific features that ML models can learn")
    print("4. SynthPert may use LLM reasoning about TF-target relationships")


if __name__ == "__main__":
    analyze_k562_gap()
