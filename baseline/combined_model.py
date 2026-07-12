"""
combined_model.py
=================

Combined prediction model using:
1. Gene prior (majority vote baseline)
2. GenePT embeddings (gene-gene similarity)
3. MSigDB pathways (pathway co-membership)
4. Training co-occurrence

Compare against SynthPert baseline.
"""

import pickle
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine
import gseapy as gp
from perturbqa import load_de, load_dir

# SynthPert reported numbers (from paper Table 2)
# https://arxiv.org/html/2509.25346v1
SYNTHPERT_BASELINE = {
    'k562': {'accuracy': 0.89, 'macro_f1': 0.70, 'up_f1': 0.73, 'down_f1': 0.53},
    'rpe1': {'accuracy': 0.87, 'macro_f1': 0.72, 'up_f1': 0.75, 'down_f1': 0.54},
}


def load_genept_embeddings(path='/Users/yiquntchen/Downloads/GenePT_emebdding_v2/GenePT_gene_embedding_ada_text.pickle'):
    """Load GenePT gene embeddings."""
    print("Loading GenePT embeddings...")
    with open(path, 'rb') as f:
        embeddings = pickle.load(f)
    # Convert to numpy arrays
    embeddings = {k: np.array(v) for k, v in embeddings.items()}
    print(f"  Loaded {len(embeddings)} gene embeddings (dim={len(list(embeddings.values())[0])})")
    return embeddings


def load_pathway_data():
    """Load MSigDB pathway gene sets."""
    print("Loading MSigDB pathways...")

    # Load KEGG and GO pathways
    kegg = gp.get_library('KEGG_2019_Human')
    go_bp = gp.get_library('GO_Biological_Process_2023')

    # Build gene -> pathways mapping
    gene_to_pathways = defaultdict(set)

    for pathway, genes in kegg.items():
        for gene in genes:
            gene_to_pathways[gene.upper()].add(pathway)

    for pathway, genes in go_bp.items():
        for gene in genes:
            gene_to_pathways[gene.upper()].add(pathway)

    print(f"  Loaded {len(kegg)} KEGG + {len(go_bp)} GO pathways")
    print(f"  {len(gene_to_pathways)} genes with pathway annotations")

    return gene_to_pathways


def compute_pathway_similarity(gene1, gene2, gene_to_pathways):
    """Compute Jaccard similarity of pathway memberships."""
    p1 = gene_to_pathways.get(gene1.upper(), set())
    p2 = gene_to_pathways.get(gene2.upper(), set())

    if not p1 or not p2:
        return 0.0

    intersection = len(p1 & p2)
    union = len(p1 | p2)
    return intersection / union if union > 0 else 0.0


def compute_embedding_similarity(gene1, gene2, embeddings):
    """Compute cosine similarity between gene embeddings."""
    e1 = embeddings.get(gene1.upper())
    e2 = embeddings.get(gene2.upper())

    if e1 is None or e2 is None:
        return 0.0

    # Cosine similarity
    return 1 - cosine(e1, e2)


class CombinedPredictor:
    """Combined predictor using majority vote + embeddings + pathways."""

    def __init__(self, embeddings, gene_to_pathways):
        self.embeddings = embeddings
        self.gene_to_pathways = gene_to_pathways
        self.gene_de_prior = {}
        self.gene_dir_prior = {}
        self.gene_cooccur = defaultdict(lambda: defaultdict(int))
        self.gene_de_count = defaultdict(int)
        self.scaler = None
        self.model_de = None
        self.model_dir = None

    def fit(self, train_de, train_dir):
        """Fit the model on training data."""
        print("Fitting combined model...")

        # Compute gene priors
        gene_de_count = defaultdict(int)
        gene_de_total = defaultdict(int)
        gene_dir_count = defaultdict(int)
        gene_dir_total = defaultdict(int)
        pert_de_genes = defaultdict(set)

        for item in train_de:
            gene_de_total[item['gene']] += 1
            if item['label'] == 1:
                gene_de_count[item['gene']] += 1
                pert_de_genes[item['pert']].add(item['gene'])

        for item in train_dir:
            gene_dir_total[item['gene']] += 1
            if item['label'] == 1:
                gene_dir_count[item['gene']] += 1

        self.gene_de_prior = {g: gene_de_count[g] / gene_de_total[g] for g in gene_de_total}
        self.gene_dir_prior = {g: gene_dir_count[g] / gene_dir_total[g] if gene_dir_total[g] > 0 else 0.5
                               for g in gene_dir_total}
        self.gene_de_count = dict(gene_de_count)

        # Compute co-occurrence
        for pert, de_genes in pert_de_genes.items():
            for g1 in de_genes:
                for g2 in de_genes:
                    if g1 != g2:
                        self.gene_cooccur[g1][g2] += 1

        # Build training features
        print("  Building training features...")
        np.random.seed(42)
        train_idx = np.random.choice(len(train_de), size=min(30000, len(train_de)), replace=False)

        X_train = []
        y_train = []
        for i in train_idx:
            item = train_de[i]
            features = self._get_features(item['pert'], item['gene'])
            X_train.append(features)
            y_train.append(item['label'])

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Train DE model
        print("  Training DE model...")
        self.model_de = LogisticRegression(max_iter=1000)
        self.model_de.fit(X_train_scaled, y_train)

        # Train DIR model
        print("  Training DIR model...")
        train_idx_dir = np.random.choice(len(train_dir), size=min(30000, len(train_dir)), replace=False)
        X_train_dir = np.array([self._get_features(train_dir[i]['pert'], train_dir[i]['gene'])
                                for i in train_idx_dir])
        y_train_dir = np.array([train_dir[i]['label'] for i in train_idx_dir])

        X_train_dir_scaled = self.scaler.transform(X_train_dir)
        self.model_dir = LogisticRegression(max_iter=1000)
        self.model_dir.fit(X_train_dir_scaled, y_train_dir)

        # Print feature importance
        feature_names = ['downstream_prior', 'target_prior', 'embedding_sim',
                        'pathway_sim', 'cooccur_norm', 'target_has_data']
        print("  DE model coefficients:")
        for name, coef in zip(feature_names, self.model_de.coef_[0]):
            print(f"    {name}: {coef:.4f}")

    def _get_features(self, pert_target, downstream_gene):
        """Extract features for a (perturbation, gene) pair."""
        features = []

        # 1. Downstream gene DE prior
        features.append(self.gene_de_prior.get(downstream_gene, 0.5))

        # 2. Target gene DE prior (responsiveness)
        features.append(self.gene_de_prior.get(pert_target, 0.5))

        # 3. GenePT embedding similarity
        emb_sim = compute_embedding_similarity(pert_target, downstream_gene, self.embeddings)
        features.append(emb_sim)

        # 4. Pathway co-membership similarity
        pathway_sim = compute_pathway_similarity(pert_target, downstream_gene, self.gene_to_pathways)
        features.append(pathway_sim)

        # 5. Training co-occurrence (normalized)
        cooccur = self.gene_cooccur.get(pert_target, {}).get(downstream_gene, 0)
        pert_de = self.gene_de_count.get(pert_target, 1)
        down_de = self.gene_de_count.get(downstream_gene, 1)
        cooccur_norm = cooccur / (pert_de + down_de - cooccur + 1) if cooccur > 0 else 0
        features.append(cooccur_norm)

        # 6. Target has data
        features.append(1 if pert_target in self.gene_de_prior else 0)

        return features

    def predict_proba(self, test_keys, task='de'):
        """Predict probabilities for test pairs."""
        X_test = np.array([self._get_features(p, g) for p, g in test_keys])
        X_test_scaled = self.scaler.transform(X_test)

        if task == 'de':
            return self.model_de.predict_proba(X_test_scaled)[:, 1]
        else:
            return self.model_dir.predict_proba(X_test_scaled)[:, 1]

    def predict_3class(self, test_keys, de_threshold=0.5, dir_threshold=0.5):
        """Predict 3-class labels."""
        pred_de = self.predict_proba(test_keys, 'de')
        pred_dir = self.predict_proba(test_keys, 'dir')

        predictions = []
        for i in range(len(test_keys)):
            if pred_de[i] < de_threshold:
                predictions.append('not_de')
            elif pred_dir[i] >= dir_threshold:
                predictions.append('up')
            else:
                predictions.append('down')

        return predictions, pred_de, pred_dir


def evaluate_dataset(dataset, embeddings, gene_to_pathways):
    """Evaluate on a dataset."""
    print(f"\n{'='*80}")
    print(f"Dataset: {dataset.upper()}")
    print(f"{'='*80}")

    # Load data
    data_de = load_de(dataset)
    data_dir = load_dir(dataset)
    train_de, test_de = data_de['train'], data_de['test']
    train_dir, test_dir = data_dir['train'], data_dir['test']

    print(f"Train DE: {len(train_de)}, Test DE: {len(test_de)}")

    # Fit combined model
    model = CombinedPredictor(embeddings, gene_to_pathways)
    model.fit(train_de, train_dir)

    # Build test data
    test_keys = [(x['pert'], x['gene']) for x in test_de]
    de_test_dict = {(x['pert'], x['gene']): x['label'] for x in test_de}
    dir_test_dict = {(x['pert'], x['gene']): x['label'] for x in test_dir}

    y_true_de = np.array([x['label'] for x in test_de])
    y_true_3class = []
    for pert, gene in test_keys:
        is_de = de_test_dict.get((pert, gene), 0)
        is_up = dir_test_dict.get((pert, gene), 0)
        if is_de == 0:
            y_true_3class.append('not_de')
        elif is_up == 1:
            y_true_3class.append('up')
        else:
            y_true_3class.append('down')

    # === BASELINE: Gene prior only ===
    print("\n--- BASELINE: Majority Vote (Gene Prior) ---")
    baseline_de_pred = np.array([model.gene_de_prior.get(g, 0.5) for p, g in test_keys])
    baseline_dir_pred = np.array([model.gene_dir_prior.get(g, 0.5) for p, g in test_keys])

    baseline_3class = []
    for i, (p, g) in enumerate(test_keys):
        if baseline_de_pred[i] < 0.5:
            baseline_3class.append('not_de')
        elif baseline_dir_pred[i] >= 0.5:
            baseline_3class.append('up')
        else:
            baseline_3class.append('down')

    labels = ['not_de', 'up', 'down']
    baseline_report = classification_report(y_true_3class, baseline_3class, labels=labels,
                                            output_dict=True, zero_division=0)

    print(f"Accuracy:  {baseline_report['accuracy']:.4f}")
    print(f"Macro F1:  {baseline_report['macro avg']['f1-score']:.4f}")
    print(f"Up F1:     {baseline_report['up']['f1-score']:.4f}")
    print(f"Down F1:   {baseline_report['down']['f1-score']:.4f}")

    # Per-gene AUROC for baseline
    gene_groups = defaultdict(list)
    for i, (pert, gene) in enumerate(test_keys):
        gene_groups[gene].append(i)

    baseline_aurocs = []
    for gene, indices in gene_groups.items():
        if len(indices) < 2:
            continue
        y_true_g = y_true_de[indices]
        if len(np.unique(y_true_g)) < 2:
            continue
        baseline_aurocs.append(roc_auc_score(y_true_g, baseline_de_pred[indices]))

    print(f"Per-gene AUROC: {np.mean(baseline_aurocs):.4f}")

    # === COMBINED MODEL ===
    print("\n--- COMBINED MODEL (Vote + Embeddings + Pathways) ---")
    model_3class, pred_de, pred_dir = model.predict_3class(test_keys)

    model_report = classification_report(y_true_3class, model_3class, labels=labels,
                                         output_dict=True, zero_division=0)

    print(f"Accuracy:  {model_report['accuracy']:.4f}")
    print(f"Macro F1:  {model_report['macro avg']['f1-score']:.4f}")
    print(f"Up F1:     {model_report['up']['f1-score']:.4f}")
    print(f"Down F1:   {model_report['down']['f1-score']:.4f}")

    # Per-gene AUROC for combined model
    model_aurocs = []
    for gene, indices in gene_groups.items():
        if len(indices) < 2:
            continue
        y_true_g = y_true_de[indices]
        if len(np.unique(y_true_g)) < 2:
            continue
        model_aurocs.append(roc_auc_score(y_true_g, pred_de[indices]))

    print(f"Per-gene AUROC: {np.mean(model_aurocs):.4f}")

    # === SYNTHPERT COMPARISON ===
    if dataset in SYNTHPERT_BASELINE:
        print(f"\n--- SYNTHPERT (from paper) ---")
        sp = SYNTHPERT_BASELINE[dataset]
        print(f"Accuracy:  {sp['accuracy']:.4f}")
        print(f"Macro F1:  {sp['macro_f1']:.4f}")
        print(f"Up F1:     {sp['up_f1']:.4f}")
        print(f"Down F1:   {sp['down_f1']:.4f}")

    # === SUMMARY ===
    print(f"\n--- IMPROVEMENT SUMMARY ---")
    print(f"{'Metric':<20} {'Baseline':<12} {'Combined':<12} {'Delta':<12}")
    print("-" * 55)
    print(f"{'Accuracy':<20} {baseline_report['accuracy']:<12.4f} {model_report['accuracy']:<12.4f} {model_report['accuracy']-baseline_report['accuracy']:+.4f}")
    print(f"{'Macro F1':<20} {baseline_report['macro avg']['f1-score']:<12.4f} {model_report['macro avg']['f1-score']:<12.4f} {model_report['macro avg']['f1-score']-baseline_report['macro avg']['f1-score']:+.4f}")
    print(f"{'Down F1':<20} {baseline_report['down']['f1-score']:<12.4f} {model_report['down']['f1-score']:<12.4f} {model_report['down']['f1-score']-baseline_report['down']['f1-score']:+.4f}")
    print(f"{'Up F1':<20} {baseline_report['up']['f1-score']:<12.4f} {model_report['up']['f1-score']:<12.4f} {model_report['up']['f1-score']-baseline_report['up']['f1-score']:+.4f}")
    print(f"{'Per-gene AUROC':<20} {np.mean(baseline_aurocs):<12.4f} {np.mean(model_aurocs):<12.4f} {np.mean(model_aurocs)-np.mean(baseline_aurocs):+.4f}")

    return {
        'dataset': dataset,
        'baseline': baseline_report,
        'model': model_report,
        'baseline_auroc': np.mean(baseline_aurocs),
        'model_auroc': np.mean(model_aurocs),
    }


if __name__ == "__main__":
    # Load external data
    embeddings = load_genept_embeddings()
    gene_to_pathways = load_pathway_data()

    # Evaluate on datasets
    results = []
    for dataset in ['k562', 'rpe1', 'hepg2', 'jurkat']:
        try:
            r = evaluate_dataset(dataset, embeddings, gene_to_pathways)
            results.append(r)
        except Exception as e:
            print(f"Error with {dataset}: {e}")
            import traceback
            traceback.print_exc()

    # Final summary
    print("\n" + "="*80)
    print("FINAL COMPARISON: Baseline vs Combined vs SynthPert")
    print("="*80)
    print(f"{'Dataset':<10} {'Baseline Acc':<12} {'Combined Acc':<12} {'SynthPert Acc':<14} {'Best':<10}")
    print("-"*60)

    for r in results:
        ds = r['dataset']
        base_acc = r['baseline']['accuracy']
        model_acc = r['model']['accuracy']
        sp_acc = SYNTHPERT_BASELINE.get(ds, {}).get('accuracy', 'N/A')

        if isinstance(sp_acc, float):
            best = 'Combined' if model_acc > max(base_acc, sp_acc) else ('Baseline' if base_acc > sp_acc else 'SynthPert')
            print(f"{ds:<10} {base_acc:<12.4f} {model_acc:<12.4f} {sp_acc:<14.4f} {best:<10}")
        else:
            best = 'Combined' if model_acc > base_acc else 'Baseline'
            print(f"{ds:<10} {base_acc:<12.4f} {model_acc:<12.4f} {'N/A':<14} {best:<10}")
