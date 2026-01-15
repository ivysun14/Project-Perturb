"""
error_utility.py
================

Helper class and functions for comprehensive error analysis for the majority vote baseline on PerturbQA.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import Dict, List, Tuple
from scipy.stats import mannwhitneyu


class ErrorAnalyzer:
    """
    Analyze errors from majority vote baseline predictions. Supports both binary and multi-class classification.
    """

    def __init__(self, dataset, test_keys, predictions, true_labels, adata=None, task_name="DE", class_names=None):
        """
        Initialize error analyzer.

        Args:
            dataset: String of dataset name (for book-keeping)
            test_keys: List of (pert, gene) tuples
            predictions: Array of predictions
            true_labels: Array of true labels
            adata: AnnData object (optional, for expression analysis)
            task_name: "DE" or "DOC"
            class_names: Optional list of class names for display (e.g., ['Non-DE', 'Up', 'Down'])
        """

        self.dataset = dataset
        self.test_keys = test_keys
        self.predictions = np.asarray(predictions)
        self.true_labels = np.asarray(true_labels)
        self.adata = adata
        self.task_name = task_name

        # Determine number of classes
        self.n_classes = max(len(np.unique(true_labels)), len(np.unique(predictions)))
        self.is_binary = (self.n_classes == 2)

        # Set class names
        if class_names is not None:
            self.class_names = class_names
        elif self.is_binary:
            self.class_names = ['Negative', 'Positive']
        else:
            self.class_names = [f'Class {i}' for i in range(self.n_classes)]

        # Categorize predictions
        if self.is_binary:
            # Binary classification: use traditional TP/TN/FP/FN
            self.tp_mask = (self.predictions == 1) & (self.true_labels == 1)
            self.tn_mask = (self.predictions == 0) & (self.true_labels == 0)
            self.fp_mask = (self.predictions == 1) & (self.true_labels == 0)
            self.fn_mask = (self.predictions == 0) & (self.true_labels == 1)
        else:
            # Multi-class: compute per-class TP/FP/FN
            self.class_tp_masks = {}
            self.class_fp_masks = {}
            self.class_fn_masks = {}
            for c in range(self.n_classes):
                self.class_tp_masks[c] = (self.predictions == c) & (self.true_labels == c)
                self.class_fp_masks[c] = (self.predictions == c) & (self.true_labels != c)
                self.class_fn_masks[c] = (self.predictions != c) & (self.true_labels == c)

        # Correct/incorrect masks (works for any number of classes)
        self.correct_mask = self.predictions == self.true_labels
        self.error_mask = ~self.correct_mask

        # Extract perturbations and genes
        self.perts = [k[0] for k in test_keys]
        self.genes = [k[1] for k in test_keys]

        # Print summary
        print(f"\n{'='*60}")
        print(f"Error Analysis for {task_name}")
        print(f"{'='*60}")
        print(f"Total samples: {len(self.predictions)}")
        print(f"Number of classes: {self.n_classes}")

        if self.is_binary:
            print(f"True Positives: {self.tp_mask.sum()}")
            print(f"True Negatives: {self.tn_mask.sum()}")
            print(f"False Positives: {self.fp_mask.sum()}")
            print(f"False Negatives: {self.fn_mask.sum()}")
        else:
            print(f"\nPer-class breakdown:")
            for c in range(self.n_classes):
                class_name = self.class_names[c] if c < len(self.class_names) else f'Class {c}'
                tp = self.class_tp_masks[c].sum()
                fp = self.class_fp_masks[c].sum()
                fn = self.class_fn_masks[c].sum()
                print(f"  {class_name}: TP={tp}, FP={fp}, FN={fn}")

        print(f"\nCorrect: {self.correct_mask.sum()}")
        print(f"Errors: {self.error_mask.sum()}")
        print(f"Accuracy: {self.correct_mask.sum() / len(self.predictions):.3f}")
        
    def create_error_dataframe(self):
        """
        Create a dataframe with all predictions and error categorization.

        Return:
            Dataframe summarizing prediction results and error information.
        """

        df = pd.DataFrame({
            'perturbation': self.perts,
            'gene': self.genes,
            'prediction': self.predictions,
            'true_label': self.true_labels,
            'correct': self.correct_mask,
        })

        if self.is_binary:
            # Binary: use traditional TP/TN/FP/FN labels
            df['error_type'] = ['TP' if tp else 'TN' if tn else 'FP' if fp else 'FN'
                               for tp, tn, fp, fn in zip(self.tp_mask, self.tn_mask,
                                                          self.fp_mask, self.fn_mask)]
        else:
            # Multi-class: use "Correct" or "pred->true" format for errors
            error_types = []
            for i in range(len(self.predictions)):
                if self.correct_mask[i]:
                    error_types.append('Correct')
                else:
                    pred_name = self.class_names[self.predictions[i]] if self.predictions[i] < len(self.class_names) else f'C{self.predictions[i]}'
                    true_name = self.class_names[self.true_labels[i]] if self.true_labels[i] < len(self.class_names) else f'C{self.true_labels[i]}'
                    error_types.append(f'{pred_name}->{true_name}')
            df['error_type'] = error_types

        # Save full results
        output_file = f"error_analysis/{self.dataset}_{self.task_name.lower()}_predictions.csv"
        df.to_csv(output_file, index=False)
        print(f"\nFull predictions with true labels saved to {output_file}")

        return df
    
    def analyze_by_gene(self, top_k=20):
        """
        Analyze which genes have highest error rates.

        Args:
            top_k: Output top k number of genes with highest error rates.

        Return:
            Dataframe summarizing for each gene the error rate and per-class error counts.
        """

        print(f"\n{'='*60}")
        print("Error Analysis by Gene")
        print(f"{'='*60}")

        if self.is_binary:
            # Binary classification: track fp/fn
            gene_stats = defaultdict(lambda: {'total': 0, 'errors': 0, 'fp': 0, 'fn': 0})

            for i, gene in enumerate(self.genes):
                gene_stats[gene]['total'] += 1
                if self.error_mask[i]:
                    gene_stats[gene]['errors'] += 1
                if self.fp_mask[i]:
                    gene_stats[gene]['fp'] += 1
                if self.fn_mask[i]:
                    gene_stats[gene]['fn'] += 1

            # Calculate error rates
            gene_error_rates = []
            for gene, stats in gene_stats.items():
                if stats['total'] == 0:
                    raise ValueError(f'Total count of gene {gene} is 0.')
                error_rate = stats['errors'] / stats['total']
                gene_error_rates.append({
                    'gene': gene,
                    'total': stats['total'],
                    'errors': stats['errors'],
                    'error_rate': error_rate,
                    'fp': stats['fp'],
                    'fn': stats['fn']
                })
        else:
            # Multi-class: track per-class FP/FN
            default_stats = {'total': 0, 'errors': 0}
            for c in range(self.n_classes):
                default_stats[f'fp_class_{c}'] = 0
                default_stats[f'fn_class_{c}'] = 0

            gene_stats = defaultdict(lambda: dict(default_stats))

            for i, gene in enumerate(self.genes):
                gene_stats[gene]['total'] += 1
                if self.error_mask[i]:
                    gene_stats[gene]['errors'] += 1
                for c in range(self.n_classes):
                    if self.class_fp_masks[c][i]:
                        gene_stats[gene][f'fp_class_{c}'] += 1
                    if self.class_fn_masks[c][i]:
                        gene_stats[gene][f'fn_class_{c}'] += 1

            # Calculate error rates
            gene_error_rates = []
            for gene, stats in gene_stats.items():
                if stats['total'] == 0:
                    raise ValueError(f'Total count of gene {gene} is 0.')
                error_rate = stats['errors'] / stats['total']
                row = {
                    'gene': gene,
                    'total': stats['total'],
                    'errors': stats['errors'],
                    'error_rate': error_rate,
                }
                for c in range(self.n_classes):
                    class_name = self.class_names[c] if c < len(self.class_names) else f'class_{c}'
                    row[f'fp_{class_name}'] = stats[f'fp_class_{c}']
                    row[f'fn_{class_name}'] = stats[f'fn_class_{c}']
                gene_error_rates.append(row)

        df = pd.DataFrame(gene_error_rates).sort_values('error_rate', ascending=False)

        print(f"\nTop {top_k} genes with highest error rates:")
        print(df.head(top_k).to_string(index=False))

        # Save full results
        output_file = f"error_analysis/{self.dataset}_{self.task_name.lower()}_gene_errors.csv"
        df.to_csv(output_file, index=False)
        print(f"\nFull gene error analysis saved to {output_file}")

        return df
    
    def analyze_by_perturbation(self, top_k=20):
        """
        Analyze which perturbations have highest error rates.

        Args:
            top_k: Output top k number of perturbations with highest error rates.

        Return:
            Dataframe summarizing for each perturbation the error rate and per-class error counts.
        """

        print(f"\n{'='*60}")
        print("Error Analysis by Perturbation")
        print(f"{'='*60}")

        if self.is_binary:
            # Binary classification: track fp/fn
            pert_stats = defaultdict(lambda: {'total': 0, 'errors': 0, 'fp': 0, 'fn': 0})

            for i, pert in enumerate(self.perts):
                pert_stats[pert]['total'] += 1
                if self.error_mask[i]:
                    pert_stats[pert]['errors'] += 1
                if self.fp_mask[i]:
                    pert_stats[pert]['fp'] += 1
                if self.fn_mask[i]:
                    pert_stats[pert]['fn'] += 1

            # Calculate error rates
            pert_error_rates = []
            for pert, stats in pert_stats.items():
                if stats['total'] == 0:
                    raise ValueError(f'Total count of perturbation {pert} is 0.')
                error_rate = stats['errors'] / stats['total']
                pert_error_rates.append({
                    'perturbation': pert,
                    'total': stats['total'],
                    'errors': stats['errors'],
                    'error_rate': error_rate,
                    'fp': stats['fp'],
                    'fn': stats['fn']
                })
        else:
            # Multi-class: track per-class FP/FN
            default_stats = {'total': 0, 'errors': 0}
            for c in range(self.n_classes):
                default_stats[f'fp_class_{c}'] = 0
                default_stats[f'fn_class_{c}'] = 0

            pert_stats = defaultdict(lambda: dict(default_stats))

            for i, pert in enumerate(self.perts):
                pert_stats[pert]['total'] += 1
                if self.error_mask[i]:
                    pert_stats[pert]['errors'] += 1
                for c in range(self.n_classes):
                    if self.class_fp_masks[c][i]:
                        pert_stats[pert][f'fp_class_{c}'] += 1
                    if self.class_fn_masks[c][i]:
                        pert_stats[pert][f'fn_class_{c}'] += 1

            # Calculate error rates
            pert_error_rates = []
            for pert, stats in pert_stats.items():
                if stats['total'] == 0:
                    raise ValueError(f'Total count of perturbation {pert} is 0.')
                error_rate = stats['errors'] / stats['total']
                row = {
                    'perturbation': pert,
                    'total': stats['total'],
                    'errors': stats['errors'],
                    'error_rate': error_rate,
                }
                for c in range(self.n_classes):
                    class_name = self.class_names[c] if c < len(self.class_names) else f'class_{c}'
                    row[f'fp_{class_name}'] = stats[f'fp_class_{c}']
                    row[f'fn_{class_name}'] = stats[f'fn_class_{c}']
                pert_error_rates.append(row)

        df = pd.DataFrame(pert_error_rates).sort_values('error_rate', ascending=False)

        print(f"\nTop {top_k} perturbations with highest error rates:")
        print(df.head(top_k).to_string(index=False))

        # Save full results
        output_file = f"error_analysis/{self.dataset}_{self.task_name.lower()}_perturbation_errors.csv"
        df.to_csv(output_file, index=False)
        print(f"\nFull perturbation error analysis saved to {output_file}")

        return df
    
    def analyze_gene_expression_patterns(self, control_key='non-targeting', target_col='target_gene', gene_row="gene_name"):
        """
        Analyze if errors correlate with gene expression characteristics.

        Args:
            control_key: name in adata.obs[@target_col] indicating controls
            target_col: adata.obs key indicating perturbations
            gene_row: adata.var row name representing genes measured

        Return:
            Dictionary of gene expression error analysis results separated by error type.
        
        TODO: Need to verify this function, so far not put into use!
        """

        if self.adata is None:
            print("\nSkipping expression analysis (AnnData not provided)")
            return None

        print(f"\n{'='*60}")
        print("Gene Expression Pattern Analysis")
        print(f"{'='*60}")

        # Get control expression
        control_mask = self.adata.obs[target_col] == control_key
        control_expr = self.adata.X[control_mask]

        # Calculate gene-level statistics
        gene_means = {}
        gene_vars = {}
        gene_zeros = {}

        for gene_idx, gene_name in enumerate(self.adata.var[gene_row]):
            vals = control_expr[:, gene_idx]
            gene_means[gene_name] = np.mean(vals)
            gene_vars[gene_name] = np.var(vals)
            gene_zeros[gene_name] = (vals == 0).sum() / len(vals)

        # Helper function to compute stats for a given error mask
        def compute_expression_stats(error_mask, error_name):
            error_genes = [self.genes[i] for i in range(len(self.genes)) if error_mask[i]]
            correct_genes = [self.genes[i] for i in range(len(self.genes)) if not error_mask[i]]

            if len(error_genes) == 0 or len(correct_genes) == 0:
                print(f"\n{error_name}: Skipping (insufficient samples)")
                return None

            # Get statistics for error vs correct
            error_means = [gene_means.get(g, 0) for g in error_genes]
            correct_means = [gene_means.get(g, 0) for g in correct_genes]

            error_vars = [gene_vars.get(g, 0) for g in error_genes]
            correct_vars = [gene_vars.get(g, 0) for g in correct_genes]

            error_zeros = [gene_zeros.get(g, 0) for g in error_genes]
            correct_zeros = [gene_zeros.get(g, 0) for g in correct_genes]

            # Statistical tests
            _, p_mean = mannwhitneyu(error_means, correct_means, alternative='two-sided')
            _, p_var = mannwhitneyu(error_vars, correct_vars, alternative='two-sided')
            _, p_zero = mannwhitneyu(error_zeros, correct_zeros, alternative='two-sided')

            return {
                'mean_expr': {
                    'error': np.mean(error_means),
                    'correct': np.mean(correct_means),
                    'p_value': p_mean
                },
                'variance': {
                    'error': np.mean(error_vars),
                    'correct': np.mean(correct_vars),
                    'p_value': p_var
                },
                'zero_rate': {
                    'error': np.mean(error_zeros),
                    'correct': np.mean(correct_zeros),
                    'p_value': p_zero
                }
            }

        results = {}

        if self.is_binary:
            # Binary classification: analyze FP and FN separately
            for error_type, mask in [('FP', self.fp_mask), ('FN', self.fn_mask)]:
                stats = compute_expression_stats(mask, error_type)
                if stats is not None:
                    results[error_type] = stats
        else:
            # Multi-class: analyze per-class FP and FN
            for c in range(self.n_classes):
                class_name = self.class_names[c] if c < len(self.class_names) else f'Class {c}'
                for error_type, mask in [(f'FP_{class_name}', self.class_fp_masks[c]),
                                          (f'FN_{class_name}', self.class_fn_masks[c])]:
                    stats = compute_expression_stats(mask, error_type)
                    if stats is not None:
                        results[error_type] = stats

            # Also analyze overall errors vs correct
            overall_stats = compute_expression_stats(self.error_mask, 'Overall Errors')
            if overall_stats is not None:
                results['Overall'] = overall_stats

        # Print results
        for error_type, stats in results.items():
            print(f"\n{error_type} Errors:")
            print(f"  Mean expression: {stats['mean_expr']['error']:.3f} vs {stats['mean_expr']['correct']:.3f} (p={stats['mean_expr']['p_value']:.4f})")
            print(f"  Variance: {stats['variance']['error']:.3f} vs {stats['variance']['correct']:.3f} (p={stats['variance']['p_value']:.4f})")
            print(f"  Zero rate: {stats['zero_rate']['error']:.3f} vs {stats['zero_rate']['correct']:.3f} (p={stats['zero_rate']['p_value']:.4f})")

        return results
    
    def analyze_prediction_distribution(self):
        """
        Analyze the distribution of predictions vs true labels.
        """

        print(f"\n{'='*60}")
        print("Prediction Distribution Analysis")
        print(f"{'='*60}")
        
        # Get unique predictions
        unique_preds = np.unique(self.predictions)
        
        print(f"\nPrediction distribution:")
        for pred in unique_preds:
            count = (self.predictions == pred).sum()
            pct = count / len(self.predictions) * 100
            print(f"  Predict {pred}: {count} ({pct:.1f}%)")
        
        print(f"\nTrue label distribution:")
        for label in np.unique(self.true_labels):
            count = (self.true_labels == label).sum()
            pct = count / len(self.true_labels) * 100
            print(f"  True {label}: {count} ({pct:.1f}%)")
        
        # Class imbalance analysis
        if len(unique_preds) == 1:
            print(f"\n  WARNING: Model only predicts class {unique_preds[0]}!")
            print("This could be a degenerate baseline - all genes have same majority class.")
    
    def plot_error_distributions(self, save_path=None):
        """
        Create visualization of error patterns.

        Args:
            save_path: path to save figures, default None
        """

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(self.true_labels, self.predictions)
        # Use class names for labels if available
        if self.n_classes <= len(self.class_names):
            tick_labels = self.class_names[:self.n_classes]
        else:
            tick_labels = [str(i) for i in range(self.n_classes)]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                    xticklabels=tick_labels, yticklabels=tick_labels)
        axes[0, 0].set_title(f'Confusion Matrix - {self.task_name}')
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')

        # 2. Error rate by gene (top 20 genes with more than 5 total appearance)
        gene_stats = defaultdict(lambda: {'total': 0, 'errors': 0})
        for i, gene in enumerate(self.genes):
            gene_stats[gene]['total'] += 1
            if self.error_mask[i]:
                gene_stats[gene]['errors'] += 1

        gene_error_rates = [(g, s['errors']/s['total']) for g, s in gene_stats.items() if s['total'] >= 5]
        gene_error_rates.sort(key=lambda x: x[1], reverse=True)
        top_genes = gene_error_rates[:20]

        if top_genes:
            genes_plot, rates_plot = zip(*top_genes)
            axes[0, 1].barh(range(len(genes_plot)), rates_plot)
            axes[0, 1].set_yticks(range(len(genes_plot)))
            axes[0, 1].set_yticklabels(genes_plot, fontsize=8)
            axes[0, 1].set_xlabel('Error Rate')
            axes[0, 1].set_title('Top 20 Genes by Error Rate')
            axes[0, 1].invert_yaxis()

        # 3. Error rate by perturbation (top 20 perturbations with more than 5 total appearance)
        pert_stats = defaultdict(lambda: {'total': 0, 'errors': 0})
        for i, pert in enumerate(self.perts):
            pert_stats[pert]['total'] += 1
            if self.error_mask[i]:
                pert_stats[pert]['errors'] += 1

        pert_error_rates = [(p, s['errors']/s['total']) for p, s in pert_stats.items() if s['total'] >= 5]
        pert_error_rates.sort(key=lambda x: x[1], reverse=True)
        top_perts = pert_error_rates[:20]

        if top_perts:
            perts_plot, rates_plot = zip(*top_perts)
            axes[1, 0].barh(range(len(perts_plot)), rates_plot)
            axes[1, 0].set_yticks(range(len(perts_plot)))
            axes[1, 0].set_yticklabels(perts_plot, fontsize=8)
            axes[1, 0].set_xlabel('Error Rate')
            axes[1, 0].set_title('Top 20 Perturbations by Error Rate')
            axes[1, 0].invert_yaxis()

        # 4. Error type distribution
        if self.is_binary:
            # Binary: show TP/TN/FP/FN
            error_counts = {
                'TP': self.tp_mask.sum(),
                'TN': self.tn_mask.sum(),
                'FP': self.fp_mask.sum(),
                'FN': self.fn_mask.sum()
            }
            axes[1, 1].bar(error_counts.keys(), error_counts.values())
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_title('Prediction Categories')
            for i, (k, v) in enumerate(error_counts.items()):
                axes[1, 1].text(i, v, str(v), ha='center', va='bottom')
        else:
            # Multi-class: show per-class correct/incorrect or TP counts
            correct_counts = []
            error_counts = []
            for c in range(self.n_classes):
                correct_counts.append(self.class_tp_masks[c].sum())
                error_counts.append(self.class_fp_masks[c].sum() + self.class_fn_masks[c].sum())

            x = np.arange(self.n_classes)
            width = 0.35
            bars1 = axes[1, 1].bar(x - width/2, correct_counts, width, label='Correct (TP)', color='steelblue')
            bars2 = axes[1, 1].bar(x + width/2, error_counts, width, label='Errors (FP+FN)', color='darkorange')

            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_title('Per-Class Prediction Summary')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(tick_labels, fontsize=9)
            axes[1, 1].legend()

            for bar in bars1:
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, height, str(int(height)),
                               ha='center', va='bottom', fontsize=8)
            for bar in bars2:
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, height, str(int(height)),
                               ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nError distribution plots saved to {save_path}")
        else:
            plt.savefig(f'error_analysis/{self.dataset}_{self.task_name.lower()}_error_analysis.png', dpi=300, bbox_inches='tight')
            print(f"\nError distribution plots saved to error_analysis/{self.task_name.lower()}_error_analysis.png")
        
        plt.close()
    
    def run_full_analysis(self):
        """
        Run complete error analysis pipeline.
        """

        print(f"\n{'#'*60}")
        print(f"# COMPREHENSIVE ERROR ANALYSIS - {self.task_name}")
        print(f"{'#'*60}")
        
        # Create error dataframe
        error_df = self.create_error_dataframe()
        
        # Analyze by gene
        gene_df = self.analyze_by_gene()
        
        # Analyze by perturbation
        pert_df = self.analyze_by_perturbation()
        
        # Analyze prediction distribution
        self.analyze_prediction_distribution()
        
        # Analyze expression patterns if available
        if self.adata is not None:
            expr_results = self.analyze_gene_expression_patterns()
        
        # Create visualizations
        self.plot_error_distributions()
        
        print(f"\n{'#'*60}")
        print("# ERROR ANALYSIS COMPLETE")
        print(f"{'#'*60}")
        
        return {
            'error_df': error_df,
            'gene_errors': gene_df,
            'pert_errors': pert_df
        }


def run_error_analysis_from_results(
    dataset,
    test_keys,
    predictions,
    true_labels,
    adata=None,
    task_name="DE",
    class_names=None
):
    """
    Convenience function to run error analysis.

    Args:
        dataset: String of dataset name (for book-keeping)
        test_keys: List of (pert, gene) tuples
        predictions: Array of predictions
        true_labels: Array of true labels
        adata: AnnData object (optional)
        task_name: "DE" or "DOC"
        class_names: Optional list of class names for display (e.g., ['Non-DE', 'Up', 'Down'])

    Returns:
        ErrorAnalyzer object with results
    """
    analyzer = ErrorAnalyzer(dataset, test_keys, predictions, true_labels, adata, task_name, class_names)
    results = analyzer.run_full_analysis()
    return analyzer, results


if __name__ == "__main__":
    print("Error Analysis Module")
    print("Import this module and use run_error_analysis_from_results()")