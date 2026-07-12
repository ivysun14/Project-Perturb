"""
Compute classification metrics for perturbation prediction outputs.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix
)
import os
import argparse


def compute_metrics_for_file(pred_file):
    """Compute all classification metrics for a prediction file."""
    df = pd.read_csv(pred_file)
    y_true = df['true_label'].values
    y_pred = df['pred_label'].values

    classes = sorted(np.unique(np.concatenate([y_true, y_pred])))

    results = {
        'n_samples': len(df),
        'classes': classes,
        'class_distribution_true': dict(zip(*np.unique(y_true, return_counts=True))),
        'class_distribution_pred': dict(zip(*np.unique(y_pred, return_counts=True))),
    }

    # Overall accuracy
    results['accuracy'] = accuracy_score(y_true, y_pred)

    # Per-class metrics
    per_class = {}
    for cls in classes:
        mask = y_true == cls
        per_class[cls] = {
            'accuracy': accuracy_score(y_true[mask], y_pred[mask]),
            'n_samples': int(mask.sum()),
            'precision': precision_score(y_true, y_pred, labels=[cls], average=None, zero_division=0)[0],
            'recall': recall_score(y_true, y_pred, labels=[cls], average=None, zero_division=0)[0],
            'f1': f1_score(y_true, y_pred, labels=[cls], average=None, zero_division=0)[0],
        }
    results['per_class'] = per_class

    # Macro averages
    results['macro_precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    results['macro_recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    results['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # Weighted averages
    results['weighted_precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    results['weighted_recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    results['weighted_f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Confusion matrix
    results['confusion_matrix'] = confusion_matrix(y_true, y_pred, labels=classes)

    return results


def print_metrics(dataset, task, metrics):
    """Print metrics in a formatted way."""
    print(f"\n{'='*70}")
    print(f"Dataset: {dataset.upper()} | Task: {task.upper()}")
    print(f"{'='*70}")
    print(f"Total samples: {metrics['n_samples']}")
    print(f"Class distribution (true): {metrics['class_distribution_true']}")
    print(f"Class distribution (pred): {metrics['class_distribution_pred']}")

    print(f"\n--- Overall Metrics ---")
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")

    print(f"\n--- Per-Class Accuracy ---")
    for cls, cls_metrics in metrics['per_class'].items():
        print(f"  Class {cls}: {cls_metrics['accuracy']:.4f} (n={cls_metrics['n_samples']})")

    print(f"\n--- Per-Class Precision, Recall, F1 ---")
    for cls, cls_metrics in metrics['per_class'].items():
        print(f"  Class {cls}: Precision={cls_metrics['precision']:.4f}, "
              f"Recall={cls_metrics['recall']:.4f}, F1={cls_metrics['f1']:.4f}")

    print(f"\n--- Macro Averages ---")
    print(f"  Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall: {metrics['macro_recall']:.4f}")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")

    print(f"\n--- Weighted Averages ---")
    print(f"  Weighted Precision: {metrics['weighted_precision']:.4f}")
    print(f"  Weighted Recall: {metrics['weighted_recall']:.4f}")
    print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")

    print(f"\n--- Confusion Matrix ---")
    print(f"  Labels: {metrics['classes']}")
    print(f"  {metrics['confusion_matrix']}")


def main(output_dir, save_csv=True):
    """Compute metrics for all datasets and tasks."""
    datasets = ['k562', 'hepg2', 'jurkat', 'rpe1']
    tasks = ['de', 'dir']

    summary_rows = []
    detailed_rows = []

    for dataset in datasets:
        for task in tasks:
            pred_file = os.path.join(output_dir, f'{dataset}_{task}_predictions.csv')
            if not os.path.exists(pred_file):
                print(f"File not found: {pred_file}")
                continue

            metrics = compute_metrics_for_file(pred_file)
            print_metrics(dataset, task, metrics)

            # Summary row
            summary_rows.append({
                'dataset': dataset,
                'task': task,
                'n_samples': metrics['n_samples'],
                'accuracy': metrics['accuracy'],
                'macro_precision': metrics['macro_precision'],
                'macro_recall': metrics['macro_recall'],
                'macro_f1': metrics['macro_f1'],
                'weighted_precision': metrics['weighted_precision'],
                'weighted_recall': metrics['weighted_recall'],
                'weighted_f1': metrics['weighted_f1'],
            })

            # Detailed per-class rows
            for cls, cls_metrics in metrics['per_class'].items():
                detailed_rows.append({
                    'dataset': dataset,
                    'task': task,
                    'class': cls,
                    'n_samples': cls_metrics['n_samples'],
                    'accuracy': cls_metrics['accuracy'],
                    'precision': cls_metrics['precision'],
                    'recall': cls_metrics['recall'],
                    'f1': cls_metrics['f1'],
                })

    # Print summary
    print(f"\n\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    if save_csv:
        # Save summary metrics
        summary_path = os.path.join(output_dir, 'metrics_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSaved summary metrics to: {summary_path}")

        # Save detailed per-class metrics
        detailed_df = pd.DataFrame(detailed_rows)
        detailed_path = os.path.join(output_dir, 'metrics_per_class.csv')
        detailed_df.to_csv(detailed_path, index=False)
        print(f"Saved per-class metrics to: {detailed_path}")

    return summary_df, pd.DataFrame(detailed_rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute classification metrics')
    parser.add_argument('--output_dir', type=str,
                        default='/Users/yiquntchen/Desktop/chen-lab/Project-Perturb/output',
                        help='Directory containing prediction CSV files')
    parser.add_argument('--no_save', action='store_true',
                        help='Do not save metrics to CSV')
    args = parser.parse_args()

    main(args.output_dir, save_csv=not args.no_save)
