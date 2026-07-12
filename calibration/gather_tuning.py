"""
gather_tuning.py
================
Merge partial CSVs produced by tune_calibration.py array tasks into one
final tuning CSV per (method, model) combination.

Usage
-----
    python gather_tuning.py --model Scouter     --method var      --out_dir calibration/tuning_results
    python gather_tuning.py --model TranScouter --method sparsity --out_dir calibration/tuning_results
    python gather_tuning.py --model all         --method all      --out_dir calibration/tuning_results
"""

import argparse
import os
import glob
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',   default='all')
    parser.add_argument('--method',  default='all')
    parser.add_argument('--out_dir', default='calibration/tuning_results')
    args = parser.parse_args()

    out_dir = args.out_dir
    pattern = os.path.join(out_dir, f'{args.method}_{args.model}_task*_partial.csv')
    partial_files = sorted(glob.glob(pattern))

    if not partial_files:
        print(f"No partial CSVs found matching: {pattern}")
        raise SystemExit(1)

    df = pd.concat([pd.read_csv(f) for f in partial_files], ignore_index=True)
    df = df.sort_values(['model', 'method', 'split', 'r']).reset_index(drop=True)

    # One final CSV per (method, model) combination
    for (method, model), group in df.groupby(['method', 'model']):
        out_path = os.path.join(out_dir, f'{method}_{model}_tuning.csv')
        group.to_csv(out_path, index=False)
        print(f"Saved {len(group)} rows → {out_path}")
        # Print best r per split by balanced_accuracy
        best = (group.sort_values('balanced_accuracy', ascending=False)
                     .groupby('split').first()[['r', 'balanced_accuracy', 'f1_macro']])
        print(best.to_string())
        print()
