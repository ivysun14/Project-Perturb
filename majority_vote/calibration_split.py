'''
calibration_split.py
====================

Adds a `split_calib` column to an existing DE/DOC splitted CSV.

The original `split` column ('train'/'test') is left untouched so all
existing scripts continue to work unchanged. The new `split_calib` column
mirrors `split` except that 20% of training perturbations are relabelled
'calibration' for tuning post-prediction calibration hyperparameters.

Usage:
    python calibration_split.py <DE_DOC_*_splitted.csv>
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split_calibration(
    df: pd.DataFrame,
    file_path: str,
    cal_size: float = 0.20,
    random_state: int = 305,
) -> pd.DataFrame:
    """
    Add a `split_calib` column that mirrors `split` but reassigns a fraction
    of training perturbations to 'calibration'. The original `split` column
    is not modified.

    Args:
        df           : DataFrame with columns [pert, gene, label, split].
        file_path    : Path to write the updated CSV (overwrites input).
        cal_size     : Fraction of training perturbations to use as calibration.
                       Default 0.20 gives a 60/15/25 train/cal/test overall
                       split when the prior train/test was 75/25.
        random_state : Random seed for reproducibility.

    Returns:
        DataFrame with the new `split_calib` column added.
    """
    train_perts = df[df['split'] == 'train']['pert'].unique()
    print(f"Training perturbations available: {len(train_perts)}")

    # Stratify by dominant DE class — same approach as train_test_split.py
    train_sub = df[df['pert'].isin(train_perts)]
    counts = train_sub.groupby('pert')['label'].value_counts().unstack(fill_value=0)
    for label in [0, 1, 2]:
        if label not in counts.columns:
            counts[label] = 0
    dominant = counts[[0, 1, 2]].div(counts[[0, 1, 2]].sum(axis=1), axis=0).idxmax(axis=1)

    # Merge rare strata (<2 members) into the most common one
    strata_counts = dominant.value_counts()
    rare = strata_counts[strata_counts < 2].index.tolist()
    if rare:
        dominant = dominant.replace(rare, strata_counts.idxmax())

    _, cal_perts = train_test_split(
        dominant.index.values,
        test_size=cal_size,
        random_state=random_state,
        stratify=dominant.values,
    )

    # Build split_calib by copying split and relabelling calibration perturbations
    cal_set = set(cal_perts)
    df['split_calib'] = df.apply(
        lambda r: 'calibration' if r['pert'] in cal_set else r['split'], axis=1
    )

    print(f"\nSplit statistics (split_calib):")
    for s in ['train', 'calibration', 'test']:
        sub = df[df['split_calib'] == s]
        n_perts = sub['pert'].nunique()
        print(f"  {s.capitalize():12s}: {n_perts} perturbations, {len(sub)} pairs")
        for label in [0, 1, 2]:
            count = (sub['label'] == label).sum()
            pct = 100 * count / len(sub)
            print(f"    Label {label}: {count} ({pct:.1f}%)")

    df.to_csv(file_path, index=False)
    print(f"\nSaved to {file_path}")
    return df


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python calibration_split.py <DE_DOC_*_splitted.csv>")
        sys.exit(1)

    input_path = sys.argv[1]
    df = pd.read_csv(input_path)
    split_calibration(df, file_path=input_path)
    print("\nDone!")
