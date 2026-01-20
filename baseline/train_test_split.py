'''
train_test_split.py
===================

Train test split on perturbations with stratified label balancing.

The approach is:
1. Group perturbations by their label distribution (proportion of 0, 1, 2).
2. Use stratified sampling on perturbations to ensure train/test have similar 
   overall label proportions.
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from analysis_utility import update_DE_DOC_labels

def split_train_test(df: pd.DataFrame, file_path: str, test_size: float = 0.25, random_state: int = 305) -> pd.DataFrame:
    """
    Split by perturbations while maintaining balanced label proportions in train/test.

    Uses stratified sampling based on each perturbation's dominant label class
    to ensure train and test sets have roughly equal proportions of labels (0, 1, 2).

    Args:
        df: Pandas dataframe of DE/DOC statistics with columns [pert, gene, label]
        file_path: String indicating file save path
        test_size: Fraction of perturbations to use for testing
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with 'split' column ('train' or 'test')
    """

    print(f"Loaded DE+DOC file: {len(df)} perturbation-gene pairs")
    print(f"\n3-Class label distribution (overall):")
    print(f"  Non-DE (0): {(df['label'] == 0).sum()}")
    print(f"  Up-regulated (1): {(df['label'] == 1).sum()}")
    print(f"  Down-regulated (2): {(df['label'] == 2).sum()}")

    # Compute label distribution per perturbation
    pert_label_counts = df.groupby('pert')['label'].value_counts().unstack(fill_value=0)

    # Safety check: ensure all label columns exist
    for label in [0, 1, 2]:
        if label not in pert_label_counts.columns:
            pert_label_counts[label] = 0
            print(f"WARNING: Label {label} is missing from all perturbations!")
    pert_label_counts = pert_label_counts[[0, 1, 2]]

    # Compute proportions per perturbation
    pert_totals = pert_label_counts.sum(axis=1)
    pert_proportions = pert_label_counts.div(pert_totals, axis=0)

    # Assign each perturbation to a stratum based on dominant label
    pert_dominant_label = pert_proportions.idxmax(axis=1)

    # Perform stratified split on perturbations
    perturbations = pert_dominant_label.index.values
    strata = pert_dominant_label.values

    train_perts, test_perts = train_test_split(
        perturbations,
        test_size=test_size,
        random_state=random_state,
        stratify=strata
    )

    # Add split column
    df['split'] = df['pert'].apply(lambda x: 'train' if x in set(train_perts) else 'test')

    # Print split statistics
    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test']

    print(f"\nSplit statistics:")
    print(f"  Train perturbations: {len(train_perts)}, Test perturbations: {len(test_perts)}")
    print(f"  Train pairs: {len(train_df)}, Test pairs: {len(test_df)}")

    print(f"\nTrain label distribution:")
    for label in [0, 1, 2]:
        count = (train_df['label'] == label).sum()
        pct = 100 * count / len(train_df)
        print(f"  Label {label}: {count} ({pct:.1f}%)")

    print(f"\nTest label distribution:")
    for label in [0, 1, 2]:
        count = (test_df['label'] == label).sum()
        pct = 100 * count / len(test_df)
        print(f"  Label {label}: {count} ({pct:.1f}%)")

    # Save to CSV
    df.to_csv(file_path, index=False)
    print(f"\nSaved to {file_path}")

    return df

if __name__ == "__main__":

    # read in csv
    df = pd.read_csv("files/DE_DOC_jurkat.csv")
    
    # update labels based on set p-value and log2FC threshold
    #df_strict = update_DE_DOC_labels(df, alpha=0.10, log2fc_threshold=0.0)

    # perform train/test split
    df_strict_splitted = split_train_test(df, file_path="files/DE_DOC_jurkat_splitted.csv")

    print("\nDone!")