'''
train_test_split.py
===================

Train test split on perturbations with stratified label balancing.

The approach is:
1. Group perturbations by their label distribution (proportion of 0, 1, 2).
2. Use stratified sampling on perturbations to ensure train/test have similar 
   overall label proportions.
'''

import pandas as pd
from sklearn.model_selection import train_test_split

def split_train_test(df: pd.DataFrame, file_path: str, test_size: float = 0.25, random_state: int = 305) -> pd.DataFrame:
    """
    Split by perturbations while maintaining balanced label proportions in train/test.

    Uses stratified sampling based on each perturbation's dominant label class
    to ensure train and test sets have roughly equal proportions of labels (0, 1, 2).

    Returns:
        DataFrame of Perturbation-level split summary with only two columns:
            - 'pert'
            - 'split' ('train' or 'test')
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

    # Check for rare strata (< 2 members) which would cause stratified split to fail
    strata_counts = pert_dominant_label.value_counts()
    rare_strata = strata_counts[strata_counts < 2].index.tolist()

    if rare_strata:
        print(f"\nWARNING: Strata with <2 perturbations detected: {rare_strata}")
        print("  Merging rare strata into the most common stratum for stratification.")
        most_common_stratum = strata_counts.idxmax()
        pert_dominant_label = pert_dominant_label.replace(rare_strata, most_common_stratum)

    # Perform stratified split on perturbations
    perturbations = pert_dominant_label.index.values
    strata = pert_dominant_label.values

    train_perts, test_perts = train_test_split(
        perturbations,
        test_size=test_size,
        random_state=random_state,
        stratify=strata
    )

    # Create perturbation-level summary
    split_summary = pd.DataFrame({
        "pert": perturbations,
        "split": ["train" if p in set(train_perts) else "test" for p in perturbations]
    })
    print(f"\nPerturbation-level split summary:")
    print(split_summary["split"].value_counts())

    # Save summary CSV
    split_summary.to_csv(file_path, index=False)
    print(f"\nSaved perturbation-level split summary to {file_path}")

    return split_summary

if __name__ == "__main__":

    # read in csv
    df = pd.read_csv("HepG2.csv")

    # perform train/test split
    df_strict_splitted = split_train_test(df, file_path=".csv/HepG2_splitted.csv")

    print("\nDone!")