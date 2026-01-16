import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from analysis_utility import update_DE_DOC_labels

def split_train_test(df: pd.DataFrame, file_path: str, test_size: float = 0.25, random_state: int = 305) -> pd.DataFrame:
    """
    Load DE+DOC dataframe and add train/test split column.

    Args:
        df: Pandas dataframe of DE/DOC statistics
        file_path: String indicating file save path
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with 'split' column ('train' or 'test')
    """

    print(f"Loaded DE+DOC file: {len(df)} perturbation-gene pairs")
    print(f"\n3-Class label distribution:")
    print(f"  Non-DE (0): {(df['label'] == 0).sum()}")
    print(f"  Up-regulated (1): {(df['label'] == 1).sum()}")
    print(f"  Down-regulated (2): {(df['label'] == 2).sum()}")

    # Split into train/test indices
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=test_size,
        random_state=random_state,
        stratify=df['label']
    )

    # Add split column
    df['split'] = 'test'
    df.loc[train_idx, 'split'] = 'train'

    # Save to CSV
    df.to_csv(file_path, index=False)

    return df

if __name__ == "__main__":

    # read in csv
    df = pd.read_csv("files/CD4_TCell_Rest.csv")
    
    # update labels based on set p-value and log2FC threshold
    df_strict = update_DE_DOC_labels(df, alpha=0.10, log2fc_threshold=0.0)

    # perform train/test split
    df_strict_splitted = split_train_test(df_strict, file_path="files/CD4_TCell_Rest.csv")

    print("\nDone!")