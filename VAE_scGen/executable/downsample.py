import os 
import pandas as pd
import numpy as np
import argparse

def filter_perturbqa_style(
    df: pd.DataFrame,
    de_pvalue_threshold1: float = 0.05,
    de_pvalue_threshold2: float = 0.01,
    non_de_pvalue_threshold: float = 0.1,
    min_degs_per_pert: int = 5,
    top_degs_per_pert: int = 20,
    non_degs_per_pert: int = 100,
    n_negative_controls: int = 100,
    p_value_col: str = 'p_value',
    pert_col: str = 'pert',
    gene_col: str = 'gene',
    label_col: str = 'label',
    random_seed: int = 42,
    save_path: str = None
) -> pd.DataFrame:
    """
    Filter DE_DOC results following PerturbQA selection criteria.

    Selection rules:
    1. Exclude perturbations with ≤ min_degs_per_pert DEGs (p < de_pvalue_threshold1)
       Exception: Keep n_negative_controls perturbations with 0 DEGs as negative controls
    2. For remaining perturbations:
       - DE pairs: p < de_pvalue_threshold2
       - Non-DE pairs: p > non_de_pvalue_threshold2
    3. Per perturbation, select:
       - Top top_degs_per_pert DEGs ranked by p-value as positives
       - Sample non_degs_per_pert non-DEGs as negatives
    4. Get the union of all selected genes for every perturbation to make it usable for VAE model.

    Args:
        df: DataFrame with DE_DOC statistics (output from DE_DOC_stats.py)
        de_pvalue_threshold1: P-value threshold for counting DEGs per perturbation (default: 0.05)
        de_pvalue_threshold2: P-value threshold for selecting DE pairs (default: 0.01)
        non_de_pvalue_threshold: P-value threshold for non-DE (default: 0.1)
        min_degs_per_pert: Minimum DEGs per perturbation to keep (default: 5)
        top_degs_per_pert: Number of top DEGs to select per perturbation (default: 20)
        non_degs_per_pert: Number of non-DEGs to sample per perturbation (default: 100)
        n_negative_controls: Number of 0-DEG perturbations to keep as controls (default: 100)
        p_value_col: Column name for p-values (default: 'p_value')
        pert_col: Column name for perturbation (default: 'pert')
        gene_col: Column name for gene (default: 'gene')
        label_col: Column name for DE/DOC label (default: 'label')
        random_seed: Random seed for sampling (default: 42)
        save_path: Optional path to save filtered CSV (default: None)

    Returns:
        Filtered DataFrame with selected perturbation-gene pairs

    Examples:
        >>> df = pd.read_csv('DE_DOC_rpe1.csv')
        >>> filtered_df = filter_perturbqa_style(df, save_path='DE_DOC_rpe1_filtered.csv')
    """
    np.random.seed(random_seed)

    print(f"\n{'='*70}")
    print("PerturbQA-Style Filtering")
    print(f"{'='*70}")
    print(f"Input: {len(df)} perturbation-gene pairs")
    print(f"\nParameters:")
    print(f"  DE p-value threshold 1 (for counting DEGs): {de_pvalue_threshold1}")
    print(f"  DE p-value threshold 2 (for selecting DE pairs): {de_pvalue_threshold2}")
    print(f"  Non-DE p-value threshold: {non_de_pvalue_threshold}")
    print(f"  Min DEGs per perturbation: {min_degs_per_pert}")
    print(f"  Top DEGs per perturbation: {top_degs_per_pert}")
    print(f"  Non-DEGs per perturbation: {non_degs_per_pert}")
    print(f"  Negative control perturbations: {n_negative_controls}")

    # Count DEGs per perturbation using threshold1
    deg_counts = (
        df.assign(is_deg = df[p_value_col] < de_pvalue_threshold1)
        .groupby(pert_col)["is_deg"]
        .sum()
    )
    deg_counts = deg_counts.reindex(df[pert_col].unique(), fill_value=0)

    # Separate perturbations by DEG count
    zero_deg_perts = [p for p, c in deg_counts.items() if c == 0]
    low_deg_perts = [p for p, c in deg_counts.items() if 0 < c <= min_degs_per_pert]
    high_deg_perts = [p for p, c in deg_counts.items() if c > min_degs_per_pert]

    print(f"\nPerturbation statistics:")
    print(f"  Perturbations with 0 DEGs: {len(zero_deg_perts)}")
    print(f"  Perturbations with 1-{min_degs_per_pert} DEGs: {len(low_deg_perts)}")
    print(f"  Perturbations with >{min_degs_per_pert} DEGs: {len(high_deg_perts)}")

    # Sample negative control perturbations (0 DEGs)
    if len(zero_deg_perts) > n_negative_controls:
        sampled_neg_controls = np.random.choice(
            zero_deg_perts,
            n_negative_controls,
            replace=False
        ).tolist()
    else:
        sampled_neg_controls = zero_deg_perts
        print(f"  Warning: Only {len(zero_deg_perts)} zero-DEG perturbations available, using all")

    # Perturbations to keep
    kept_perturbations = set(high_deg_perts + sampled_neg_controls)

    print(f"\nKept perturbations: {len(kept_perturbations)}")
    print(f"  High-DEG perturbations: {len(high_deg_perts)}")
    print(f"  Negative control perturbations: {len(sampled_neg_controls)}")
    print(f"\nExcluded perturbations: {len(low_deg_perts) + len(zero_deg_perts) - len(sampled_neg_controls)}")

    # Union of selected genes across perturbations
    grouped = df.groupby(pert_col)
    union_genes = set()
    for pert in kept_perturbations:
        pert_df = grouped.get_group(pert)

        # Classify as DE or non-DE based on p-value thresholds (using threshold2 for selection)
        de_pairs = pert_df[pert_df[p_value_col] < de_pvalue_threshold2]
        non_de_pairs = pert_df[pert_df[p_value_col] > non_de_pvalue_threshold]

        # Select top DEGs ranked by p-value
        if len(de_pairs) > 0:
            de_pairs_sorted = de_pairs.sort_values(p_value_col)
            selected_de = de_pairs_sorted.head(top_degs_per_pert)
            union_genes.update(selected_de[gene_col].tolist())

        # Sample non-DEGs
        if len(non_de_pairs) > 0:
            n_sample = min(non_degs_per_pert, len(non_de_pairs))
            sampled_non_de = non_de_pairs.sample(n=n_sample, random_state=random_seed)
            union_genes.update(sampled_non_de[gene_col].tolist())

    # Combine all selected pairs
    filtered_df = df[
        (df[pert_col].isin(kept_perturbations)) &
        (df[gene_col].isin(union_genes))
    ].copy()

    # Update labels based on new thresholds
    # DE: p < de_pvalue_threshold2, keep original direction labels (1=up, 2=down)
    def update_label(row):
        if row[p_value_col] < de_pvalue_threshold2:
            # Keep original direction label (1 or 2)
            return row[label_col]
        else:
            # Non-DE
            return 0

    filtered_df[label_col] = filtered_df.apply(update_label, axis=1)

    # Print summary statistics
    print(f"\n{'='*70}")
    print("Filtering Results")
    print(f"{'='*70}")
    print(f"Total filtered pairs: {len(filtered_df)}")
    print(f"\nLabel distribution:")
    print(f"  Non-DE (0): {(filtered_df[label_col] == 0).sum()}")
    print(f"  Up-regulated (1): {(filtered_df[label_col] == 1).sum()}")
    print(f"  Down-regulated (2): {(filtered_df[label_col] == 2).sum()}")

    pairs_per_pert = filtered_df.groupby(pert_col).size()
    print(f" Remaining genes: {pairs_per_pert.mean():.1f}")

    # Save if path provided
    if save_path:
        filtered_df.to_csv(save_path, index=False)
        print(f"\nFiltered results saved to {save_path}")

    return filtered_df

def run_perturbqa_filter_from_csv(
    input_csv_path: str,
    output_csv_path: str = None,
    **filter_kwargs
) -> pd.DataFrame:
    """
    Wrapper function:
    1. Read DE_DOC CSV
    2. Call filter_perturbqa_style

    Args:
        input_csv_path: Path to original DE_DOC CSV
        output_csv_path: Path to save filtered CSV
                         If None, auto-generate with suffix '_filtered.csv'
        **filter_kwargs: Any parameters passed to filter_perturbqa_style

    Returns:
        Filtered DataFrame
    """

    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Input file not found: {input_csv_path}")

    print(f"\nReading CSV from: {input_csv_path}")
    df = pd.read_csv(input_csv_path)
    print(f"Loaded {len(df)} rows")

    filtered_df = filter_perturbqa_style(
        df,
        save_path=output_csv_path, 
        **filter_kwargs
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate accuracy of predicted DE labels against ground truth')
    parser.add_argument('--input_path', type=str, required=True, help='Path to original CSV file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to filtered CSV file')
    parser.add_argument('--non_degs_per_pert', type=int, required=True, help='Number of non-DEGs to sample per perturbation')
    parser.add_argument('--min_degs_per_pert', type=int, default=10, help='Minimum DEGs per perturbation to keep (default: 20)')
    args = parser.parse_args()
    
    run_perturbqa_filter_from_csv(
        input_csv_path=args.input_path,
        output_csv_path=args.output_path,
        non_degs_per_pert=args.non_degs_per_pert,
        min_degs_per_pert = args.min_degs_per_pert
        # Add any specific filter parameters here
    )
    # Example usage:
    # python downsample.py --input_path /path/to/DE_DOC_rpe1.csv --output_path /path/to/DE_DOC_rpe1_filtered.csv --non_degs_per_pert 1 --min_degs_per_pert 10
