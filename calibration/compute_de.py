"""
compute_de.py
=============
Compute DE statistics on a calibrated prediction AnnData.

Runs Wilcoxon rank-sum test with BH correction for every (perturbation, gene)
pair and saves the result as a CSV for downstream metric evaluation.
"""

import argparse
import os
import sys
import anndata as ad

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_project_root)
sys.path.append(os.path.join(_project_root, "majority_vote"))
from majority_vote.DE_DOC_stats import compute_de_statistics
from majority_vote.analysis_utility import resolve_repeats

ALPHA = 0.01
LOG2FC_THRESHOLD = 0.00

def main():
    parser = argparse.ArgumentParser(description="Compute DE statistics on calibrated Scouter predictions.")
    parser.add_argument("--input", required=True, help="Path to calibrated h5ad.")
    parser.add_argument("--dataset", required=True, help="Dataset name used for output filename.")
    parser.add_argument("--target_col", default="pert", help="obs column for perturbation identity.")
    parser.add_argument("--control_name", default="non-targeting",help="Control label in target_col.")
    parser.add_argument("--out_dir",help="Directory for output DE CSV.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading {args.input} ...")
    adata = ad.read_h5ad(args.input)
    print(f"  {adata.n_obs} cells x {adata.n_vars} genes")
    adata = resolve_repeats(adata, gene_row="gene_name")
    print(f"  After resolve_repeats: {adata.n_obs} cells x {adata.n_vars} genes")

    out_path = os.path.join(args.out_dir, f"{args.dataset}_pred_DE.csv")
    compute_de_statistics(
        adata=adata,
        control_key=args.control_name,
        target_col=args.target_col,
        gene_row="gene_name",
        alpha=ALPHA,
        log2fc_threshold=LOG2FC_THRESHOLD,
        save_path=out_path
    )
    print(f"\nDE results saved to {out_path}")

if __name__ == "__main__":
    main()
