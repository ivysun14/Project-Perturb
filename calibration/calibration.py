"""
calibration.py
==============
Post-prediction zero calibration for an output AnnData object.
"""

import argparse
import os
import numpy as np
import anndata as ad
from scipy import sparse


def calibrate_zeros(    
    adata: ad.AnnData,
    target_col: str,
    control_name: str,
    ratio: float
):
    """
    Perform per-gene zero-value calibration for perturbed cells based on the zero-expression ratio observed in control cells.

    For each gene g:

        1. Compute zero ratio in control cells:
               z_g = (# zeros in control cells) / (# control cells)
        2. Define a calibration quantile:
               q_g = z_g * {ratio}
        3. Compute threshold (pooled across all perturbed cells):
               T_g = quantile(X_pert[:, g], q_g)
        4. Set values below threshold to zero:
               X_pert_calibrated[i, g] = 0 
                   if X_pert[i, g] <= T_g

    Important args:
        ratio : Scaling factor for zero calibration strength.
    """
    X = adata.X
    if sparse.issparse(X):
        X = X.toarray()
    ctrl_mask = adata.obs[target_col] == control_name
    X_ctrl = X[ctrl_mask]
    X_pert = X[~ctrl_mask]

    n_genes = X.shape[1]
    zero_ratios = (X_ctrl == 0).mean(axis=0)

    print(f"Control cells          : {ctrl_mask.sum()}")
    print(f"Perturbed cells        : {(~ctrl_mask).sum()}")
    print(f"Zero fraction (before) : {(X_pert == 0).mean():.4f}")
    print(f"Start calibrating output by setting some zeroes...")

    X_pert_calibrated = X_pert.copy()
    Q = 0
    for g in range(n_genes):
        q = zero_ratios[g] * ratio
        threshold = np.quantile(X_pert[:, g], q)
        X_pert_calibrated[:, g] = np.where(
            X_pert[:, g] <= threshold,
            0,
            X_pert[:, g],
        )
        Q += zero_ratios[g]

    print(f"Average zero expression ratio in control data: {Q/n_genes:.4f}")
    print(f"Zero fraction (after)  : {(X_pert_calibrated == 0).mean():.4f}")

    X_new = X.copy()
    X_new[~ctrl_mask] = X_pert_calibrated
    adata.X = X_new
    return adata


def main():
    parser = argparse.ArgumentParser(description="Post-prediction zero calibration for AnnData outputs.")
    parser.add_argument("--input", required=True, help="Path to predicted h5ad.")
    parser.add_argument("--target_col", default="pert", help="obs column for perturbation identity.")
    parser.add_argument("--control_name", default="non-targeting", help="Control label in target_col.")
    parser.add_argument("--ratio", type=float, default=0.8, help="Calibration strength in [0, 1].")
    parser.add_argument("--out_dir", default="calibrated", help="Output directory.")
    args = parser.parse_args()

    print(f"Loading {args.input} ...")
    adata = ad.read_h5ad(args.input)
    print(f"  {adata.n_obs} cells x {adata.n_vars} genes")
    unique_perts = sorted(
        p for p in adata.obs[args.target_col].unique()
        if p != args.control_name
    )
    print(f"  Unique perturbations   : {len(unique_perts)}")
    print(f"  Perturbation-gene pairs: {len(unique_perts) * adata.n_vars}")

    adata = calibrate_zeros(
        adata,
        target_col=args.target_col,
        control_name=args.control_name,
        ratio=args.ratio,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, os.path.basename(args.input))
    print(f"Saving to {out_path} ...")
    adata.write_h5ad(out_path)
    print("Done.")

if __name__ == "__main__":
    main()