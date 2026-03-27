"""
calibrate_var.py
================
Post-prediction variance calibration on AnnData object.
"""

import argparse
import os
import numpy as np
import anndata as ad
from scipy import sparse

# Cap on per-gene scale factor to prevent explosion when sigma^2_pred close to 0
_MAX_SCALE = 10.0


def calibrate_variance(
    adata: ad.AnnData,
    target_col: str,
    control_name: str,
    ratio: float,
) -> ad.AnnData:
    """
    Per-gene variance scaling anchored to control cells.

    For each gene g:
        1. Compute target variance from control cells: sigma^2_ctrl(g)
        2. Compute predicted variance from perturbed cells: sigma^2_pred(g)
        3. Compute a scale factor interpolated by `ratio`:
            s(g) = 1 + ratio * ( sqrt(sigma^2_ctrl(g) / sigma^2_pred(g)) - 1 )
        ratio=0 → no change; 
        ratio=1 → full scaling to match control variance.
        4. Shift each perturbed cell's expression toward/away from the gene mean:
            X_calibrated[i, g] = μ_pred(g) + s(g) * (X_pred[i, g] - μ_pred(g))
        5. Clip to [0, \infty) — log-normalised expression cannot be negative.

    Args:
        adata        : AnnData containing both control and predicted perturbed cells.
        target_col   : obs column that identifies perturbation identity.
        control_name : Value in target_col that marks control cells.
        ratio        : Calibration strength in [0, 1].
                       0 = no change; 1 = fully match control variance per gene.
    """
    X = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
    ctrl_mask = (adata.obs[target_col] == control_name).values
    X_ctrl = X[ctrl_mask]
    X_pert = X[~ctrl_mask]

    print(f"Control cells          : {ctrl_mask.sum()}")
    print(f"Perturbed cells        : {(~ctrl_mask).sum()}")

    var_ctrl = X_ctrl.var(axis=0)           # (n_genes,) target variance
    mean_pred = X_pert.mean(axis=0)         # (n_genes,) predicted mean (preserved)
    var_pred = X_pert.var(axis=0)           # (n_genes,) predicted variance

    # Genes with near-zero predicted variance cannot be meaningfully rescaled;
    # the denominator in the scaling factor will be extremely small;
    # their scale factor is capped to avoid numerical explosion.
    with np.errstate(divide='ignore', invalid='ignore'):
        raw_scale = np.where(var_pred > 0, np.sqrt(var_ctrl / var_pred), _MAX_SCALE)
    raw_scale = np.clip(raw_scale, 0.0, _MAX_SCALE) 
    scale = 1.0 + ratio * (raw_scale - 1.0)

    print(f"Variance (before) — mean across genes: {var_pred.mean():.4f}")
    print(f"Variance target   — mean across genes: {var_ctrl.mean():.4f}")
    print(f"Scale factor      — mean={scale.mean():.3f}  "
          f"median={np.median(scale):.3f}  max={scale.max():.3f}")

    # Scale deviations from the predicted mean, then clip to non-negative
    X_pert_calibrated = mean_pred + scale * (X_pert - mean_pred)
    X_pert_calibrated = np.clip(X_pert_calibrated, 0.0, None)

    var_after = X_pert_calibrated.var(axis=0)
    print(f"Variance (after)  — mean across genes: {var_after.mean():.4f}")

    X_new = X.copy()
    X_new[~ctrl_mask] = X_pert_calibrated
    adata.X = X_new
    return adata


def main():
    parser = argparse.ArgumentParser(description="Post-prediction variance calibration for AnnData outputs.")
    parser.add_argument("--input", required=True, help="Path to predicted h5ad.")
    parser.add_argument("--target_col", default="pert", help="obs column for perturbation identity.")
    parser.add_argument("--control_name", default="non-targeting", help="Control label in target_col.")
    parser.add_argument("--ratio", type=float, default=0.8, help="Calibration strength in [0, 1].")
    parser.add_argument("--out_dir", help="Output directory.")
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

    adata = calibrate_variance(
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
