import numpy as np
import anndata as ad
from scipy import sparse

def calibrate_zeros(    
    adata: ad.AnnData,
    target_col: str,
    control_name: str,
    RATIO: float = 0.8,
):
    """
    Perform per-gene zero-value calibration for perturbed cells based on the zero-expression ratio observed in control cells.

    For each gene g:

        1. Compute zero ratio in control cells:
               z_g = (# zeros in control cells) / (# control cells)
        2. Define a calibration quantile:
               q_g = z_g * {RATIO}
        3. Compute threshold:
               T_g = quantile(X_pert[:, g], q_g)
        4. Set values below threshold to zero:
               X_pert_calibrated[i, g] = 0 
                   if X_pert[i, g] <= T_g

    Important args:
        RATIO : Scaling factor for zero calibration strength.
    """
    X = adata.X
    if sparse.issparse(X):
        X = X.toarray()
    ctrl_mask = adata.obs[target_col] == control_name
    X_ctrl = X[ctrl_mask]
    X_pert = X[~ctrl_mask]

    n_genes = X.shape[1]
    zero_ratios = (X_ctrl == 0).mean(axis=0)

    X_pert_calibrated = X_pert.copy()
    Q = 0
    for g in range(n_genes):
        q = zero_ratios[g] * RATIO
        threshold = np.quantile(X_pert[:, g], q)
        X_pert_calibrated[:, g] = np.where(
            X_pert[:, g] <= threshold,
            0,
            X_pert[:, g],
        )
        Q += zero_ratios[g]

    print(f"Average zero expression ratio in control data: {Q/n_genes}")

    X_new = X.copy()
    X_new[~ctrl_mask] = X_pert_calibrated
    adata.X = X_new
    return adata