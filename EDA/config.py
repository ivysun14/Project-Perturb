"""
config.py
=========
Shared dataset configurations for all data exploration scripts.
"""

import os
_DATA_DIR = "/home/jsun140/scratchyiqunchen/project_perturb_data"

# Gene universe shared across all models; rows are in the same order as STATE h5ad columns.
GENE_UNIVERSE_CSV = os.path.join(_DATA_DIR, 'common_gene_universe.csv')

# Per-dataset list of test+calib perturbations to drop before cross-cell-line evaluation.
DROPPED_PERTS_JSON = os.path.join(_DATA_DIR, 'dropped_test_perts.json')

DATASETS = {
    'k562': {
        'h5ad':        os.path.join(_DATA_DIR, 'h5ad_objects', 'K562_raw.h5ad'),
        'de_csv':      os.path.join(_DATA_DIR, 'csv', 'DE_DOC_K562_splitted.csv'),
        'target_col':  'gene',
        'control_key': 'non-targeting',
        'gene_name_col': 'gene_name',
        'display_name': 'K562',
    },
    'hepg2': {
        'h5ad':        os.path.join(_DATA_DIR, 'h5ad_objects', 'HepG2_raw.h5ad'),
        'de_csv':      os.path.join(_DATA_DIR, 'csv', 'DE_DOC_HepG2_splitted.csv'),
        'target_col':  'gene',
        'control_key': 'non-targeting',
        'gene_name_col': 'gene_name',
        'display_name': 'HepG2',
    },
    'jurkat': {
        'h5ad':        os.path.join(_DATA_DIR, 'h5ad_objects', 'Jurkat_raw.h5ad'),
        'de_csv':      os.path.join(_DATA_DIR, 'csv', 'DE_DOC_Jurkat_splitted.csv'),
        'target_col':  'gene',
        'control_key': 'non-targeting',
        'gene_name_col': 'gene_name',
        'display_name': 'Jurkat',
    },
    'rpe1': {
        'h5ad':        os.path.join(_DATA_DIR, 'h5ad_objects', 'RPE1_raw.h5ad'),
        'de_csv':      os.path.join(_DATA_DIR, 'csv', 'DE_DOC_RPE1_splitted.csv'),
        'target_col':  'gene',
        'control_key': 'non-targeting',
        'gene_name_col': 'gene_name',
        'display_name': 'RPE1',
    },
}

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')

DATASET_COLORS = {
    'k562':   '#E64B35',
    'hepg2':  '#4DBBD5',
    'jurkat': '#00A087',
    'rpe1':   '#3C5488',
}

_PROJECT_ROOT = '/home/jsun140/scratchyiqunchen/jsun140/Project-Perturb'

# Pre-calibration prediction h5ad paths (pred and real)
PRED_PATHS = {
    'Scouter': {
        ds: {
            'pred': (
                os.path.join(_PROJECT_ROOT, f'Scouter/prediction/{ds}/{ds}_calib_pred.h5ad'),
                os.path.join(_PROJECT_ROOT, f'Scouter/prediction/{ds}/{ds}_test_pred.h5ad'),
            ),
            'real': (
                os.path.join(_PROJECT_ROOT, f'Scouter/prediction/{ds}/{ds}_calib_real.h5ad'),
                os.path.join(_PROJECT_ROOT, f'Scouter/prediction/{ds}/{ds}_test_real.h5ad'),
            ),
        }
        for ds in ['k562', 'rpe1', 'hepg2', 'jurkat']
    },
    'TranScouter': {
        ds: {
            'pred': os.path.join(_PROJECT_ROOT, f'TranScouter/prediction/{ds}/{ds}_testcalib_pred.h5ad'),
            'real': os.path.join(_PROJECT_ROOT, f'TranScouter/prediction/{ds}/{ds}_testcalib_real.h5ad'),
        }
        for ds in ['k562', 'rpe1', 'hepg2', 'jurkat']
    },
    'STATE': {
        ds: {
            'pred': os.path.join(_PROJECT_ROOT, f'STATE/results/{ds}/eval_task1_merged.ckpt/adata_pred.h5ad'),
            'real': os.path.join(_PROJECT_ROOT, f'STATE/results/{ds}/eval_task1_merged.ckpt/adata_real.h5ad'),
        }
        for ds in ['k562', 'rpe1', 'hepg2', 'jurkat']
    },
}

# Post-calibration (sparsity) prediction h5ad paths
CALIB_PATHS = {
    model: {
        ds: os.path.join(_PROJECT_ROOT, f'calibration/sparsity_calibrated/{model}/{ds}_testcalib_pred.h5ad')
        for ds in ['k562', 'rpe1', 'hepg2', 'jurkat']
    }
    for model in ['Scouter', 'TranScouter', 'STATE']
}


def load_scouter_pred(ds, kind='pred'):
    """
    Load and concatenate Scouter calib + test h5ads into one AnnData.

    Test split is filtered to non-control cells before concatenation, matching
    the convention in scouter_metric_full.py.
    """
    import anndata as ad
    import scanpy as sc
    calib_path, test_path = PRED_PATHS['Scouter'][ds][kind]
    calib_ad = sc.read_h5ad(calib_path)
    test_ad  = sc.read_h5ad(test_path)
    test_perts = test_ad[test_ad.obs['gene'] != 'ctrl']
    combined = ad.concat([calib_ad, test_perts], join='inner')
    combined.obs_names_make_unique()
    return combined
