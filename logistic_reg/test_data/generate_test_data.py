"""
generate_test_data.py
=====================
Generates small synthetic data for tracing logistic_reg.py.

Design
------
4 cell lines (CellA/B/C/D), 10 perturbations each, 5 shared genes.
Split per cell line: P1-P6 = train, P7-P8 = calibration, P9-P10 = test.

Code paths exercised
--------------------
- G5 is always class 0 in training  → stored as constant predictor
- G6 exists only in CellD           → unseen gene in cross-cell-line mode
                                       (tests class-0 default + no_model_genes JSON)
- G1-G4 have all 3 classes          → normal LogisticRegression fit
- Embeddings are 4-dimensional      → fast; use --Cs 1.0 to skip CV

Outputs (all in this directory)
-------------------------------
  CellA.csv, CellB.csv, CellC.csv, CellD.csv
  test_embeddings.pkl

Example commands (run from logistic_reg/)
-----------------------------------------
Within-cell-line (CellA):
  python logistic_reg.py \
    --de_doc_file      test_data/CellA.csv \
    --embedding_file   test_data/test_embeddings.pkl \
    --dataset_name     CellA_test \
    --output_dir       test_data/results \
    --Cs 1.0

Cross-cell-line (held-out: CellD, trained on A+B+C):
  python logistic_reg.py \
    --train_de_doc_files test_data/CellA.csv,test_data/CellB.csv,test_data/CellC.csv \
    --test_de_doc_file   test_data/CellD.csv \
    --embedding_file     test_data/test_embeddings.pkl \
    --dataset_name       CellD_transfer_test \
    --output_dir         test_data/results \
    --Cs 1.0
"""

import os
import pickle
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
os.makedirs(os.path.dirname(__file__) or '.', exist_ok=True)

EMBED_DIM    = 4
SHARED_GENES = ['G1', 'G2', 'G3', 'G4', 'G5']

# Fixed label patterns (index = perturbation index 0-9)
# G5 is all-0 in training (indices 0-5) → constant predictor
GENE_LABELS = {
    'G1': [0, 1, 2, 0, 1, 2, 0, 1, 2, 0],   # cycling through all classes
    'G2': [1, 1, 0, 0, 2, 2, 1, 0, 2, 1],   # mixed
    'G3': [2, 2, 2, 1, 0, 0, 2, 1, 0, 2],   # mostly down-regulated
    'G4': [0, 0, 1, 2, 0, 1, 0, 2, 1, 0],   # mostly non-DE
    'G5': [0, 0, 0, 0, 0, 0, 1, 0, 2, 0],   # train-only class 0 → constant predictor
}
G6_LABELS    = [0, 1, 2, 1, 0, 2, 1, 0, 2, 1]   # CellD-only gene


def make_cell_df(cell_name: str, perts: list, genes: list, label_map: dict) -> pd.DataFrame:
    """Build full (pert x gene) DataFrame with split_calib column."""
    n       = len(perts)
    n_train = round(n * 0.6)
    n_calib = round(n * 0.2)
    split_map = {}
    for i, p in enumerate(perts):
        if i < n_train:
            split_map[p] = 'train'
        elif i < n_train + n_calib:
            split_map[p] = 'calibration'
        else:
            split_map[p] = 'test'

    rows = []
    for p in perts:
        for g in genes:
            lbl = label_map[g][perts.index(p)]
            rows.append({
                'pert':       p,
                'gene':       g,
                'label':      lbl,
                'log2FC':     rng.uniform(-3.0, 3.0),
                'p_value':    rng.uniform(0.0, 0.05) if lbl != 0 else rng.uniform(0.05, 1.0),
                'split_calib': split_map[p],
            })
    return pd.DataFrame(rows)


out_dir = os.path.dirname(os.path.abspath(__file__))

# CellA, CellB, CellC — shared genes only
for cell in ['CellA', 'CellB', 'CellC']:
    perts = [f'{cell}_P{i+1}' for i in range(10)]
    df = make_cell_df(cell, perts, SHARED_GENES, GENE_LABELS)
    path = os.path.join(out_dir, f'{cell}.csv')
    df.to_csv(path, index=False)
    train_n = (df['split_calib'] == 'train').sum()
    calib_n = (df['split_calib'] == 'calibration').sum()
    test_n  = (df['split_calib'] == 'test').sum()
    print(f"Saved {cell}.csv  —  {len(df)} pairs  "
          f"(train={train_n}, calib={calib_n}, test={test_n})")

# CellD — shared genes + G6 (unseen in training → triggers no_model_genes default)
cellD_perts = [f'CellD_P{i+1}' for i in range(10)]
cellD_label_map = {**GENE_LABELS, 'G6': G6_LABELS}
cellD_df = make_cell_df('CellD', cellD_perts, SHARED_GENES + ['G6'], cellD_label_map)
path = os.path.join(out_dir, 'CellD.csv')
cellD_df.to_csv(path, index=False)
train_n = (cellD_df['split_calib'] == 'train').sum()
calib_n = (cellD_df['split_calib'] == 'calibration').sum()
test_n  = (cellD_df['split_calib'] == 'test').sum()
print(f"Saved CellD.csv   —  {len(cellD_df)} pairs  "
      f"(train={train_n}, calib={calib_n}, test={test_n})  [includes G6]")

# Embeddings: 4-dim random vectors for every perturbation
all_perts = [f'{cell}_P{i+1}' for cell in ['CellA', 'CellB', 'CellC', 'CellD']
             for i in range(10)]
embedding_dict = {p: rng.standard_normal(EMBED_DIM).astype(np.float32) for p in all_perts}
emb_path = os.path.join(out_dir, 'test_embeddings.pkl')
with open(emb_path, 'wb') as f:
    pickle.dump(embedding_dict, f)
print(f"Saved test_embeddings.pkl  —  {len(embedding_dict)} perts, dim={EMBED_DIM}")

print("\nWhat to expect")
print("--------------")
print("Within-cell-line (CellA):")
print("  G5 train labels are all 0 → constant predictor (prob=[1,0,0])")
print("  No unseen genes → no_model_genes JSON should be empty")
print()
print("Cross-cell-line (CellD held out):")
print("  G6 not in CellA/B/C training → no model → defaults to class 0, [1/3,1/3,1/3]")
print("  G5 all-0 in training → constant predictor")
print("  no_model_genes JSON should list ['G6']")