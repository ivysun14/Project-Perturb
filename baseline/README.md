## Gene-Based Majority Vote Baseline for 3-Class Classification

### Overview
Gene-level majority vote predictor for 3-class differential expression problem:
- **Class 0**: Non-DE (not differentially expressed)
- **Class 1**: Up-regulated (DE with positive log2FC)
- **Class 2**: Down-regulated (DE with negative log2FC)

### Scripts

#### 1. `DE_DOC_stats.py`
Compute differential expression statistics and direction of change for perturbation-gene pairs in a given AnnData object.
```bash
python DE_DOC_stats.py \
  --adata "PATH/TO/ADATA.h5ad" \
  --output "PATH/TO/SAVE/OUTPUT.CSV" \
  --alpha 0.01 \
  --log2fc_threshold 0.00
```

#### 2. `majority_perturbQA_DE_file.py` - PerturbQA Datasets
For use with PerturbQA pre-split DE and DOC files:
```bash
python majority_perturbQA_DE_file.py \
    --de_file "PerturbQA/perturbqa/datasets/hepg2-de.csv" \
    --doc_file "PerturbQA/perturbqa/datasets/hepg2-dir.csv" \
    --n_runs 3 \
    --train_fraction 0.8
```

#### 3. `majority_DE_DOC_file.py` - Custom DE Statistics
For use with combined DE+DOC files from `DE_DOC_stats.py`:
```bash
python majority_DE_DOC_file.py \
    --de_doc_file "PATH/TO/DE_DOC_FILE.csv" \
    --test_size 0.25 \
    --n_runs 3 \
    --train_fraction 0.8
```

#### 4. `analysis_utility.py`
Utility functions for model training and DE/DOC label updating.

- `update_DE_DOC_labels`: update labels (0/1/2) for perturbation-gene pairs using updated alpha and log2FC thresholds
- `filter_perturbqa_style`: filter out noisy perturbation-gene pairs based on workflow proposed in PerturbQA, accepts files outputted from `DE_DOC_stats.py` as input