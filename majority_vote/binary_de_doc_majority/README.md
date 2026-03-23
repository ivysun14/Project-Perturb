## Gene-Based Majority Vote Baseline: Two-Step Classification Approach

### Overview

- Step 1 (DE vs non-DE): For each gene, counts how often it appears as DE (label 1 or 2) vs non-DE (label 0) in training data, then uses majority vote to classify
- Step 2 (Up vs Down): For genes classified as DE in Step 1, looks at the majority direction (up=1 vs down=2) among the gene's DE instances in training data

### Error Analysis (separate binary analyses)

- Step 1: Binary error analysis on DE (1) vs non-DE (0) for all test samples
- Step 2: Binary error analysis on Up (1) vs Down (0) for DE samples only

### Usage
```bash
python -u two_step_majority.py \
  --de_doc_file "PATH/TO/DE/DOC/FILE.csv" \
  --dataset_name "DATASET_NAME" \
  --test_size 0.25 \
  --n_runs 3 \
  --train_fraction 0.8 \
  --random_seed 305
```

### Metrics

- 3-class overall metrics (accuracy, balanced accuracy, macro F1)
- Step 1 binary metrics (DE vs non-DE)
- Step 2 binary metrics (Up vs Down, on truly DE samples only)