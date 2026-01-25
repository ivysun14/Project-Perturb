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
python -u DE_DOC_stats.py \
  --adata "PATH/TO/ADATA.h5ad" \
  --output "PATH/TO/SAVE/OUTPUT.CSV" \
  --alpha 0.01 \
  --log2fc_threshold 0.00
```

#### 2. `train_test_split.py`
Perform a 75/25 percent train test split on outputs from `DE_DOC_stats.py`. The split is based on **perturbations** and ensure label proportions are balanced between train and test set. [Currently have to manually change inputs in the actual script.]
```bash
python -u train_test_split.py
```

#### 3. `majority_perturbQA_DE_file.py` - PerturbQA Datasets
For use with PerturbQA pre-split DE and DOC files. Train a majority vote predictor using the pre-split 75% training set, then evaluate on the 25% testing set.
```bash
python -u majority_perturbQA_DE_file.py \
    --de_file "PerturbQA/perturbqa/datasets/hepg2-de.csv" \
    --doc_file "PerturbQA/perturbqa/datasets/hepg2-dir.csv" \
    --dataset_name hepg2 \
    --n_runs 3 \
    --train_fraction 0.8
```

#### 4. `majority_DE_DOC_file.py` - Custom DE Statistics
For use with combined DE+DOC files from `DE_DOC_stats.py` and have gone through train test split using `train_test_split.py`. Train a majority vote predictor on the pre-split training set and evaluate on the test set.
```bash
python -u majority_DE_DOC_file.py \
    --de_doc_file "PATH/TO/DE_DOC_FILE.csv" \
    --dataset_name "THIS_WOULD_BE_THE_PREFIX_FOR_ALL_OUTPUTS"
    --test_size 0.25 \
    --n_runs 3 \
    --train_fraction 0.8 \
    --random_seed 305
```

#### 5. `analysis_utility.py`
Utility functions for model training and DE/DOC label updating.

- `update_DE_DOC_labels`: update labels (0/1/2) for perturbation-gene pairs using updated alpha and log2FC thresholds
- `filter_perturbqa_style`: filter out noisy perturbation-gene pairs based on workflow proposed in PerturbQA, accepts files outputted from `DE_DOC_stats.py` as input

#### 6. `error_utility.py`
Utility class for error analysis.