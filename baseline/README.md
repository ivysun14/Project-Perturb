## PerturbQA Testing for DE and DOC
```
import numpy as np
from perturbqa import load_de, load_dir, auc_per_gene

data_de = load_de("rpe1")
X_train = data_de["train"]
X_test = data_de["test"]

len(X_train) # 149147
len(X_test)  # 37942

keys = [(x["pert"], x["gene"]) for x in X_test]
true = [x["label"] for x in X_test]

# 0: 'gene' not differentially expressed upon perturbation 'pert'
# 1: 'gene' differentially expressed upon perturbation 'pert'
np.unique(true)  # array([0, 1])

pred = np.random.randn(len(keys))
auc = auc_per_gene(keys, pred, true)
```

## The Naive Baseline: Majority Vote 
```
python baseline.py \
    --adata "../../VCC2025/vcc_raw_data/rpe1_raw_singlecell_01.h5ad" \
    --dataset_name "rpe1" \
    --control "non-targeting" \
    --target_col "target_gene" \
    --gene_row "gene_name" \
    --alpha 0.01 \
    --n_runs 3
```

1. **Model**: For each gene, learns the majority class from training data (whether that gene is typically DE or not-DE, up or down-regulated across different perturbations)

2. **PerturbQA Evaluation**:

   - Uses pre-defined train/test splits from PerturbQA files
   - Computes AUROC **per gene** (across perturbations for that gene)
   - Averages AUROC across all genes
   - Reports mean ± std over multiple runs with subsampling

3. **Differential Expression Detection**:

   - Use Wilcoxon rank-sum test (Mann-Whitney U test) with Benjamini-Hochberg correction per perturbation
   - 1 = DE, 0 = non-DE

4. **Direction of Change**:

   - Compares mean expression: pert vs control
   - 1 = up-regulated, 0 = down-regulated