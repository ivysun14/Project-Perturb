# pert_neighbor_esm2 Results: HEPG2

## Aggregate Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 0.9408 |
| Macro F1 | 0.4607 |
| Macro Precision | 0.4684 |
| Macro Recall | 0.4541 |

## Per-Class Metrics
| Class | Accuracy | F1 | Precision | Recall | Support |
|-------|----------|-----|-----------|--------|--------|
| up | 0.1934 | 0.1991 | 0.2052 | 0.1934 | 76,257 |
| down | 0.1970 | 0.2134 | 0.2328 | 0.1970 | 108,058 |
| not_de | 0.9718 | 0.9695 | 0.9672 | 0.9718 | 4,434,725 |

## Confusion Matrix
| | Pred: up | Pred: down | Pred: not_de |
|---|---|---|---|
| True: up | 14,748 | 1,239 | 60,270 |
| True: down | 1,016 | 21,290 | 85,752 |
| True: not_de | 56,118 | 68,923 | 4,309,684 |
