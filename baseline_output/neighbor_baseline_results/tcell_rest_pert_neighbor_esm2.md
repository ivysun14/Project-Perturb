# pert_neighbor_esm2 Results: TCELL_REST

## Aggregate Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 0.9917 |
| Macro F1 | 0.3344 |
| Macro Precision | 0.3349 |
| Macro Recall | 0.3342 |

## Per-Class Metrics
| Class | Accuracy | F1 | Precision | Recall | Support |
|-------|----------|-----|-----------|--------|--------|
| up | 0.0028 | 0.0037 | 0.0052 | 0.0028 | 74,465 |
| down | 0.0028 | 0.0036 | 0.0050 | 0.0028 | 45,323 |
| not_de | 0.9970 | 0.9958 | 0.9946 | 0.9970 | 22,181,833 |

## Confusion Matrix
| | Pred: up | Pred: down | Pred: not_de |
|---|---|---|---|
| True: up | 210 | 126 | 74,129 |
| True: down | 112 | 129 | 45,082 |
| True: not_de | 39,980 | 25,714 | 22,116,139 |
