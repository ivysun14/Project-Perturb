# pert_neighbor_copath Results: TCELL_REST

## Aggregate Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 0.9928 |
| Macro F1 | 0.3532 |
| Macro Precision | 0.3692 |
| Macro Recall | 0.3474 |

## Per-Class Metrics
| Class | Accuracy | F1 | Precision | Recall | Support |
|-------|----------|-----|-----------|--------|--------|
| up | 0.0235 | 0.0338 | 0.0601 | 0.0235 | 74,465 |
| down | 0.0205 | 0.0295 | 0.0528 | 0.0205 | 45,323 |
| not_de | 0.9980 | 0.9964 | 0.9948 | 0.9980 | 22,181,833 |

## Confusion Matrix
| | Pred: up | Pred: down | Pred: not_de |
|---|---|---|---|
| True: up | 1,751 | 115 | 72,599 |
| True: down | 159 | 930 | 44,234 |
| True: not_de | 27,217 | 16,579 | 22,138,037 |
