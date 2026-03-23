# pert_neighbor_esm2 Results: ADAMSON

## Aggregate Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 0.8831 |
| Macro F1 | 0.7729 |
| Macro Precision | 0.8073 |
| Macro Recall | 0.7449 |

## Per-Class Metrics
| Class | Accuracy | F1 | Precision | Recall | Support |
|-------|----------|-----|-----------|--------|--------|
| up | 0.5920 | 0.6391 | 0.6943 | 0.5920 | 11,247 |
| down | 0.6914 | 0.7475 | 0.8136 | 0.6914 | 8,330 |
| not_de | 0.9512 | 0.9323 | 0.9140 | 0.9512 | 71,503 |

## Confusion Matrix
| | Pred: up | Pred: down | Pred: not_de |
|---|---|---|---|
| True: up | 6,658 | 314 | 4,275 |
| True: down | 449 | 5,759 | 2,122 |
| True: not_de | 2,483 | 1,005 | 68,015 |
