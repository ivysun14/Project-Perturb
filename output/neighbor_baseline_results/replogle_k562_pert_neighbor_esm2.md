# pert_neighbor_esm2 Results: REPLOGLE_K562

## Aggregate Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 0.8753 |
| Macro F1 | 0.4716 |
| Macro Precision | 0.4849 |
| Macro Recall | 0.4610 |

## Per-Class Metrics
| Class | Accuracy | F1 | Precision | Recall | Support |
|-------|----------|-----|-----------|--------|--------|
| up | 0.1961 | 0.2147 | 0.2374 | 0.1961 | 43,365 |
| down | 0.2456 | 0.2666 | 0.2916 | 0.2456 | 57,403 |
| not_de | 0.9413 | 0.9335 | 0.9258 | 0.9413 | 994,013 |

## Confusion Matrix
| | Pred: up | Pred: down | Pred: not_de |
|---|---|---|---|
| True: up | 8,502 | 1,377 | 33,486 |
| True: down | 1,843 | 14,101 | 41,459 |
| True: not_de | 25,474 | 32,886 | 935,653 |
