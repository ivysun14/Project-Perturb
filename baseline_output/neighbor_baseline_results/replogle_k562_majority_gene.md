# majority_gene Results: REPLOGLE_K562

## Aggregate Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 0.9081 |
| Macro F1 | 0.3331 |
| Macro Precision | 0.6260 |
| Macro Recall | 0.3411 |

## Per-Class Metrics
| Class | Accuracy | F1 | Precision | Recall | Support |
|-------|----------|-----|-----------|--------|--------|
| up | 0.0070 | 0.0138 | 0.4627 | 0.0070 | 43,365 |
| down | 0.0174 | 0.0336 | 0.5063 | 0.0174 | 57,403 |
| not_de | 0.9988 | 0.9518 | 0.9091 | 0.9988 | 994,013 |

## Confusion Matrix
| | Pred: up | Pred: down | Pred: not_de |
|---|---|---|---|
| True: up | 304 | 80 | 42,981 |
| True: down | 64 | 998 | 56,341 |
| True: not_de | 289 | 893 | 992,831 |
