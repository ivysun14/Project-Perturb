# ensemble Results: REPLOGLE_K562

## Aggregate Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 0.8859 |
| Macro F1 | 0.5300 |
| Macro Precision | 0.5436 |
| Macro Recall | 0.5184 |

## Per-Class Metrics
| Class | Accuracy | F1 | Precision | Recall | Support |
|-------|----------|-----|-----------|--------|--------|
| up | 0.2841 | 0.3054 | 0.3302 | 0.2841 | 43,365 |
| down | 0.3267 | 0.3461 | 0.3678 | 0.3267 | 57,403 |
| not_de | 0.9445 | 0.9386 | 0.9328 | 0.9445 | 994,013 |

## Confusion Matrix
| | Pred: up | Pred: down | Pred: not_de |
|---|---|---|---|
| True: up | 12,318 | 867 | 30,180 |
| True: down | 1,149 | 18,754 | 37,500 |
| True: not_de | 23,841 | 31,364 | 938,808 |
