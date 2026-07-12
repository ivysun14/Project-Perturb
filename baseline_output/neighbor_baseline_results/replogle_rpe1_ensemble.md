# ensemble Results: REPLOGLE_RPE1

## Aggregate Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 0.8871 |
| Macro F1 | 0.5795 |
| Macro Precision | 0.5859 |
| Macro Recall | 0.5738 |

## Per-Class Metrics
| Class | Accuracy | F1 | Precision | Recall | Support |
|-------|----------|-----|-----------|--------|--------|
| up | 0.3322 | 0.3508 | 0.3715 | 0.3322 | 69,267 |
| down | 0.4482 | 0.4493 | 0.4504 | 0.4482 | 84,033 |
| not_de | 0.9410 | 0.9384 | 0.9358 | 0.9410 | 1,396,700 |

## Confusion Matrix
| | Pred: up | Pred: down | Pred: not_de |
|---|---|---|---|
| True: up | 23,013 | 1,232 | 45,022 |
| True: down | 1,274 | 37,667 | 45,092 |
| True: not_de | 37,658 | 44,725 | 1,314,317 |
