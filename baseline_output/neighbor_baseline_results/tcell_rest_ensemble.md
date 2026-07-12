# ensemble Results: TCELL_REST

## Aggregate Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 0.9940 |
| Macro F1 | 0.3340 |
| Macro Precision | 0.3395 |
| Macro Recall | 0.3341 |

## Per-Class Metrics
| Class | Accuracy | F1 | Precision | Recall | Support |
|-------|----------|-----|-----------|--------|--------|
| up | 0.0015 | 0.0026 | 0.0128 | 0.0015 | 74,465 |
| down | 0.0013 | 0.0024 | 0.0109 | 0.0013 | 45,323 |
| not_de | 0.9994 | 0.9970 | 0.9946 | 0.9994 | 22,181,833 |

## Confusion Matrix
| | Pred: up | Pred: down | Pred: not_de |
|---|---|---|---|
| True: up | 109 | 25 | 74,331 |
| True: down | 16 | 61 | 45,246 |
| True: not_de | 8,393 | 5,490 | 22,167,950 |
