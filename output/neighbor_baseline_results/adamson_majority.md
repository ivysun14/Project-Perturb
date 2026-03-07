# majority Results: ADAMSON

## Aggregate Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 0.8842 |
| Macro F1 | 0.7734 |
| Macro Precision | 0.8115 |
| Macro Recall | 0.7429 |

## Per-Class Metrics
| Class | Accuracy | F1 | Precision | Recall | Support |
|-------|----------|-----|-----------|--------|--------|
| up | 0.5850 | 0.6386 | 0.7030 | 0.5850 | 11,247 |
| down | 0.6898 | 0.7487 | 0.8185 | 0.6898 | 8,330 |
| not_de | 0.9539 | 0.9330 | 0.9130 | 0.9539 | 71,503 |

## Confusion Matrix
| | Pred: up | Pred: down | Pred: not_de |
|---|---|---|---|
| True: up | 6,580 | 318 | 4,349 |
| True: down | 437 | 5,746 | 2,147 |
| True: not_de | 2,343 | 956 | 68,204 |
