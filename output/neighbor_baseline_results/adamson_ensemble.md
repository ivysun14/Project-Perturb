# ensemble Results: ADAMSON

## Aggregate Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 0.8803 |
| Macro F1 | 0.7694 |
| Macro Precision | 0.8001 |
| Macro Recall | 0.7438 |

## Per-Class Metrics
| Class | Accuracy | F1 | Precision | Recall | Support |
|-------|----------|-----|-----------|--------|--------|
| up | 0.5908 | 0.6329 | 0.6813 | 0.5908 | 11,247 |
| down | 0.6930 | 0.7450 | 0.8053 | 0.6930 | 8,330 |
| not_de | 0.9477 | 0.9304 | 0.9137 | 0.9477 | 71,503 |

## Confusion Matrix
| | Pred: up | Pred: down | Pred: not_de |
|---|---|---|---|
| True: up | 6,645 | 312 | 4,290 |
| True: down | 449 | 5,773 | 2,108 |
| True: not_de | 2,659 | 1,084 | 67,760 |
