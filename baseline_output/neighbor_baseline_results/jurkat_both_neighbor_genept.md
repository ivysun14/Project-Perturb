# both_neighbor_genept Results: JURKAT

## Aggregate Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 0.9669 |
| Macro F1 | 0.3962 |
| Macro Precision | 0.4412 |
| Macro Recall | 0.3788 |

## Per-Class Metrics
| Class | Accuracy | F1 | Precision | Recall | Support |
|-------|----------|-----|-----------|--------|--------|
| up | 0.0781 | 0.1111 | 0.1927 | 0.0781 | 34,573 |
| down | 0.0677 | 0.0943 | 0.1553 | 0.0677 | 75,595 |
| not_de | 0.9906 | 0.9831 | 0.9758 | 0.9906 | 4,152,712 |

## Confusion Matrix
| | Pred: up | Pred: down | Pred: not_de |
|---|---|---|---|
| True: up | 2,699 | 108 | 31,766 |
| True: down | 106 | 5,118 | 70,371 |
| True: not_de | 11,204 | 27,734 | 4,113,774 |
