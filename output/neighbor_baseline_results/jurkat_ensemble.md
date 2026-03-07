# ensemble Results: JURKAT

## Aggregate Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 0.9611 |
| Macro F1 | 0.4262 |
| Macro Precision | 0.4424 |
| Macro Recall | 0.4150 |

## Per-Class Metrics
| Class | Accuracy | F1 | Precision | Recall | Support |
|-------|----------|-----|-----------|--------|--------|
| up | 0.1176 | 0.1421 | 0.1797 | 0.1176 | 34,573 |
| down | 0.1443 | 0.1561 | 0.1700 | 0.1443 | 75,595 |
| not_de | 0.9830 | 0.9802 | 0.9775 | 0.9830 | 4,152,712 |

## Confusion Matrix
| | Pred: up | Pred: down | Pred: not_de |
|---|---|---|---|
| True: up | 4,065 | 661 | 29,847 |
| True: down | 499 | 10,911 | 64,185 |
| True: not_de | 18,059 | 52,596 | 4,082,057 |
