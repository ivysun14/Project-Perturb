# majority Results: REPLOGLE_RPE1

## Aggregate Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 0.9055 |
| Macro F1 | 0.5006 |
| Macro Precision | 0.6724 |
| Macro Recall | 0.4545 |

## Per-Class Metrics
| Class | Accuracy | F1 | Precision | Recall | Support |
|-------|----------|-----|-----------|--------|--------|
| up | 0.1193 | 0.1950 | 0.5331 | 0.1193 | 69,267 |
| down | 0.2610 | 0.3572 | 0.5659 | 0.2610 | 84,033 |
| not_de | 0.9833 | 0.9496 | 0.9181 | 0.9833 | 1,396,700 |

## Confusion Matrix
| | Pred: up | Pred: down | Pred: not_de |
|---|---|---|---|
| True: up | 8,263 | 160 | 60,844 |
| True: down | 518 | 21,930 | 61,585 |
| True: not_de | 6,719 | 16,660 | 1,373,321 |
