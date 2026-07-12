# pert_neighbor_copath Results: JURKAT

## Aggregate Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 0.9637 |
| Macro F1 | 0.4531 |
| Macro Precision | 0.4740 |
| Macro Recall | 0.4379 |

## Per-Class Metrics
| Class | Accuracy | F1 | Precision | Recall | Support |
|-------|----------|-----|-----------|--------|--------|
| up | 0.1546 | 0.1796 | 0.2143 | 0.1546 | 34,573 |
| down | 0.1742 | 0.1980 | 0.2295 | 0.1742 | 75,595 |
| not_de | 0.9848 | 0.9816 | 0.9783 | 0.9848 | 4,152,712 |

## Confusion Matrix
| | Pred: up | Pred: down | Pred: not_de |
|---|---|---|---|
| True: up | 5,346 | 519 | 28,708 |
| True: down | 383 | 13,169 | 62,043 |
| True: not_de | 19,217 | 43,704 | 4,089,791 |
