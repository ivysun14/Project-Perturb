# pert_neighbor_genept Results: HEPG2

## Aggregate Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 0.9564 |
| Macro F1 | 0.4332 |
| Macro Precision | 0.5535 |
| Macro Recall | 0.4004 |

## Per-Class Metrics
| Class | Accuracy | F1 | Precision | Recall | Support |
|-------|----------|-----|-----------|--------|--------|
| up | 0.1062 | 0.1648 | 0.3672 | 0.1062 | 76,257 |
| down | 0.1032 | 0.1571 | 0.3294 | 0.1032 | 108,058 |
| not_de | 0.9918 | 0.9776 | 0.9639 | 0.9918 | 4,434,725 |

## Confusion Matrix
| | Pred: up | Pred: down | Pred: not_de |
|---|---|---|---|
| True: up | 8,099 | 140 | 68,018 |
| True: down | 94 | 11,150 | 96,814 |
| True: not_de | 13,866 | 22,560 | 4,398,299 |
