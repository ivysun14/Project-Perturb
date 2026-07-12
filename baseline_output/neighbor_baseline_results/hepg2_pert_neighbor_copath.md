# pert_neighbor_copath Results: HEPG2

## Aggregate Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 0.9548 |
| Macro F1 | 0.5170 |
| Macro Precision | 0.5776 |
| Macro Recall | 0.4816 |

## Per-Class Metrics
| Class | Accuracy | F1 | Precision | Recall | Support |
|-------|----------|-----|-----------|--------|--------|
| up | 0.2102 | 0.2647 | 0.3572 | 0.2102 | 76,257 |
| down | 0.2500 | 0.3096 | 0.4066 | 0.2500 | 108,058 |
| not_de | 0.9847 | 0.9767 | 0.9688 | 0.9847 | 4,434,725 |

## Confusion Matrix
| | Pred: up | Pred: down | Pred: not_de |
|---|---|---|---|
| True: up | 16,030 | 288 | 59,939 |
| True: down | 337 | 27,010 | 80,711 |
| True: not_de | 28,505 | 39,126 | 4,367,094 |
