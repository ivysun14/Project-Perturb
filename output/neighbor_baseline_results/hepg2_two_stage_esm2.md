# two_stage_esm2 Results: HEPG2

## Aggregate Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 0.9323 |
| Macro F1 | 0.4789 |
| Macro Precision | 0.4645 |
| Macro Recall | 0.4979 |

## Per-Class Metrics
| Class | Accuracy | F1 | Precision | Recall | Support |
|-------|----------|-----|-----------|--------|--------|
| up | 0.2597 | 0.2207 | 0.1918 | 0.2597 | 76,257 |
| down | 0.2742 | 0.2511 | 0.2316 | 0.2742 | 108,058 |
| not_de | 0.9599 | 0.9650 | 0.9702 | 0.9599 | 4,434,725 |

## Confusion Matrix
| | Pred: up | Pred: down | Pred: not_de |
|---|---|---|---|
| True: up | 19,803 | 2,048 | 54,406 |
| True: down | 1,967 | 29,632 | 76,459 |
| True: not_de | 81,464 | 96,256 | 4,257,005 |
