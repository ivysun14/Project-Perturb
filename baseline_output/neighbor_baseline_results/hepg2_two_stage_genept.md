# two_stage_genept Results: HEPG2

## Aggregate Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 0.9560 |
| Macro F1 | 0.4361 |
| Macro Precision | 0.5498 |
| Macro Recall | 0.4031 |

## Per-Class Metrics
| Class | Accuracy | F1 | Precision | Recall | Support |
|-------|----------|-----|-----------|--------|--------|
| up | 0.1104 | 0.1690 | 0.3601 | 0.1104 | 76,257 |
| down | 0.1077 | 0.1618 | 0.3254 | 0.1077 | 108,058 |
| not_de | 0.9912 | 0.9774 | 0.9640 | 0.9912 | 4,434,725 |

## Confusion Matrix
| | Pred: up | Pred: down | Pred: not_de |
|---|---|---|---|
| True: up | 8,420 | 143 | 67,694 |
| True: down | 99 | 11,639 | 96,320 |
| True: not_de | 14,864 | 23,985 | 4,395,876 |
