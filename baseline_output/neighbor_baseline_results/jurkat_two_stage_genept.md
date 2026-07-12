# two_stage_genept Results: JURKAT

## Aggregate Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 0.9670 |
| Macro F1 | 0.4053 |
| Macro Precision | 0.4530 |
| Macro Recall | 0.3859 |

## Per-Class Metrics
| Class | Accuracy | F1 | Precision | Recall | Support |
|-------|----------|-----|-----------|--------|--------|
| up | 0.0807 | 0.1142 | 0.1955 | 0.0807 | 34,573 |
| down | 0.0866 | 0.1185 | 0.1872 | 0.0866 | 75,595 |
| not_de | 0.9904 | 0.9832 | 0.9761 | 0.9904 | 4,152,712 |

## Confusion Matrix
| | Pred: up | Pred: down | Pred: not_de |
|---|---|---|---|
| True: up | 2,789 | 123 | 31,661 |
| True: down | 116 | 6,550 | 68,929 |
| True: not_de | 11,358 | 28,319 | 4,113,035 |
