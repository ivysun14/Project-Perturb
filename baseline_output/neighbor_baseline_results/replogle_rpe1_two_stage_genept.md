# two_stage_genept Results: REPLOGLE_RPE1

## Aggregate Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 0.9014 |
| Macro F1 | 0.5378 |
| Macro Precision | 0.6336 |
| Macro Recall | 0.4946 |

## Per-Class Metrics
| Class | Accuracy | F1 | Precision | Recall | Support |
|-------|----------|-----|-----------|--------|--------|
| up | 0.1910 | 0.2698 | 0.4592 | 0.1910 | 69,267 |
| down | 0.3213 | 0.3966 | 0.5178 | 0.3213 | 84,033 |
| not_de | 0.9715 | 0.9470 | 0.9237 | 0.9715 | 1,396,700 |

## Confusion Matrix
| | Pred: up | Pred: down | Pred: not_de |
|---|---|---|---|
| True: up | 13,231 | 342 | 55,694 |
| True: down | 620 | 27,003 | 56,410 |
| True: not_de | 14,962 | 24,807 | 1,356,931 |
