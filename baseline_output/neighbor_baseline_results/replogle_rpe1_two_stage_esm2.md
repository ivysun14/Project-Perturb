# two_stage_esm2 Results: REPLOGLE_RPE1

## Aggregate Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 0.8670 |
| Macro F1 | 0.5432 |
| Macro Precision | 0.5317 |
| Macro Recall | 0.5567 |

## Per-Class Metrics
| Class | Accuracy | F1 | Precision | Recall | Support |
|-------|----------|-----|-----------|--------|--------|
| up | 0.3089 | 0.2933 | 0.2792 | 0.3089 | 69,267 |
| down | 0.4408 | 0.4093 | 0.3820 | 0.4408 | 84,033 |
| not_de | 0.9204 | 0.9271 | 0.9339 | 0.9204 | 1,396,700 |

## Confusion Matrix
| | Pred: up | Pred: down | Pred: not_de |
|---|---|---|---|
| True: up | 21,400 | 2,128 | 45,739 |
| True: down | 1,791 | 37,044 | 45,198 |
| True: not_de | 53,455 | 57,792 | 1,285,453 |
