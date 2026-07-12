# pert_neighbor_copath Results: REPLOGLE_K562

## Aggregate Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 0.8941 |
| Macro F1 | 0.5531 |
| Macro Precision | 0.5768 |
| Macro Recall | 0.5344 |

## Per-Class Metrics
| Class | Accuracy | F1 | Precision | Recall | Support |
|-------|----------|-----|-----------|--------|--------|
| up | 0.3083 | 0.3430 | 0.3866 | 0.3083 | 43,365 |
| down | 0.3433 | 0.3734 | 0.4093 | 0.3433 | 57,403 |
| not_de | 0.9515 | 0.9429 | 0.9345 | 0.9515 | 994,013 |

## Confusion Matrix
| | Pred: up | Pred: down | Pred: not_de |
|---|---|---|---|
| True: up | 13,368 | 595 | 29,402 |
| True: down | 844 | 19,706 | 36,853 |
| True: not_de | 20,362 | 27,845 | 945,806 |
