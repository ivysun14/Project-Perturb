# both_neighbor_esm2 Results: REPLOGLE_RPE1

## Aggregate Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 0.8772 |
| Macro F1 | 0.5407 |
| Macro Precision | 0.5459 |
| Macro Recall | 0.5361 |

## Per-Class Metrics
| Class | Accuracy | F1 | Precision | Recall | Support |
|-------|----------|-----|-----------|--------|--------|
| up | 0.2698 | 0.2848 | 0.3017 | 0.2698 | 69,267 |
| down | 0.4028 | 0.4041 | 0.4053 | 0.4028 | 84,033 |
| not_de | 0.9358 | 0.9332 | 0.9306 | 0.9358 | 1,396,700 |

## Confusion Matrix
| | Pred: up | Pred: down | Pred: not_de |
|---|---|---|---|
| True: up | 18,685 | 1,798 | 48,784 |
| True: down | 1,472 | 33,850 | 48,711 |
| True: not_de | 41,783 | 47,866 | 1,307,051 |
