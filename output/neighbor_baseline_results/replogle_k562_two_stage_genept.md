# two_stage_genept Results: REPLOGLE_K562

## Aggregate Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 0.9036 |
| Macro F1 | 0.4297 |
| Macro Precision | 0.5730 |
| Macro Recall | 0.4001 |

## Per-Class Metrics
| Class | Accuracy | F1 | Precision | Recall | Support |
|-------|----------|-----|-----------|--------|--------|
| up | 0.1072 | 0.1650 | 0.3581 | 0.1072 | 43,365 |
| down | 0.1089 | 0.1750 | 0.4447 | 0.1089 | 57,403 |
| not_de | 0.9842 | 0.9490 | 0.9163 | 0.9842 | 994,013 |

## Confusion Matrix
| | Pred: up | Pred: down | Pred: not_de |
|---|---|---|---|
| True: up | 4,649 | 232 | 38,484 |
| True: down | 226 | 6,254 | 50,923 |
| True: not_de | 8,107 | 7,577 | 978,329 |
