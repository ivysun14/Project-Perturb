# pert_neighbor_genept Results: REPLOGLE_K562

## Aggregate Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 0.9031 |
| Macro F1 | 0.4151 |
| Macro Precision | 0.5581 |
| Macro Recall | 0.3899 |

## Per-Class Metrics
| Class | Accuracy | F1 | Precision | Recall | Support |
|-------|----------|-----|-----------|--------|--------|
| up | 0.0863 | 0.1366 | 0.3270 | 0.0863 | 43,365 |
| down | 0.0981 | 0.1599 | 0.4322 | 0.0981 | 57,403 |
| not_de | 0.9853 | 0.9488 | 0.9150 | 0.9853 | 994,013 |

## Confusion Matrix
| | Pred: up | Pred: down | Pred: not_de |
|---|---|---|---|
| True: up | 3,744 | 229 | 39,392 |
| True: down | 224 | 5,633 | 51,546 |
| True: not_de | 7,483 | 7,170 | 979,360 |
