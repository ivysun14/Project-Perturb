# both_neighbor_genept Results: TCELL_REST

## Aggregate Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 0.9927 |
| Macro F1 | 0.3347 |
| Macro Precision | 0.3364 |
| Macro Recall | 0.3345 |

## Per-Class Metrics
| Class | Accuracy | F1 | Precision | Recall | Support |
|-------|----------|-----|-----------|--------|--------|
| up | 0.0024 | 0.0036 | 0.0067 | 0.0024 | 74,465 |
| down | 0.0029 | 0.0042 | 0.0077 | 0.0029 | 45,323 |
| not_de | 0.9981 | 0.9963 | 0.9947 | 0.9981 | 22,181,833 |

## Confusion Matrix
| | Pred: up | Pred: down | Pred: not_de |
|---|---|---|---|
| True: up | 182 | 225 | 74,058 |
| True: down | 195 | 131 | 44,997 |
| True: not_de | 26,609 | 16,627 | 22,138,597 |
