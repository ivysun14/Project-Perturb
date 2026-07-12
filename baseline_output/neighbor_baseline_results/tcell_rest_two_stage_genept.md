# two_stage_genept Results: TCELL_REST

## Aggregate Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 0.9924 |
| Macro F1 | 0.3349 |
| Macro Precision | 0.3363 |
| Macro Recall | 0.3346 |

## Per-Class Metrics
| Class | Accuracy | F1 | Precision | Recall | Support |
|-------|----------|-----|-----------|--------|--------|
| up | 0.0027 | 0.0038 | 0.0065 | 0.0027 | 74,465 |
| down | 0.0034 | 0.0048 | 0.0078 | 0.0034 | 45,323 |
| not_de | 0.9977 | 0.9962 | 0.9947 | 0.9977 | 22,181,833 |

## Confusion Matrix
| | Pred: up | Pred: down | Pred: not_de |
|---|---|---|---|
| True: up | 202 | 227 | 74,036 |
| True: down | 198 | 155 | 44,970 |
| True: not_de | 30,539 | 19,442 | 22,131,852 |
