# two_stage_esm2 Results: TCELL_REST

## Aggregate Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 0.9879 |
| Macro F1 | 0.3376 |
| Macro Precision | 0.3372 |
| Macro Recall | 0.3382 |

## Per-Class Metrics
| Class | Accuracy | F1 | Precision | Recall | Support |
|-------|----------|-----|-----------|--------|--------|
| up | 0.0104 | 0.0091 | 0.0082 | 0.0104 | 74,465 |
| down | 0.0112 | 0.0097 | 0.0086 | 0.0112 | 45,323 |
| not_de | 0.9932 | 0.9940 | 0.9947 | 0.9932 | 22,181,833 |

## Confusion Matrix
| | Pred: up | Pred: down | Pred: not_de |
|---|---|---|---|
| True: up | 771 | 602 | 73,092 |
| True: down | 718 | 507 | 44,098 |
| True: not_de | 92,813 | 57,954 | 22,031,066 |
