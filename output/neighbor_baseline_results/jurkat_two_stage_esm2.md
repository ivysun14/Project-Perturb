# two_stage_esm2 Results: JURKAT

## Aggregate Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 0.9490 |
| Macro F1 | 0.4009 |
| Macro Precision | 0.3943 |
| Macro Recall | 0.4094 |

## Per-Class Metrics
| Class | Accuracy | F1 | Precision | Recall | Support |
|-------|----------|-----|-----------|--------|--------|
| up | 0.1416 | 0.1249 | 0.1118 | 0.1416 | 34,573 |
| down | 0.1159 | 0.1038 | 0.0940 | 0.1159 | 75,595 |
| not_de | 0.9709 | 0.9740 | 0.9772 | 0.9709 | 4,152,712 |

## Confusion Matrix
| | Pred: up | Pred: down | Pred: not_de |
|---|---|---|---|
| True: up | 4,896 | 1,016 | 28,661 |
| True: down | 1,315 | 8,758 | 65,522 |
| True: not_de | 37,587 | 83,392 | 4,031,733 |
