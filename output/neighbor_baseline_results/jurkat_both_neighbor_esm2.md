# both_neighbor_esm2 Results: JURKAT

## Aggregate Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 0.9558 |
| Macro F1 | 0.3950 |
| Macro Precision | 0.3976 |
| Macro Recall | 0.3927 |

## Per-Class Metrics
| Class | Accuracy | F1 | Precision | Recall | Support |
|-------|----------|-----|-----------|--------|--------|
| up | 0.1129 | 0.1167 | 0.1209 | 0.1129 | 34,573 |
| down | 0.0866 | 0.0908 | 0.0956 | 0.0866 | 75,595 |
| not_de | 0.9786 | 0.9775 | 0.9764 | 0.9786 | 4,152,712 |

## Confusion Matrix
| | Pred: up | Pred: down | Pred: not_de |
|---|---|---|---|
| True: up | 3,903 | 742 | 29,928 |
| True: down | 909 | 6,544 | 68,142 |
| True: not_de | 27,484 | 61,188 | 4,064,040 |
