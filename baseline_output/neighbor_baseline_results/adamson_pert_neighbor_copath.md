# pert_neighbor_copath Results: ADAMSON

## Aggregate Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 0.8804 |
| Macro F1 | 0.7671 |
| Macro Precision | 0.8033 |
| Macro Recall | 0.7379 |

## Per-Class Metrics
| Class | Accuracy | F1 | Precision | Recall | Support |
|-------|----------|-----|-----------|--------|--------|
| up | 0.5786 | 0.6284 | 0.6874 | 0.5786 | 11,247 |
| down | 0.6843 | 0.7423 | 0.8112 | 0.6843 | 8,330 |
| not_de | 0.9507 | 0.9306 | 0.9114 | 0.9507 | 71,503 |

## Confusion Matrix
| | Pred: up | Pred: down | Pred: not_de |
|---|---|---|---|
| True: up | 6,508 | 316 | 4,423 |
| True: down | 445 | 5,700 | 2,185 |
| True: not_de | 2,514 | 1,011 | 67,978 |
