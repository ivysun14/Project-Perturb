# ensemble Results: HEPG2

## Aggregate Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 0.9535 |
| Macro F1 | 0.5139 |
| Macro Precision | 0.5646 |
| Macro Recall | 0.4825 |

## Per-Class Metrics
| Class | Accuracy | F1 | Precision | Recall | Support |
|-------|----------|-----|-----------|--------|--------|
| up | 0.2115 | 0.2617 | 0.3431 | 0.2115 | 76,257 |
| down | 0.2525 | 0.3040 | 0.3816 | 0.2525 | 108,058 |
| not_de | 0.9833 | 0.9761 | 0.9690 | 0.9833 | 4,434,725 |

## Confusion Matrix
| | Pred: up | Pred: down | Pred: not_de |
|---|---|---|---|
| True: up | 16,131 | 664 | 59,462 |
| True: down | 522 | 27,290 | 80,246 |
| True: not_de | 30,361 | 43,557 | 4,360,807 |
