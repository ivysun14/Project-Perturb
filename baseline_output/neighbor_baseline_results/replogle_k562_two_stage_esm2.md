# two_stage_esm2 Results: REPLOGLE_K562

## Aggregate Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 0.8611 |
| Macro F1 | 0.4847 |
| Macro Precision | 0.4771 |
| Macro Recall | 0.4935 |

## Per-Class Metrics
| Class | Accuracy | F1 | Precision | Recall | Support |
|-------|----------|-----|-----------|--------|--------|
| up | 0.2514 | 0.2382 | 0.2264 | 0.2514 | 43,365 |
| down | 0.3097 | 0.2906 | 0.2737 | 0.3097 | 57,403 |
| not_de | 0.9195 | 0.9253 | 0.9311 | 0.9195 | 994,013 |

## Confusion Matrix
| | Pred: up | Pred: down | Pred: not_de |
|---|---|---|---|
| True: up | 10,903 | 1,949 | 30,513 |
| True: down | 2,489 | 17,776 | 37,138 |
| True: not_de | 34,775 | 45,216 | 914,022 |
