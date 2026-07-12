# Majority Vote Baseline Results: TCELL_REST

**Method**: Per-gene majority label from training

## Dataset Info
- Train samples: 67,031,080
- Test samples: 22,301,621
- Train class distribution: {'not_de': 66673173, 'up': 222193, 'down': 135714}
- Test class distribution: {'not_de': 22181833, 'down': 45323, 'up': 74465}

## Aggregate Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 0.9946 |
| Macro F1 | 0.3324 |
| Macro Precision | 0.3315 |
| Macro Recall | 0.3333 |

## Per-Class Metrics
| Class | F1 | Precision | Recall |
|-------|-----|-----------|--------|
| up | 0.0000 | 0.0000 | 0.0000 |
| down | 0.0000 | 0.0000 | 0.0000 |
| not_de | 0.9973 | 0.9946 | 1.0000 |

## LogFC Regression
- MAE: 0.1841
- NMAE: 0.5738

## Confusion Matrix
(Rows=True, Cols=Pred: up, down, not_de)

| | up | down | not_de |
|---|---|---|---|
| up | 0 | 0 | 74465 |
| down | 0 | 0 | 45323 |
| not_de | 0 | 0 | 22181833 |
