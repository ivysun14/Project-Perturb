# Majority Vote Baseline Results: REPLOGLE_RPE1

**Method**: Per-gene majority label from training

## Dataset Info
- Train samples: 4,625,000
- Test samples: 1,550,000
- Train class distribution: {'not_de': 4143547, 'down': 263131, 'up': 218322}
- Test class distribution: {'not_de': 1396700, 'down': 84033, 'up': 69267}

## Aggregate Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 0.9055 |
| Macro F1 | 0.5006 |
| Macro Precision | 0.6724 |
| Macro Recall | 0.4545 |

## Per-Class Metrics
| Class | F1 | Precision | Recall |
|-------|-----|-----------|--------|
| up | 0.1950 | 0.5331 | 0.1193 |
| down | 0.3572 | 0.5659 | 0.2610 |
| not_de | 0.9496 | 0.9181 | 0.9833 |

## LogFC Regression
- MAE: 0.1099
- NMAE: 0.5722

## Confusion Matrix
(Rows=True, Cols=Pred: up, down, not_de)

| | up | down | not_de |
|---|---|---|---|
| up | 8263 | 160 | 60844 |
| down | 518 | 21930 | 61585 |
| not_de | 6719 | 16660 | 1373321 |
