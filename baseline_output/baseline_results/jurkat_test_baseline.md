# Majority Vote Baseline Results: JURKAT

**Method**: Per-gene majority label from training

## Dataset Info
- Train samples: 12,744,235
- Test samples: 4,262,880
- Train class distribution: {'not_de': 12398193, 'down': 241702, 'up': 104340}
- Test class distribution: {'not_de': 4152712, 'down': 75595, 'up': 34573}

## Aggregate Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 0.9742 |
| Macro F1 | 0.3290 |
| Macro Precision | 0.3247 |
| Macro Recall | 0.3333 |

## Per-Class Metrics
| Class | F1 | Precision | Recall |
|-------|-----|-----------|--------|
| up | 0.0000 | 0.0000 | 0.0000 |
| down | 0.0000 | 0.0000 | 0.0000 |
| not_de | 0.9869 | 0.9742 | 1.0000 |

## LogFC Regression
- MAE: 0.0498
- NMAE: 0.6211

## Confusion Matrix
(Rows=True, Cols=Pred: up, down, not_de)

| | up | down | not_de |
|---|---|---|---|
| up | 0 | 0 | 34573 |
| down | 0 | 0 | 75595 |
| not_de | 0 | 0 | 4152712 |
