# Majority Vote Baseline Results: HEPG2

**Method**: Per-gene majority label from training

## Dataset Info
- Train samples: 13,809,005
- Test samples: 4,619,040
- Train class distribution: {'not_de': 13350589, 'down': 264082, 'up': 194334}
- Test class distribution: {'not_de': 4434725, 'up': 76257, 'down': 108058}

## Aggregate Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 0.9601 |
| Macro F1 | 0.3265 |
| Macro Precision | 0.3200 |
| Macro Recall | 0.3333 |

## Per-Class Metrics
| Class | F1 | Precision | Recall |
|-------|-----|-----------|--------|
| up | 0.0000 | 0.0000 | 0.0000 |
| down | 0.0000 | 0.0000 | 0.0000 |
| not_de | 0.9796 | 0.9601 | 1.0000 |

## LogFC Regression
- MAE: 0.0525
- NMAE: 0.6027

## Confusion Matrix
(Rows=True, Cols=Pred: up, down, not_de)

| | up | down | not_de |
|---|---|---|---|
| up | 0 | 0 | 76257 |
| down | 0 | 0 | 108058 |
| not_de | 0 | 0 | 4434725 |
