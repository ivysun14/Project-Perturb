# Majority Vote Baseline Results: ADAMSON

**Method**: Per-gene majority label from training

## Dataset Info
- Train samples: 258,060
- Test samples: 91,080
- Train class distribution: {'not_de': 203085, 'down': 25184, 'up': 29791}
- Test class distribution: {'not_de': 71503, 'up': 11247, 'down': 8330}

## Aggregate Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 0.8842 |
| Macro F1 | 0.7734 |
| Macro Precision | 0.8115 |
| Macro Recall | 0.7429 |

## Per-Class Metrics
| Class | F1 | Precision | Recall |
|-------|-----|-----------|--------|
| up | 0.6386 | 0.7030 | 0.5850 |
| down | 0.7487 | 0.8185 | 0.6898 |
| not_de | 0.9330 | 0.9130 | 0.9539 |

## LogFC Regression
- MAE: 0.0453
- NMAE: 0.3261

## Confusion Matrix
(Rows=True, Cols=Pred: up, down, not_de)

| | up | down | not_de |
|---|---|---|---|
| up | 6580 | 318 | 4349 |
| down | 437 | 5746 | 2147 |
| not_de | 2343 | 956 | 68204 |
