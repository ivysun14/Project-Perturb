# Majority Vote Baseline Results: REPLOGLE_K562

**Method**: Per-gene majority label from training

## Dataset Info
- Train samples: 3,274,345
- Test samples: 1,094,781
- Train class distribution: {'not_de': 2984119, 'up': 126550, 'down': 163676}
- Test class distribution: {'not_de': 994013, 'down': 57403, 'up': 43365}

## Aggregate Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 0.9081 |
| Macro F1 | 0.3331 |
| Macro Precision | 0.6260 |
| Macro Recall | 0.3411 |

## Per-Class Metrics
| Class | F1 | Precision | Recall |
|-------|-----|-----------|--------|
| up | 0.0138 | 0.4627 | 0.0070 |
| down | 0.0336 | 0.5063 | 0.0174 |
| not_de | 0.9518 | 0.9091 | 0.9988 |

## LogFC Regression
- MAE: 0.0825
- NMAE: 0.6613

## Confusion Matrix
(Rows=True, Cols=Pred: up, down, not_de)

| | up | down | not_de |
|---|---|---|---|
| up | 304 | 80 | 42981 |
| down | 64 | 998 | 56341 |
| not_de | 289 | 893 | 992831 |
