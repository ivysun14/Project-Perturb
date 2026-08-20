We train [TranScouter](https://www.biorxiv.org/content/10.64898/2026.07.30.741892v1) using the official codebase which can be found in this [repo](https://github.com/PancakeZoy/TranScouter).

## Complete Workflow (K562 example)

All scripts are run from the project root. `--pred_ds k562` means K562 is the held-out target cell line.

```bash
# Step 1: Curate training data
# Merges source cell lines and prepares training h5ads; outputs to TranScouter/training_data/
python -u TranScouter/step1_curation_full.py --pred_ds k562

# Step 2: Train TranScouter (GPU required, ~4h)
# Trains on all cell lines except K562; outputs predicted expression h5ads and checkpoint to TranScouter/prediction/k562/
python -u TranScouter/step2_train_full.py --pred_ds k562

# Step 2.5: Predict on calibration set and merge with test predictions
# Uses saved checkpoint to predict calib perts; merges calib+test h5ads and DE CSVs into testcalib files
python -u TranScouter/step2.5_predict_calib.py --pred_ds k562

# Step 3: Compute DE labels from predicted expression
# Runs Wilcoxon DE on predicted h5ads; outputs discretized labels to TranScouter/de_label/k562/
# Parameterized by (pred_ds, split) pairs: 0=(k562,test), 1=(k562,rpe1_calib), 2=(k562,hepg2_calib), 3=(k562,jurkat_calib). The later three are generated for additional calibration method testing, and we did not use them eventually in this benchmark.
python -u TranScouter/step3_eval_full.py 0
python -u TranScouter/step3_eval_full.py 1
python -u TranScouter/step3_eval_full.py 2
python -u TranScouter/step3_eval_full.py 3

# Step 4: Compute all metrics
# Reads merged testcalib files; outputs classification + continuous metrics to TranScouter/
python -u TranScouter/step4_metric_full.py --task_id 0

# Step 5: Compute metrics after restricting test set to a common set of perturbations
# Re-computes metrics after filtering out perturbations in dropped_test_perts.json; outputs to TranScouter/
python -u TranScouter/step5_metric_drop_perts.py --task_id 0
```