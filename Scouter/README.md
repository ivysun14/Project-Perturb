We train [Scouter](https://www.nature.com/articles/s43588-025-00912-8) using the official codebase which can be found in this [repo](https://github.com/PancakeZoy/scouter).

## Complete Workflow (K562 example)

All scripts are run from the project root and take a dataset index (0=k562, 1=rpe1, 2=hepg2, 3=jurkat).

```bash
# Step 1: Train Scouter on K562
# Argument is dataset index: 0=k562, 1=rpe1, 2=hepg2, 3=jurkat
# Outputs predicted expression h5ads and model checkpoint to Scouter/prediction/k562/
python -u Scouter/scouter_train_full.py 0

# Step 2: Compute DE labels from predicted expression
# Runs Wilcoxon DE on predicted h5ads; outputs discretized labels to Scouter/de_label/k562/
# Argument is (dataset, split) pair index: 6=(k562,test), 7=(k562,calib)
python -u Scouter/scouter_eval_full.py 6
python -u Scouter/scouter_eval_full.py 7

# Step 3: Compute all metrics
# Combines calib and test splits; outputs classification + continuous metrics to Scouter/metrics/
python -u Scouter/scouter_metric_full.py --task_id 0
```