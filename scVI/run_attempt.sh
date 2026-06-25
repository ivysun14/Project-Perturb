#!/bin/bash
#SBATCH --job-name=scvi_push_attempt
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=8:00:00

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_DIR="${SCRIPT_DIR}"
if [[ -n "${REPO_DIR:-}" ]]; then
  :
elif [[ -f "${SLURM_SUBMIT_DIR:-}/scripts/train.py" ]]; then
  REPO_DIR="${SLURM_SUBMIT_DIR}"
else
  REPO_DIR="${DEFAULT_REPO_DIR}"
fi

if [[ ! -f "${REPO_DIR}/scripts/train.py" ]]; then
  echo "ERROR: Could not find scripts/train.py under REPO_DIR=${REPO_DIR}" >&2
  exit 1
fi

cd "${REPO_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CELL_LINE="${CELL_LINE:-HepG2}"
DATA_DIR="${DATA_DIR:-${REPO_DIR}/data/h5ad_objects}"
SPLIT_CSV="${SPLIT_CSV:-${REPO_DIR}/data/csv/DE_DOC_${CELL_LINE}_splitted.csv}"
OUT_DIR="${OUT_DIR:-${REPO_DIR}/outputs/smoke_test_push}"
CHECKPOINT="${CHECKPOINT:-final.ckpt}"

# Smoke-test settings. Override MAX_STEPS for a longer attempt.
MAX_STEPS="${MAX_STEPS:-2}"
VAL_FREQ="${VAL_FREQ:-1}"
CKPT_EVERY_N_STEPS="${CKPT_EVERY_N_STEPS:-1}"
PREDICTION_MODES="${PREDICTION_MODES:-both}"

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "Repo: ${REPO_DIR}"
echo "Python: ${PYTHON_BIN}"
echo "Cell line: ${CELL_LINE}"
echo "Dataset dir: ${DATA_DIR}"
echo "Split CSV: ${SPLIT_CSV}"
echo "Output dir: ${OUT_DIR}"
echo "Prediction modes: ${PREDICTION_MODES}"

echo "Checking GPU availability..."
"${PYTHON_BIN}" - <<'PY'
import torch
print("torch.cuda.is_available() =", torch.cuda.is_available())
print("torch.cuda.device_count() =", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda device 0 =", torch.cuda.get_device_name(0))
PY

echo "Dry run: train cross"
"${PYTHON_BIN}" scripts/train.py \
  --cell-line "${CELL_LINE}" \
  --mode cross \
  --dataset-dir "${DATA_DIR}" \
  --split-csv "${SPLIT_CSV}" \
  --output-dir "${OUT_DIR}" \
  --python-bin "${PYTHON_BIN}" \
  --extra-override training.max_steps="${MAX_STEPS}" \
  --extra-override training.val_freq="${VAL_FREQ}" \
  --extra-override +training.ckpt_every_n_steps="${CKPT_EVERY_N_STEPS}" \
  --dry-run

echo "Dry run: train within"
"${PYTHON_BIN}" scripts/train.py \
  --cell-line "${CELL_LINE}" \
  --mode within \
  --dataset-dir "${DATA_DIR}" \
  --split-csv "${SPLIT_CSV}" \
  --output-dir "${OUT_DIR}" \
  --python-bin "${PYTHON_BIN}" \
  --train-split-groups train \
  --test-split-groups test \
  --extra-override training.max_steps="${MAX_STEPS}" \
  --extra-override training.val_freq="${VAL_FREQ}" \
  --extra-override +training.ckpt_every_n_steps="${CKPT_EVERY_N_STEPS}" \
  --dry-run

echo "Train cross smoke test"
"${PYTHON_BIN}" scripts/train.py \
  --cell-line "${CELL_LINE}" \
  --mode cross \
  --dataset-dir "${DATA_DIR}" \
  --split-csv "${SPLIT_CSV}" \
  --output-dir "${OUT_DIR}" \
  --python-bin "${PYTHON_BIN}" \
  --extra-override training.max_steps="${MAX_STEPS}" \
  --extra-override training.val_freq="${VAL_FREQ}" \
  --extra-override +training.ckpt_every_n_steps="${CKPT_EVERY_N_STEPS}"

echo "Train within smoke test"
"${PYTHON_BIN}" scripts/train.py \
  --cell-line "${CELL_LINE}" \
  --mode within \
  --dataset-dir "${DATA_DIR}" \
  --split-csv "${SPLIT_CSV}" \
  --output-dir "${OUT_DIR}" \
  --python-bin "${PYTHON_BIN}" \
  --train-split-groups train \
  --test-split-groups test \
  --extra-override training.max_steps="${MAX_STEPS}" \
  --extra-override training.val_freq="${VAL_FREQ}" \
  --extra-override +training.ckpt_every_n_steps="${CKPT_EVERY_N_STEPS}"

echo "Predict cross test set"
"${PYTHON_BIN}" scripts/predict.py \
  --cell-line "${CELL_LINE}" \
  --mode cross \
  --run-dir "${OUT_DIR}/${CELL_LINE}_cross" \
  --split-csv "${SPLIT_CSV}" \
  --checkpoint "${CHECKPOINT}" \
  --prediction-modes "${PREDICTION_MODES}" \
  --python-bin "${PYTHON_BIN}"

echo "Predict within test set"
"${PYTHON_BIN}" scripts/predict.py \
  --cell-line "${CELL_LINE}" \
  --mode within \
  --run-dir "${OUT_DIR}/${CELL_LINE}_within" \
  --split-csv "${SPLIT_CSV}" \
  --checkpoint "${CHECKPOINT}" \
  --prediction-modes "${PREDICTION_MODES}" \
  --python-bin "${PYTHON_BIN}"

echo "Predict cross calibration set"
"${PYTHON_BIN}" scripts/predict_calib.py \
  --cell-line "${CELL_LINE}" \
  --run-dir "${OUT_DIR}/${CELL_LINE}_cross" \
  --split-csv "${SPLIT_CSV}" \
  --checkpoint "${CHECKPOINT}" \
  --prediction-modes "${PREDICTION_MODES}" \
  --python-bin "${PYTHON_BIN}"

echo "Attempt finished successfully."
