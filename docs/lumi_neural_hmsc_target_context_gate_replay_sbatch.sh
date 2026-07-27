#!/bin/bash -l
#SBATCH --job-name=neural-target-gate
#SBATCH --account=project_462000131
#SBATCH --partition=dev-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=output/%x-%j.out
#SBATCH --error=output/%x-%j.err

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-project_462000131}"
USER_WORK="${USER_WORK:-/scratch/${PROJECT_ID}/anisrahm}"
VENV="${VENV:-${USER_WORK}/venvs/hmsc_tf_env}"
PYTHON="${PYTHON:-${VENV}/bin/python3}"
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
RUN_NAME="${RUN_NAME:-neural_hmsc_target_context_gate_replay_${SLURM_JOB_ID:-manual}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${USER_WORK}/hmsc-hpc-runs/${RUN_NAME}}"
FROZEN_RUN_ROOT="${FROZEN_RUN_ROOT:-${USER_WORK}/hmsc-hpc-runs/neural_hmsc_source_transfer_realdata_sensitivity_20260720}"
SEEDS="${SEEDS:-20260721 20260722 20260723}"
TARGET_CONTEXT_GATE_DATASETS="${TARGET_CONTEXT_GATE_DATASETS:-32}"
TARGET_CONTEXT_GATE_MAX_BRIER_RATIO="${TARGET_CONTEXT_GATE_MAX_BRIER_RATIO:-1.0}"
TARGET_CONTEXT_GATE_MAX_LOG_LOSS_RATIO="${TARGET_CONTEXT_GATE_MAX_LOG_LOSS_RATIO:-1.0}"
TARGET_CONTEXT_GATE_MIN_IMPROVEMENT="${TARGET_CONTEXT_GATE_MIN_IMPROVEMENT:-0.0001}"

mkdir -p output "${OUTPUT_ROOT}"
module use /appl/local/csc/modulefiles
module load tensorflow/2.16
source "${VENV}/bin/activate"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export SINGULARITYENV_PYTHONPATH="${PYTHONPATH}"
export APPTAINERENV_PYTHONPATH="${PYTHONPATH}"

echo "Repository: ${REPO_DIR}"
echo "Frozen run root: ${FROZEN_RUN_ROOT}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Seeds: ${SEEDS}"
echo "Target-context datasets per pool: ${TARGET_CONTEXT_GATE_DATASETS}"
"${PYTHON}" -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"

SECONDS=0
"${PYTHON}" examples/replay_neural_hmsc_target_context_gate.py \
  --run-root "${FROZEN_RUN_ROOT}" \
  --seeds ${SEEDS} \
  --output "${OUTPUT_ROOT}" \
  --datasets "${TARGET_CONTEXT_GATE_DATASETS}" \
  --max-brier-ratio "${TARGET_CONTEXT_GATE_MAX_BRIER_RATIO}" \
  --max-log-loss-ratio "${TARGET_CONTEXT_GATE_MAX_LOG_LOSS_RATIO}" \
  --min-score-improvement "${TARGET_CONTEXT_GATE_MIN_IMPROVEMENT}"

printf 'wall_time_seconds=%s\n' "${SECONDS}" | tee "${OUTPUT_ROOT}/wall_time.txt"
echo "Target-context frozen replay complete: ${OUTPUT_ROOT}"
