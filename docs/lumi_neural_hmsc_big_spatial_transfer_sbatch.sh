#!/bin/bash -l
#SBATCH --job-name=neural-bigsp-transfer
#SBATCH --account=project_462000131
#SBATCH --partition=dev-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=output/%x-%j.out
#SBATCH --error=output/%x-%j.err

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-project_462000131}"
USER_WORK="${USER_WORK:-/scratch/${PROJECT_ID}/anisrahm}"
VENV="${VENV:-${USER_WORK}/venvs/hmsc_tf_env}"
PYTHON="${PYTHON:-${VENV}/bin/python3}"
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
RUN_NAME="${RUN_NAME:-neural_big_spatial_transfer_${SLURM_JOB_ID:-manual}}"
RUN_ROOT="${RUN_ROOT:-${USER_WORK}/hmsc-hpc-runs/${RUN_NAME}}"
FROZEN_RUN="${FROZEN_RUN:-${USER_WORK}/hmsc-hpc-runs/neural_whittaker_requalification_20260701}"
NEURAL_CHAINS="${NEURAL_CHAINS:-4}"
NEURAL_DRAWS="${NEURAL_DRAWS:-1000}"
MCMC_CHAINS="${MCMC_CHAINS:-2}"
MCMC_SAMPLES="${MCMC_SAMPLES:-1000}"
MCMC_TRANSIENT="${MCMC_TRANSIENT:-500}"
MCMC_THIN="${MCMC_THIN:-5}"
MCMC_VERBOSE="${MCMC_VERBOSE:-500}"
REFERENCE_PARITY_METRICS="${REFERENCE_PARITY_METRICS:-}"
QUALIFIED_REFERENCE_LABEL="${QUALIFIED_REFERENCE_LABEL:-qualified_python_mcmc_fixed}"
SEED="${SEED:-20260701}"
TARGET_CONTEXT_GATE="${TARGET_CONTEXT_GATE:-none}"
TARGET_CONTEXT_GATE_DATASETS="${TARGET_CONTEXT_GATE_DATASETS:-12}"
TARGET_CONTEXT_GATE_MAX_BRIER_RATIO="${TARGET_CONTEXT_GATE_MAX_BRIER_RATIO:-1.0}"
TARGET_CONTEXT_GATE_MAX_LOG_LOSS_RATIO="${TARGET_CONTEXT_GATE_MAX_LOG_LOSS_RATIO:-1.0}"
TARGET_CONTEXT_GATE_MIN_IMPROVEMENT="${TARGET_CONTEXT_GATE_MIN_IMPROVEMENT:-0.0001}"

mkdir -p output "${RUN_ROOT}"
module use /appl/local/csc/modulefiles
module load tensorflow/2.16
source "${VENV}/bin/activate"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export SINGULARITYENV_PYTHONPATH="${PYTHONPATH}"
export APPTAINERENV_PYTHONPATH="${PYTHONPATH}"

echo "Repository: ${REPO_DIR}"
echo "Run root: ${RUN_ROOT}"
echo "Frozen Whittaker run: ${FROZEN_RUN}"
if [[ -n "${REFERENCE_PARITY_METRICS}" ]]; then
  echo "Reference parity metrics: ${REFERENCE_PARITY_METRICS}"
fi
"${PYTHON}" -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"

SECONDS=0
REFERENCE_ARGS=()
if [[ -n "${REFERENCE_PARITY_METRICS}" ]]; then
  REFERENCE_ARGS=(
    --reference-parity-metrics "${REFERENCE_PARITY_METRICS}"
    --qualified-reference-label "${QUALIFIED_REFERENCE_LABEL}"
  )
fi

"${PYTHON}" examples/run_neural_hmsc_big_spatial_transfer.py \
  --frozen-run "${FROZEN_RUN}" \
  --output "${RUN_ROOT}" \
  --neural-chains "${NEURAL_CHAINS}" \
  --neural-draws "${NEURAL_DRAWS}" \
  --mcmc-chains "${MCMC_CHAINS}" \
  --mcmc-samples "${MCMC_SAMPLES}" \
  --mcmc-transient "${MCMC_TRANSIENT}" \
  --mcmc-thin "${MCMC_THIN}" \
  --mcmc-verbose "${MCMC_VERBOSE}" \
  --target-context-gate "${TARGET_CONTEXT_GATE}" \
  --target-context-gate-datasets "${TARGET_CONTEXT_GATE_DATASETS}" \
  --target-context-gate-max-brier-ratio "${TARGET_CONTEXT_GATE_MAX_BRIER_RATIO}" \
  --target-context-gate-max-log-loss-ratio "${TARGET_CONTEXT_GATE_MAX_LOG_LOSS_RATIO}" \
  --target-context-gate-min-improvement "${TARGET_CONTEXT_GATE_MIN_IMPROVEMENT}" \
  "${REFERENCE_ARGS[@]}" \
  --seed "${SEED}"

printf 'wall_time_seconds=%s\n' "${SECONDS}" | tee "${RUN_ROOT}/wall_time.txt"
echo "Frozen Big Spatial Plant transfer complete: ${RUN_ROOT}"
