#!/bin/bash -l
#SBATCH --job-name=neural-whittaker
#SBATCH --account=project_462000131
#SBATCH --partition=dev-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=output/%x-%j.out
#SBATCH --error=output/%x-%j.err

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-project_462000131}"
USER_WORK="${USER_WORK:-/scratch/${PROJECT_ID}/anisrahm}"
VENV="${VENV:-${USER_WORK}/venvs/hmsc_tf_env}"
PYTHON="${PYTHON:-${VENV}/bin/python3}"
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
RUN_NAME="${RUN_NAME:-neural_whittaker_${SLURM_JOB_ID:-manual}}"
RUN_ROOT="${RUN_ROOT:-${USER_WORK}/hmsc-hpc-runs/${RUN_NAME}}"
TRAIN_DATASETS="${TRAIN_DATASETS:-512}"
CALIBRATION_DATASETS="${CALIBRATION_DATASETS:-128}"
SBC_DATASETS="${SBC_DATASETS:-128}"
SBC_DRAWS="${SBC_DRAWS:-512}"
EPOCHS="${EPOCHS:-120}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NEURAL_CHAINS="${NEURAL_CHAINS:-4}"
NEURAL_DRAWS="${NEURAL_DRAWS:-1000}"
MCMC_CHAINS="${MCMC_CHAINS:-2}"
MCMC_SAMPLES="${MCMC_SAMPLES:-1000}"
MCMC_TRANSIENT="${MCMC_TRANSIENT:-500}"
MCMC_THIN="${MCMC_THIN:-5}"
MCMC_VERBOSE="${MCMC_VERBOSE:-500}"
SEED="${SEED:-20260629}"

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
echo "Training/calibration/SBC datasets: ${TRAIN_DATASETS}/${CALIBRATION_DATASETS}/${SBC_DATASETS}"
echo "Epochs/batch size: ${EPOCHS}/${BATCH_SIZE}"
"${PYTHON}" -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"

SECONDS=0
"${PYTHON}" examples/run_neural_hmsc_whittaker.py \
  --output "${RUN_ROOT}" \
  --train-datasets "${TRAIN_DATASETS}" \
  --calibration-datasets "${CALIBRATION_DATASETS}" \
  --sbc-datasets "${SBC_DATASETS}" \
  --sbc-draws "${SBC_DRAWS}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --neural-chains "${NEURAL_CHAINS}" \
  --neural-draws "${NEURAL_DRAWS}" \
  --mcmc-chains "${MCMC_CHAINS}" \
  --mcmc-samples "${MCMC_SAMPLES}" \
  --mcmc-transient "${MCMC_TRANSIENT}" \
  --mcmc-thin "${MCMC_THIN}" \
  --mcmc-verbose "${MCMC_VERBOSE}" \
  --seed "${SEED}"

printf 'wall_time_seconds=%s\n' "${SECONDS}" | tee "${RUN_ROOT}/wall_time.txt"
echo "Whittaker Neural-HMSC workflow complete: ${RUN_ROOT}"
