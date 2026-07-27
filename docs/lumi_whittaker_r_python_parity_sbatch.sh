#!/bin/bash -l
#SBATCH --job-name=whittaker-rpy-parity
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
RUN_NAME="${RUN_NAME:-whittaker_r_python_parity_${SLURM_JOB_ID:-manual}}"
RUN_ROOT="${RUN_ROOT:-${USER_WORK}/hmsc-hpc-runs/${RUN_NAME}}"
SAMPLES="${SAMPLES:-1000}"
TRANSIENT="${TRANSIENT:-500}"
THIN="${THIN:-10}"
CHAINS="${CHAINS:-2}"
VERBOSE="${VERBOSE:-500}"
RNG_SEED="${RNG_SEED:-20260716}"
FP="${FP:-64}"
RSCRIPT="${RSCRIPT:-Rscript}"
R_MODULE="${R_MODULE:-cray-R/4.4.0}"
R_LIBS_USER="${R_LIBS_USER:-${USER_WORK}/R/library/4.4}"

mkdir -p output "${RUN_ROOT}"

module use /appl/local/csc/modulefiles
module load "${R_MODULE}"
module load tensorflow/2.16
source "${VENV}/bin/activate"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export SINGULARITYENV_PYTHONPATH="${PYTHONPATH}"
export APPTAINERENV_PYTHONPATH="${PYTHONPATH}"
export R_LIBS_USER

echo "Repository: ${REPO_DIR}"
echo "Run root: ${RUN_ROOT}"
echo "Samples/transient/thin/chains: ${SAMPLES}/${TRANSIENT}/${THIN}/${CHAINS}"
"${PYTHON}" -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"
"${PYTHON}" -c "import pyreadr; print('pyreadr: available')"
"${RSCRIPT}" -e 'library(Hmsc); library(jsonify); cat("R/Hmsc/jsonify available\n")'

SECONDS=0
"${PYTHON}" examples/run_whittaker_r_python_parity.py \
  --output "${RUN_ROOT}" \
  --samples "${SAMPLES}" \
  --transient "${TRANSIENT}" \
  --thin "${THIN}" \
  --chains "${CHAINS}" \
  --verbose "${VERBOSE}" \
  --rng-seed "${RNG_SEED}" \
  --fp "${FP}" \
  --prepare-r-init-script

"${RSCRIPT}" "${RUN_ROOT}/r_bridge/make_init.R"

"${PYTHON}" examples/run_whittaker_r_python_parity.py \
  --output "${RUN_ROOT}" \
  --samples "${SAMPLES}" \
  --transient "${TRANSIENT}" \
  --thin "${THIN}" \
  --chains "${CHAINS}" \
  --verbose "${VERBOSE}" \
  --rng-seed "${RNG_SEED}" \
  --fp "${FP}" \
  --rscript "${RSCRIPT}" \
  --python "${PYTHON}" \
  --r-init-file "${RUN_ROOT}/r_bridge/init_file.rds"

printf 'wall_time_seconds=%s\n' "${SECONDS}" | tee "${RUN_ROOT}/wall_time.txt"
echo "Whittaker R/Python parity complete: ${RUN_ROOT}"
