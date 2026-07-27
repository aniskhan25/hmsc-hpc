#!/bin/bash -l
#SBATCH --job-name=spatial-boundary
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
RUN_NAME="${RUN_NAME:-spatial_boundary_inspection_${SLURM_JOB_ID:-manual}}"
RUN_ROOT="${RUN_ROOT:-${USER_WORK}/hmsc-hpc-runs/${RUN_NAME}}"
SAMPLES="${SAMPLES:-10}"
TRANSIENT="${TRANSIENT:-5}"
THIN="${THIN:-1}"
CHAINS="${CHAINS:-2}"
VERBOSE="${VERBOSE:-10}"
RSCRIPT="${RSCRIPT:-Rscript}"
R_MODULE="${R_MODULE:-cray-R/4.4.0}"
R_LIBS_USER="${R_LIBS_USER:-${USER_WORK}/R/library/4.4}"
CONFIGS="${CONFIGS:-examples/projects/simulated_spatial_validation/model_spatial_full.yaml examples/projects/simulated_spatial_holdout_validation/model_spatial_gpp.yaml examples/projects/simulated_spatial_holdout_validation/model_spatial_nngp.yaml}"

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
echo "Configs: ${CONFIGS}"
"${PYTHON}" -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"
"${PYTHON}" -c "import pyreadr; print('pyreadr: available')"
"${RSCRIPT}" -e 'library(Hmsc); library(jsonify); cat("R/Hmsc/jsonify available\n")'

SECONDS=0
for CONFIG in ${CONFIGS}; do
  CASE_NAME="$(basename "$(dirname "${CONFIG}")")_$(basename "${CONFIG}" .yaml)"
  CASE_ROOT="${RUN_ROOT}/${CASE_NAME}"
  mkdir -p "${CASE_ROOT}"
  echo "Inspecting R/Hmsc spatial boundary for ${CONFIG} -> ${CASE_ROOT}"

  "${PYTHON}" examples/inspect_r_spatial_boundary.py \
    --config "${CONFIG}" \
    --output "${CASE_ROOT}" \
    --samples "${SAMPLES}" \
    --transient "${TRANSIENT}" \
    --thin "${THIN}" \
    --chains "${CHAINS}" \
    --verbose "${VERBOSE}" \
    --prepare-r-init-script

  "${RSCRIPT}" "${CASE_ROOT}/r_bridge/make_spatial_init.R"

  "${PYTHON}" examples/inspect_r_spatial_boundary.py \
    --config "${CONFIG}" \
    --output "${CASE_ROOT}" \
    --samples "${SAMPLES}" \
    --transient "${TRANSIENT}" \
    --thin "${THIN}" \
    --chains "${CHAINS}" \
    --verbose "${VERBOSE}" \
    --r-init-file "${CASE_ROOT}/r_bridge/init_file.rds"
done

printf 'wall_time_seconds=%s\n' "${SECONDS}" | tee "${RUN_ROOT}/wall_time.txt"
echo "Spatial boundary inspection complete: ${RUN_ROOT}"
