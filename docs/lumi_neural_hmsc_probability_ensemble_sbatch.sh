#!/bin/bash -l
#SBATCH --job-name=neural-prob-ensemble
#SBATCH --account=project_462000131
#SBATCH --partition=dev-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=00:20:00
#SBATCH --output=output/%x-%j.out
#SBATCH --error=output/%x-%j.err

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-project_462000131}"
USER_WORK="${USER_WORK:-/scratch/${PROJECT_ID}/anisrahm}"
VENV="${VENV:-${USER_WORK}/venvs/hmsc_tf_env}"
PYTHON="${PYTHON:-${VENV}/bin/python3}"
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
RUN_NAME="${RUN_NAME:-neural_hmsc_probability_ensemble_${SLURM_JOB_ID:-manual}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${USER_WORK}/hmsc-hpc-runs/${RUN_NAME}}"
FROZEN_RUN_ROOT="${FROZEN_RUN_ROOT:-${USER_WORK}/hmsc-hpc-runs/neural_hmsc_source_transfer_realdata_sensitivity_20260720}"
SEEDS="${SEEDS:-20260721 20260722 20260723}"

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

SECONDS=0
"${PYTHON}" examples/evaluate_neural_hmsc_probability_ensemble.py \
  --run-root "${FROZEN_RUN_ROOT}" \
  --seeds ${SEEDS} \
  --output "${OUTPUT_ROOT}"

printf 'wall_time_seconds=%s\n' "${SECONDS}" | tee "${OUTPUT_ROOT}/wall_time.txt"
echo "Frozen probability ensemble evaluation complete: ${OUTPUT_ROOT}"
