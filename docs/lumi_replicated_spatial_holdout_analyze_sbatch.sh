#!/bin/bash -l
#SBATCH --job-name=pyhmsc-spat-repl-analysis
#SBATCH --account=project_462000131
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=output/%x-%j.out
#SBATCH --error=output/%x-%j.err

set -euo pipefail

PROJECT_ID="project_462000131"
USER_WORK="/scratch/${PROJECT_ID}/anisrahm"
VENV="${USER_WORK}/venvs/hmsc_tf_env"
PYTHON="${VENV}/bin/python3"
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
RUN_NAME="${RUN_NAME:-replicated_spatial_holdout_validation}"
RUN_ROOT="${USER_WORK}/hmsc-hpc-runs/${RUN_NAME}"
PROJECT_ROOT="${PROJECT_ROOT:-${RUN_ROOT}/projects}"

mkdir -p output "${RUN_ROOT}"
module use /appl/local/csc/modulefiles
module load tensorflow/2.16
source "${VENV}/bin/activate"
cd "${REPO_DIR}"
export PYHMSC_REPO_ROOT="${USER_WORK}/hmsc-hpc"
export PYTHONPATH="${PYHMSC_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON}" examples/analyze_replicated_spatial_holdout_validation.py \
  --manifest "${PROJECT_ROOT}/tasks.csv" \
  --run-root "${RUN_ROOT}" \
  --prediction-seed "${PREDICTION_SEED:-17}" \
  --output "${RUN_ROOT}/replicated_spatial_holdout_report.txt"

echo "Report: ${RUN_ROOT}/replicated_spatial_holdout_report.txt"
