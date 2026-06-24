#!/bin/bash -l
#SBATCH --job-name=pyhmsc-whit-holdout
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

PROJECT_ID="project_462000131"
USER_WORK="/scratch/${PROJECT_ID}/anisrahm"
VENV="${USER_WORK}/venvs/hmsc_tf_env"
PYTHON="${VENV}/bin/python3"
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
RUN_NAME="${RUN_NAME:-whittaker_holdout_validation_${SLURM_JOB_ID:-manual}}"
RUN_ROOT="${USER_WORK}/hmsc-hpc-runs/${RUN_NAME}"
PROJECT_DIR="${PROJECT_DIR:-${RUN_ROOT}/project}"
SAMPLES="${SAMPLES:-1000}"
TRANSIENT="${TRANSIENT:-500}"
THIN="${THIN:-10}"

mkdir -p output "${RUN_ROOT}"
module use /appl/local/csc/modulefiles
module load tensorflow/2.16
source "${VENV}/bin/activate"
cd "${REPO_DIR}"
export PYTHONPATH="${USER_WORK}/hmsc-hpc${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON}" examples/generate_whittaker_holdout_validation.py \
  --source examples/projects/whittaker_plants_hmsc_book \
  --output "${PROJECT_DIR}"

run_model() {
  local name="$1"
  local model_root="${RUN_ROOT}/${name}"
  local compiled="${model_root}/compiled"
  local posterior="${model_root}/posterior.h5"
  local resource="${model_root}/resource_metrics.txt"
  mkdir -p "${model_root}"

  "${PYTHON}" -m pyhmsc compile "${PROJECT_DIR}/model_${name}.yaml" --output "${compiled}"
  "${PYTHON}" -m pyhmsc validate-init "${compiled}/init.json" --strict
  srun /usr/bin/time \
    -f $'elapsed_seconds=%e\nmax_rss_kb=%M' \
    -o "${resource}" \
    "${PYTHON}" -m pyhmsc sample "${compiled}/init.json" \
    --output "${posterior}" \
    --samples "${SAMPLES}" \
    --transient "${TRANSIENT}" \
    --thin "${THIN}" \
    --verbose "${VERBOSE:-100}"
  {
    echo "samples=${SAMPLES}"
    echo "transient=${TRANSIENT}"
    echo "thin=${THIN}"
  } >> "${resource}"
}

run_model fixed
run_model iid

"${PYTHON}" examples/analyze_whittaker_holdout_validation.py \
  --project "${PROJECT_DIR}" \
  --run-root "${RUN_ROOT}" \
  --output "${RUN_ROOT}/whittaker_holdout_report.txt"

echo "Report: ${RUN_ROOT}/whittaker_holdout_report.txt"
