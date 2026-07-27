#!/bin/bash -l
#SBATCH --job-name=neural-baseline-freeze
#SBATCH --account=project_462000131
#SBATCH --partition=dev-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=40G
#SBATCH --time=00:10:00
#SBATCH --output=output/%x-%j.out
#SBATCH --error=output/%x-%j.err

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-project_462000131}"
USER_WORK="${USER_WORK:-/scratch/${PROJECT_ID}/anisrahm}"
VENV="${VENV:-${USER_WORK}/venvs/hmsc_tf_env}"
PYTHON="${PYTHON:-${VENV}/bin/python3}"
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
RUN_NAME="${RUN_NAME:-neural_hmsc_predictive_baseline_freeze_${SLURM_JOB_ID:-manual}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${USER_WORK}/hmsc-hpc-runs/${RUN_NAME}}"
REGISTRY_ROOT="${REGISTRY_ROOT:-${USER_WORK}/hmsc-hpc-deployments}"
BASELINE_ID="${BASELINE_ID:-neural_predictive_affine_v1}"
REQUALIFICATION_ROOT="${REQUALIFICATION_ROOT:-${USER_WORK}/hmsc-hpc-runs/neural_hmsc_probability_ensemble_api_requalification_20260720}"
SMOKE_ROOT="${SMOKE_ROOT:-${USER_WORK}/hmsc-hpc-runs/neural_hmsc_predictive_deployment_smoke_20260720}"
FROZEN_RUN_ROOT="${FROZEN_RUN_ROOT:-${USER_WORK}/hmsc-hpc-runs/neural_hmsc_source_transfer_realdata_sensitivity_20260720}"
FROZEN_SEED="${FROZEN_SEED:-20260721}"

mkdir -p output "${OUTPUT_ROOT}"
module use /appl/local/csc/modulefiles
module load tensorflow/2.16
source "${VENV}/bin/activate"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export SINGULARITYENV_PYTHONPATH="${PYTHONPATH}"
export APPTAINERENV_PYTHONPATH="${PYTHONPATH}"

echo "Registry root: ${REGISTRY_ROOT}"
echo "Stable baseline identifier: ${BASELINE_ID}"
echo "Requalification evidence: ${REQUALIFICATION_ROOT}"
echo "Scheduler-smoke evidence: ${SMOKE_ROOT}"

SECONDS=0
"${PYTHON}" examples/freeze_neural_hmsc_predictive_baseline.py \
  --registry-root "${REGISTRY_ROOT}" \
  --requalification-root "${REQUALIFICATION_ROOT}" \
  --smoke-root "${SMOKE_ROOT}" \
  --baseline-id "${BASELINE_ID}" \
  > "${OUTPUT_ROOT}/baseline_freeze.json"

"${PYTHON}" examples/smoke_neural_hmsc_predictive_deployment.py \
  --baseline-root "${REGISTRY_ROOT}" \
  --baseline-id "${BASELINE_ID}" \
  --frozen-run-root "${FROZEN_RUN_ROOT}" \
  --seed "${FROZEN_SEED}" \
  --output "${OUTPUT_ROOT}"

printf 'wall_time_seconds=%s\n' "${SECONDS}" | tee "${OUTPUT_ROOT}/wall_time.txt"
echo "Frozen predictive baseline validated: ${REGISTRY_ROOT}/${BASELINE_ID}"
