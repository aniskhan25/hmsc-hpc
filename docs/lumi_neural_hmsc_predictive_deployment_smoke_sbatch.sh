#!/bin/bash -l
#SBATCH --job-name=neural-deploy-smoke
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
RUN_NAME="${RUN_NAME:-neural_hmsc_predictive_deployment_smoke_${SLURM_JOB_ID:-manual}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${USER_WORK}/hmsc-hpc-runs/${RUN_NAME}}"
REQUALIFICATION_ROOT="${REQUALIFICATION_ROOT:-${USER_WORK}/hmsc-hpc-runs/neural_hmsc_probability_ensemble_api_requalification_20260720}"
MANIFEST_DIR="${MANIFEST_DIR:-${REQUALIFICATION_ROOT}/manifests}"
FROZEN_RUN_ROOT="${FROZEN_RUN_ROOT:-${USER_WORK}/hmsc-hpc-runs/neural_hmsc_source_transfer_realdata_sensitivity_20260720}"
FROZEN_SEED="${FROZEN_SEED:-20260721}"
PREDICTIVE_MEAN_POLICY="${PREDICTIVE_MEAN_POLICY:-affine_branch}"
PREDICTIVE_MEAN_FALLBACK_POLICY="${PREDICTIVE_MEAN_FALLBACK_POLICY:-scale_only}"

mkdir -p output "${OUTPUT_ROOT}"
module use /appl/local/csc/modulefiles
module load tensorflow/2.16
source "${VENV}/bin/activate"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export SINGULARITYENV_PYTHONPATH="${PYTHONPATH}"
export APPTAINERENV_PYTHONPATH="${PYTHONPATH}"

echo "Manifest directory: ${MANIFEST_DIR}"
echo "Default predictive-mean policy: ${PREDICTIVE_MEAN_POLICY}"
echo "Fallback predictive-mean policy: ${PREDICTIVE_MEAN_FALLBACK_POLICY}"
echo "Qualified Python MCMC: statistical reference only"

SECONDS=0
"${PYTHON}" examples/smoke_neural_hmsc_predictive_deployment.py \
  --manifest-dir "${MANIFEST_DIR}" \
  --frozen-run-root "${FROZEN_RUN_ROOT}" \
  --seed "${FROZEN_SEED}" \
  --policy "${PREDICTIVE_MEAN_POLICY}" \
  --fallback-policy "${PREDICTIVE_MEAN_FALLBACK_POLICY}" \
  --output "${OUTPUT_ROOT}"

printf 'wall_time_seconds=%s\n' "${SECONDS}" | tee "${OUTPUT_ROOT}/wall_time.txt"
echo "Neural predictive deployment smoke complete: ${OUTPUT_ROOT}"
