#!/bin/bash -l
#SBATCH --job-name=pyhmsc-big-nngp
#SBATCH --account=project_462000131
#SBATCH --partition=dev-g
#SBATCH --array=0-1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=output/%x-%A_%a.out
#SBATCH --error=output/%x-%A_%a.err

set -euo pipefail

PROJECT_ID="project_462000131"
USER_WORK="/scratch/${PROJECT_ID}/anisrahm"
VENV="${USER_WORK}/venvs/hmsc_tf_env"
PYTHON="${VENV}/bin/python3"
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
RUN_NAME="${RUN_NAME:-big_spatial_holdout_validation_real}"
RUN_ROOT="${USER_WORK}/hmsc-hpc-runs/${RUN_NAME}"
MODEL_ROOT="${RUN_ROOT}/spatial_nngp"
COMPILED="${MODEL_ROOT}/compiled"
CHAIN_DIR="${MODEL_ROOT}/chains"
CHAIN="${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm array}"
POSTERIOR="${CHAIN_DIR}/posterior_chain_${CHAIN}.h5"
RESOURCE="${CHAIN_DIR}/resource_chain_${CHAIN}.txt"

mkdir -p output "${CHAIN_DIR}"
module use /appl/local/csc/modulefiles
module load tensorflow/2.16
source "${VENV}/bin/activate"
cd "${REPO_DIR}"
export PYTHONPATH="${USER_WORK}/hmsc-hpc${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON}" -m pyhmsc validate-init "${COMPILED}/init.json" --strict
if [[ "${SKIP_EXISTING:-1}" == "1" && -s "${POSTERIOR}" ]]; then
  echo "Chain ${CHAIN} posterior exists; skipping sample."
  exit 0
fi

srun /usr/bin/time \
  -f $'elapsed_seconds=%e\nmax_rss_kb=%M' \
  -o "${RESOURCE}" \
  "${PYTHON}" -m pyhmsc sample \
  "${COMPILED}/init.json" \
  --output "${POSTERIOR}" \
  --samples "${SAMPLES:-1000}" \
  --transient "${TRANSIENT:-500}" \
  --thin "${THIN:-10}" \
  --verbose "${VERBOSE:-100}" \
  --chains "${CHAIN}"

echo "Completed NNGP chain ${CHAIN}: ${POSTERIOR}"
