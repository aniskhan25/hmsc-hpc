#!/bin/bash -l
#SBATCH --job-name=pyhmsc-merge
#SBATCH --account=project_462000131
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=output/%x-%j.out
#SBATCH --error=output/%x-%j.err

set -euo pipefail

# Submit after docs/lumi_python_native_array_sbatch.sh finishes.
#
# Required when the array job used a custom name:
#   RUN_NAME=my_model sbatch docs/lumi_python_native_merge_sbatch.sh

PROJECT_ID="project_462000131"
USER_WORK="/scratch/${PROJECT_ID}/anisrahm"
VENV="${USER_WORK}/venvs/hmsc_tf_env"
PYTHON="${VENV}/bin/python3"
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
RUN_NAME="${RUN_NAME:?Set RUN_NAME to the array run name, for example array_<jobid>}"
RUN_ROOT="${USER_WORK}/hmsc-hpc-runs/${RUN_NAME}"
CHAIN_DIR="${RUN_ROOT}/chains"
OUTPUT="${RUN_ROOT}/posterior.h5"
EXPECTED_CHAINS="${EXPECTED_CHAINS:-0 1}"

mkdir -p output

module use /appl/local/csc/modulefiles
module load tensorflow/2.16
source "${VENV}/bin/activate"
cd "${REPO_DIR}"
"${PYTHON}" -c "import h5py; print('h5py:', h5py.__version__)"

inputs=("${CHAIN_DIR}"/posterior_chain_*.h5)
if [[ ! -e "${inputs[0]}" ]]; then
  echo "No chain posterior files found in ${CHAIN_DIR}" >&2
  exit 2
fi

"${PYTHON}" -m pyhmsc chain-status "${CHAIN_DIR}" \
  --expected-chains ${EXPECTED_CHAINS} \
  --run-name "${RUN_NAME}" \
  --strict

"${PYTHON}" -m pyhmsc merge "${inputs[@]}" \
  --expected-chains ${EXPECTED_CHAINS} \
  --output "${OUTPUT}"
"${PYTHON}" -m pyhmsc summarize "${OUTPUT}" --param Beta \
  > "${RUN_ROOT}/beta_summary.txt"

echo "Merged posterior: ${OUTPUT}"
echo "Summary: ${RUN_ROOT}/beta_summary.txt"
