#!/bin/bash -l
#SBATCH --job-name=pyhmsc-chain
#SBATCH --account=project_462000131
#SBATCH --partition=small-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=output/%x-%A_%a.out
#SBATCH --error=output/%x-%A_%a.err

set -euo pipefail

# Submit from the hmsc-hpc repository root after running
# docs/lumi_python_native_compile_sbatch.sh. Each array task samples one chain
# selected by SLURM_ARRAY_TASK_ID from the shared compiled model.
#
# Optional overrides:
#   RUN_NAME=my_model
#   SAMPLES=1000 TRANSIENT=500 THIN=10 VERBOSE=100
#   SKIP_EXISTING=1

PROJECT_ID="project_462000131"
USER_WORK="/scratch/${PROJECT_ID}/anisrahm"
VENV="${USER_WORK}/venvs/hmsc_tf_env"
PYTHON="${VENV}/bin/python3"
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
RUN_NAME="${RUN_NAME:-array_${SLURM_ARRAY_JOB_ID:-manual}}"
RUN_ROOT="${USER_WORK}/hmsc-hpc-runs/${RUN_NAME}"
CHAIN_ID="${SLURM_ARRAY_TASK_ID}"
COMPILED_INIT="${RUN_ROOT}/compiled/init.json"
CHAIN_DIR="${RUN_ROOT}/chains"
CHAIN_OUTPUT="${CHAIN_DIR}/posterior_chain_${CHAIN_ID}.h5"

mkdir -p output "${CHAIN_DIR}"

module use /appl/local/csc/modulefiles
module load tensorflow/2.16
source "${VENV}/bin/activate"
cd "${REPO_DIR}"

echo "Repository: ${REPO_DIR}"
echo "Run root: ${RUN_ROOT}"
echo "Chain id: ${CHAIN_ID}"
echo "Compiled init: ${COMPILED_INIT}"
echo "Python: ${PYTHON}"
"${PYTHON}" -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"

if [[ ! -f "${COMPILED_INIT}" ]]; then
  cat >&2 <<EOF
Missing compiled model:
  ${COMPILED_INIT}

Run the compile job first:
  RUN_NAME=${RUN_NAME} sbatch docs/lumi_python_native_compile_sbatch.sh
EOF
  exit 2
fi

"${PYTHON}" -m pyhmsc validate-init "${COMPILED_INIT}" --strict

if [[ "${SKIP_EXISTING:-0}" == "1" && -s "${CHAIN_OUTPUT}" ]]; then
  echo "Skipping existing chain output: ${CHAIN_OUTPUT}"
  exit 0
fi

srun "${PYTHON}" -m pyhmsc sample \
  "${COMPILED_INIT}" \
  --output "${CHAIN_OUTPUT}" \
  --chains "${CHAIN_ID}" \
  --samples "${SAMPLES:-1000}" \
  --transient "${TRANSIENT:-500}" \
  --thin "${THIN:-10}" \
  --verbose "${VERBOSE:-100}"

echo "Done. Chain output: ${CHAIN_OUTPUT}"
