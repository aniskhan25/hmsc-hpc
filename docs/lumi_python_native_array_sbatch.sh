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

# Submit from the hmsc-hpc repository root. Each array task compiles the model
# into its own task directory and samples one chain selected by
# SLURM_ARRAY_TASK_ID.
#
# Optional overrides:
#   MODEL_CONFIG=/path/to/model.yaml
#   RUN_NAME=my_model
#   SAMPLES=1000 TRANSIENT=500 THIN=10 VERBOSE=100

PROJECT_ID="project_462000131"
USER_WORK="/scratch/${PROJECT_ID}/anisrahm"
VENV="${USER_WORK}/venvs/hmsc_tf_env"
PYTHON="${VENV}/bin/python3"
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
RUN_NAME="${RUN_NAME:-array_${SLURM_ARRAY_JOB_ID:-manual}}"
RUN_ROOT="${USER_WORK}/hmsc-hpc-runs/${RUN_NAME}"
CHAIN_ID="${SLURM_ARRAY_TASK_ID}"
TASK_ROOT="${RUN_ROOT}/tasks/${CHAIN_ID}"
CHAIN_DIR="${RUN_ROOT}/chains"
MODEL_CONFIG="${MODEL_CONFIG:-${REPO_DIR}/examples/projects/fixed_poisson/model.yaml}"

mkdir -p output "${TASK_ROOT}" "${CHAIN_DIR}"

module use /appl/local/csc/modulefiles
module load tensorflow/2.16
source "${VENV}/bin/activate"
cd "${REPO_DIR}"

echo "Repository: ${REPO_DIR}"
echo "Run root: ${RUN_ROOT}"
echo "Chain id: ${CHAIN_ID}"
echo "Model config: ${MODEL_CONFIG}"
echo "Python: ${PYTHON}"
"${PYTHON}" -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"

"${PYTHON}" -m pyhmsc compile "${MODEL_CONFIG}" --output "${TASK_ROOT}/compiled"
"${PYTHON}" -m pyhmsc validate-init "${TASK_ROOT}/compiled/init.json" --strict

srun "${PYTHON}" -m pyhmsc sample \
  "${TASK_ROOT}/compiled/init.json" \
  --output "${CHAIN_DIR}/posterior_chain_${CHAIN_ID}.h5" \
  --chains "${CHAIN_ID}" \
  --samples "${SAMPLES:-1000}" \
  --transient "${TRANSIENT:-500}" \
  --thin "${THIN:-10}" \
  --verbose "${VERBOSE:-100}"

echo "Done. Chain output: ${CHAIN_DIR}/posterior_chain_${CHAIN_ID}.h5"
