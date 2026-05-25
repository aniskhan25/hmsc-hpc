#!/bin/bash -l
#SBATCH --job-name=pyhmsc-compile
#SBATCH --account=project_462000131
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=output/%x-%j.out
#SBATCH --error=output/%x-%j.err

set -euo pipefail

# Compile a Python-native model once before launching chain array jobs.
#
# Required:
#   RUN_NAME=my_model sbatch docs/lumi_python_native_compile_sbatch.sh
#
# Optional:
#   MODEL_CONFIG=/path/to/model.yaml

PROJECT_ID="project_462000131"
USER_WORK="/scratch/${PROJECT_ID}/anisrahm"
VENV="${USER_WORK}/venvs/hmsc_tf_env"
PYTHON="${VENV}/bin/python3"
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
RUN_NAME="${RUN_NAME:?Set RUN_NAME, for example fixed_poisson_array}"
RUN_ROOT="${USER_WORK}/hmsc-hpc-runs/${RUN_NAME}"
COMPILED_DIR="${RUN_ROOT}/compiled"
MODEL_CONFIG="${MODEL_CONFIG:-${REPO_DIR}/examples/projects/fixed_poisson/model.yaml}"

mkdir -p output "${RUN_ROOT}"

module use /appl/local/csc/modulefiles
module load tensorflow/2.16
source "${VENV}/bin/activate"
cd "${REPO_DIR}"

echo "Repository: ${REPO_DIR}"
echo "Run root: ${RUN_ROOT}"
echo "Model config: ${MODEL_CONFIG}"
echo "Python: ${PYTHON}"

"${PYTHON}" -m pyhmsc compile "${MODEL_CONFIG}" --output "${COMPILED_DIR}"
"${PYTHON}" -m pyhmsc validate-init "${COMPILED_DIR}/init.json" --strict

echo "Compiled model: ${COMPILED_DIR}/init.json"
