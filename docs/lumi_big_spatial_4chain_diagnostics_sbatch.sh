#!/bin/bash -l
#SBATCH --job-name=pyhmsc-spatial-4diag
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

# Spatial-only diagnostic run for the compact big_spatial plant project.
# This keeps resource use bounded by running only model_spatial_full.yaml.
#
# Submit from the hmsc-hpc repository root:
#
#   RUN_NAME=big_spatial_4chain_diag sbatch docs/lumi_big_spatial_4chain_diagnostics_sbatch.sh
#
# Optional overrides:
#   CHAINS=4 SAMPLES=2000 TRANSIENT=1000 THIN=10 VERBOSE=200

PROJECT_ID="project_462000131"
USER_WORK="/scratch/${PROJECT_ID}/anisrahm"
VENV="${USER_WORK}/venvs/hmsc_tf_env"
PYTHON="${VENV}/bin/python3"
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
RUN_NAME="${RUN_NAME:-big_spatial_4chain_diag_${SLURM_JOB_ID:-manual}}"
RUN_ROOT="${USER_WORK}/hmsc-hpc-runs/${RUN_NAME}"
PROJECT_DIR="${REPO_DIR}/examples/projects/big_spatial_plants_validation"
MODEL_CONFIG="${MODEL_CONFIG:-${PROJECT_DIR}/model_spatial_full.yaml}"
COMPILED_DIR="${RUN_ROOT}/compiled"
POSTERIOR="${RUN_ROOT}/posterior.h5"
CHAINS="${CHAINS:-4}"

mkdir -p output "${RUN_ROOT}"

module use /appl/local/csc/modulefiles
module load tensorflow/2.16
source "${VENV}/bin/activate"
cd "${REPO_DIR}"

echo "Repository: ${REPO_DIR}"
echo "Run root: ${RUN_ROOT}"
echo "Model config: ${MODEL_CONFIG}"
echo "Chains: ${CHAINS}"
echo "Python: ${PYTHON}"
"${PYTHON}" -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"
"${PYTHON}" -c "import tf_keras; print('tf_keras:', tf_keras.__version__)"
"${PYTHON}" -c "import tensorflow_probability as tfp; print('TFP:', tfp.__version__)"
"${PYTHON}" -c "import h5py; print('h5py:', h5py.__version__)"
"${PYTHON}" -c "import hmsc, pyhmsc; print('hmsc-hpc import: ok')"

"${PYTHON}" -m pyhmsc compile "${MODEL_CONFIG}" \
  --chains "${CHAINS}" \
  --output "${COMPILED_DIR}"
"${PYTHON}" -m pyhmsc validate-init "${COMPILED_DIR}/init.json" --strict

srun "${PYTHON}" -m pyhmsc sample \
  "${COMPILED_DIR}/init.json" \
  --output "${POSTERIOR}" \
  --samples "${SAMPLES:-2000}" \
  --transient "${TRANSIENT:-1000}" \
  --thin "${THIN:-10}" \
  --verbose "${VERBOSE:-200}"

"${PYTHON}" -m pyhmsc summarize "${POSTERIOR}" --param Beta \
  > "${RUN_ROOT}/beta_summary.txt"
"${PYTHON}" -m pyhmsc diagnostics "${POSTERIOR}" --param Beta \
  --output "${RUN_ROOT}/beta_diagnostics.txt"
"${PYTHON}" -m pyhmsc summarize "${POSTERIOR}" --param Eta \
  > "${RUN_ROOT}/eta_summary.txt"
"${PYTHON}" -m pyhmsc diagnostics "${POSTERIOR}" --param Eta \
  --output "${RUN_ROOT}/eta_diagnostics.txt"
"${PYTHON}" -m pyhmsc diagnostics "${POSTERIOR}" --param Eta --align-factors \
  --output "${RUN_ROOT}/eta_aligned_diagnostics.txt"
"${PYTHON}" -m pyhmsc summarize "${POSTERIOR}" --param Lambda \
  > "${RUN_ROOT}/lambda_summary.txt"
"${PYTHON}" -m pyhmsc diagnostics "${POSTERIOR}" --param Lambda \
  --output "${RUN_ROOT}/lambda_diagnostics.txt"
"${PYTHON}" -m pyhmsc diagnostics "${POSTERIOR}" --param Lambda --align-factors \
  --output "${RUN_ROOT}/lambda_aligned_diagnostics.txt"
"${PYTHON}" -m pyhmsc diagnostics "${POSTERIOR}" --param Associations \
  --output "${RUN_ROOT}/association_diagnostics.txt"
"${PYTHON}" -m pyhmsc associations "${POSTERIOR}" \
  > "${RUN_ROOT}/species_associations.txt"

echo
echo "Posterior: ${POSTERIOR}"
echo "Beta diagnostics: ${RUN_ROOT}/beta_diagnostics.txt"
echo "Eta diagnostics: ${RUN_ROOT}/eta_diagnostics.txt"
echo "Eta aligned diagnostics: ${RUN_ROOT}/eta_aligned_diagnostics.txt"
echo "Lambda diagnostics: ${RUN_ROOT}/lambda_diagnostics.txt"
echo "Lambda aligned diagnostics: ${RUN_ROOT}/lambda_aligned_diagnostics.txt"
echo "Association diagnostics: ${RUN_ROOT}/association_diagnostics.txt"
echo "Species associations: ${RUN_ROOT}/species_associations.txt"
