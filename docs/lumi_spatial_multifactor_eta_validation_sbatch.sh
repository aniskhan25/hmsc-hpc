#!/bin/bash -l
#SBATCH --job-name=pyhmsc-multieta-val
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

# End-to-end NNGP multi-factor Eta validation run.
# Submit from the hmsc-hpc repository root:
#
#   RUN_NAME=spatial_multifactor_eta_validation sbatch docs/lumi_spatial_multifactor_eta_validation_sbatch.sh

PROJECT_ID="project_462000131"
USER_WORK="/scratch/${PROJECT_ID}/anisrahm"
VENV="${USER_WORK}/venvs/hmsc_tf_env"
PYTHON="${VENV}/bin/python3"
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
RUN_NAME="${RUN_NAME:-spatial_multifactor_eta_validation_${SLURM_JOB_ID:-manual}}"
RUN_ROOT="${USER_WORK}/hmsc-hpc-runs/${RUN_NAME}"
PROJECT_DIR="${PROJECT_DIR:-${REPO_DIR}/examples/projects/simulated_spatial_multifactor_eta_validation}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

mkdir -p output "${RUN_ROOT}"

module use /appl/local/csc/modulefiles
module load tensorflow/2.16
source "${VENV}/bin/activate"
cd "${REPO_DIR}"

echo "Repository: ${REPO_DIR}"
echo "Run root: ${RUN_ROOT}"
echo "Project: ${PROJECT_DIR}"
echo "Python: ${PYTHON}"
"${PYTHON}" -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"
"${PYTHON}" -c "import tf_keras; print('tf_keras:', tf_keras.__version__)"
"${PYTHON}" -c "import tensorflow_probability as tfp; print('TFP:', tfp.__version__)"
"${PYTHON}" -c "import h5py; print('h5py:', h5py.__version__)"
"${PYTHON}" -c "import hmsc, pyhmsc; print('hmsc-hpc import: ok')"

compiled="${RUN_ROOT}/compiled"
posterior="${RUN_ROOT}/posterior.h5"

if [[ "${SKIP_EXISTING}" == "1" && -s "${compiled}/init.json" ]]; then
  echo "Compiled init exists; skipping compile: ${compiled}/init.json"
else
  "${PYTHON}" -m pyhmsc compile "${PROJECT_DIR}/model_spatial_nngp.yaml" --output "${compiled}"
fi
"${PYTHON}" -m pyhmsc validate-init "${compiled}/init.json" --strict

if [[ "${SKIP_EXISTING}" == "1" && -s "${posterior}" ]]; then
  echo "Posterior exists; skipping sample: ${posterior}"
else
  srun "${PYTHON}" -m pyhmsc sample \
    "${compiled}/init.json" \
    --output "${posterior}" \
    --samples "${SAMPLES:-1000}" \
    --transient "${TRANSIENT:-500}" \
    --thin "${THIN:-10}" \
    --verbose "${VERBOSE:-100}"
fi

"${PYTHON}" -m pyhmsc summarize "${posterior}" --param Beta > "${RUN_ROOT}/beta_summary.txt"
"${PYTHON}" -m pyhmsc diagnostics "${posterior}" --param Beta --output "${RUN_ROOT}/beta_diagnostics.txt"
"${PYTHON}" -m pyhmsc summarize "${posterior}" --param Eta --align-factors > "${RUN_ROOT}/eta_aligned_summary.txt"
"${PYTHON}" -m pyhmsc diagnostics "${posterior}" --param Eta --align-factors --output "${RUN_ROOT}/eta_aligned_diagnostics.txt"
"${PYTHON}" -m pyhmsc summarize "${posterior}" --param Lambda --align-factors > "${RUN_ROOT}/lambda_aligned_summary.txt"
"${PYTHON}" -m pyhmsc diagnostics "${posterior}" --param Lambda --align-factors --output "${RUN_ROOT}/lambda_aligned_diagnostics.txt"
"${PYTHON}" -m pyhmsc diagnostics "${posterior}" --param Associations --output "${RUN_ROOT}/association_diagnostics.txt"

"${PYTHON}" examples/analyze_spatial_multifactor_eta_validation.py \
  --project "${PROJECT_DIR}" \
  --posterior "${posterior}" \
  --output "${RUN_ROOT}/spatial_multifactor_eta_validation_report.txt"

echo
echo "Posterior: ${posterior}"
echo "Validation report: ${RUN_ROOT}/spatial_multifactor_eta_validation_report.txt"
