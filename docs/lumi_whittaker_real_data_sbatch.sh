#!/bin/bash -l
#SBATCH --job-name=pyhmsc-whittaker
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

# End-to-end real-data validation run using the Whittaker plant dataset from
# the HMSC book example. Submit from the hmsc-hpc repository root:
#
#   RUN_NAME=whittaker_real_test sbatch docs/lumi_whittaker_real_data_sbatch.sh
#
# Optional overrides:
#   SAMPLES=1000 TRANSIENT=500 THIN=10 VERBOSE=100
#   MODEL_CONFIG=examples/projects/whittaker_plants_hmsc_book/model_iid_site.yaml

PROJECT_ID="project_462000131"
USER_WORK="/scratch/${PROJECT_ID}/anisrahm"
VENV="${USER_WORK}/venvs/hmsc_tf_env"
PYTHON="${VENV}/bin/python3"
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
RUN_NAME="${RUN_NAME:-whittaker_real_${SLURM_JOB_ID:-manual}}"
RUN_ROOT="${USER_WORK}/hmsc-hpc-runs/${RUN_NAME}"
PROJECT_DIR="${REPO_DIR}/examples/projects/whittaker_plants_hmsc_book"
MODEL_CONFIG="${MODEL_CONFIG:-${PROJECT_DIR}/model.yaml}"
if [[ "${MODEL_CONFIG}" != /* ]]; then
  MODEL_CONFIG="${REPO_DIR}/${MODEL_CONFIG}"
fi
COMPILED_DIR="${RUN_ROOT}/compiled"
POSTERIOR="${RUN_ROOT}/posterior.h5"

mkdir -p output "${RUN_ROOT}"

module use /appl/local/csc/modulefiles
module load tensorflow/2.16
source "${VENV}/bin/activate"
cd "${REPO_DIR}"

echo "Repository: ${REPO_DIR}"
echo "Run root: ${RUN_ROOT}"
echo "Model config: ${MODEL_CONFIG}"
echo "Python: ${PYTHON}"
"${PYTHON}" -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"
"${PYTHON}" -c "import tf_keras; print('tf_keras:', tf_keras.__version__)"
"${PYTHON}" -c "import tensorflow_probability as tfp; print('TFP:', tfp.__version__)"
"${PYTHON}" -c "import h5py; print('h5py:', h5py.__version__)"
"${PYTHON}" -c "import hmsc, pyhmsc; print('hmsc-hpc import: ok')"

if grep -q '^study_design:' "${MODEL_CONFIG}"; then
  PPC_RANDOM_EFFECTS="${PPC_RANDOM_EFFECTS:-known}"
else
  PPC_RANDOM_EFFECTS="${PPC_RANDOM_EFFECTS:-none}"
fi

"${PYTHON}" -m pyhmsc compile "${MODEL_CONFIG}" --output "${COMPILED_DIR}"
"${PYTHON}" -m pyhmsc validate-init "${COMPILED_DIR}/init.json" --strict

srun "${PYTHON}" -m pyhmsc sample \
  "${COMPILED_DIR}/init.json" \
  --output "${POSTERIOR}" \
  --samples "${SAMPLES:-1000}" \
  --transient "${TRANSIENT:-500}" \
  --thin "${THIN:-10}" \
  --verbose "${VERBOSE:-100}"

"${PYTHON}" -m pyhmsc summarize "${POSTERIOR}" --param Beta \
  > "${RUN_ROOT}/beta_summary.txt"
"${PYTHON}" -m pyhmsc summarize "${POSTERIOR}" --param Gamma \
  > "${RUN_ROOT}/gamma_summary.txt"
"${PYTHON}" -m pyhmsc diagnostics "${POSTERIOR}" --param Beta \
  --output "${RUN_ROOT}/beta_diagnostics.txt"
"${PYTHON}" -m pyhmsc diagnostics "${POSTERIOR}" --param Gamma \
  --output "${RUN_ROOT}/gamma_diagnostics.txt"
if [[ "${PPC_RANDOM_EFFECTS}" != "none" ]]; then
  "${PYTHON}" -m pyhmsc diagnostics "${POSTERIOR}" --param Eta \
    --output "${RUN_ROOT}/eta_diagnostics.txt"
  "${PYTHON}" -m pyhmsc diagnostics "${POSTERIOR}" --param Lambda \
    --output "${RUN_ROOT}/lambda_diagnostics.txt"
fi
"${PYTHON}" examples/analyze_whittaker_plants.py \
  --posterior "${POSTERIOR}" \
  --project "${PROJECT_DIR}" \
  --model-config "${MODEL_CONFIG}" \
  --random-effects "${PPC_RANDOM_EFFECTS}" \
  --ppc-output "${RUN_ROOT}/posterior_predictive_check.txt" \
  --output "${RUN_ROOT}/whittaker_report.txt"

echo "Posterior: ${POSTERIOR}"
echo "Beta summary: ${RUN_ROOT}/beta_summary.txt"
echo "Gamma summary: ${RUN_ROOT}/gamma_summary.txt"
echo "Beta diagnostics: ${RUN_ROOT}/beta_diagnostics.txt"
echo "Gamma diagnostics: ${RUN_ROOT}/gamma_diagnostics.txt"
if [[ "${PPC_RANDOM_EFFECTS}" != "none" ]]; then
  echo "Eta diagnostics: ${RUN_ROOT}/eta_diagnostics.txt"
  echo "Lambda diagnostics: ${RUN_ROOT}/lambda_diagnostics.txt"
fi
echo "Posterior predictive check: ${RUN_ROOT}/posterior_predictive_check.txt"
echo "Validation report: ${RUN_ROOT}/whittaker_report.txt"
