#!/bin/bash -l
#SBATCH --job-name=pyhmsc-spatial-val
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

# End-to-end simulated spatial validation run. Submit from the hmsc-hpc
# repository root:
#
#   RUN_NAME=spatial_validation_test sbatch docs/lumi_spatial_validation_sbatch.sh
#
# Optional overrides:
#   SAMPLES=1000 TRANSIENT=500 THIN=10 VERBOSE=100

PROJECT_ID="project_462000131"
USER_WORK="/scratch/${PROJECT_ID}/anisrahm"
VENV="${USER_WORK}/venvs/hmsc_tf_env"
PYTHON="${VENV}/bin/python3"
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
RUN_NAME="${RUN_NAME:-spatial_validation_${SLURM_JOB_ID:-manual}}"
RUN_ROOT="${USER_WORK}/hmsc-hpc-runs/${RUN_NAME}"
PROJECT_DIR="${REPO_DIR}/examples/projects/simulated_spatial_validation"

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

run_model() {
  local name="$1"
  local config="$2"
  local model_run_root="${RUN_ROOT}/${name}"
  local compiled="${model_run_root}/compiled"
  local posterior="${model_run_root}/posterior.h5"

  mkdir -p "${model_run_root}"

  echo
  echo "== ${name} =="
  echo "Model config: ${config}"

  "${PYTHON}" -m pyhmsc compile "${config}" --output "${compiled}"
  "${PYTHON}" -m pyhmsc validate-init "${compiled}/init.json" --strict

  srun "${PYTHON}" -m pyhmsc sample \
    "${compiled}/init.json" \
    --output "${posterior}" \
    --samples "${SAMPLES:-1000}" \
    --transient "${TRANSIENT:-500}" \
    --thin "${THIN:-10}" \
    --verbose "${VERBOSE:-100}"

  "${PYTHON}" -m pyhmsc summarize "${posterior}" --param Beta \
    > "${model_run_root}/beta_summary.txt"

  if [[ "${name}" != "fixed" ]]; then
    "${PYTHON}" -m pyhmsc summarize "${posterior}" --param Eta \
      > "${model_run_root}/eta_summary.txt"
    "${PYTHON}" -m pyhmsc summarize "${posterior}" --param Lambda \
      > "${model_run_root}/lambda_summary.txt"
    "${PYTHON}" -m pyhmsc diagnostics "${posterior}" --param Eta \
      --output "${model_run_root}/eta_diagnostics.txt"
    "${PYTHON}" -m pyhmsc diagnostics "${posterior}" --param Lambda \
      --output "${model_run_root}/lambda_diagnostics.txt"
    "${PYTHON}" -m pyhmsc diagnostics "${posterior}" --param Eta --align-factors \
      --output "${model_run_root}/eta_aligned_diagnostics.txt"
    "${PYTHON}" -m pyhmsc diagnostics "${posterior}" --param Lambda --align-factors \
      --output "${model_run_root}/lambda_aligned_diagnostics.txt"
    "${PYTHON}" -m pyhmsc diagnostics "${posterior}" --param Associations \
      --output "${model_run_root}/association_diagnostics.txt"
    "${PYTHON}" -m pyhmsc associations "${posterior}" \
      > "${model_run_root}/species_associations.txt"
  fi
}

run_model "fixed" "${PROJECT_DIR}/model_fixed.yaml"
run_model "iid" "${PROJECT_DIR}/model_iid.yaml"
run_model "spatial" "${PROJECT_DIR}/model_spatial_full.yaml"

"${PYTHON}" examples/analyze_spatial_validation.py \
  --project "${PROJECT_DIR}" \
  --fixed-posterior "${RUN_ROOT}/fixed/posterior.h5" \
  --iid-posterior "${RUN_ROOT}/iid/posterior.h5" \
  --spatial-posterior "${RUN_ROOT}/spatial/posterior.h5" \
  --output "${RUN_ROOT}/spatial_validation_report.txt"

echo
echo "Fixed posterior: ${RUN_ROOT}/fixed/posterior.h5"
echo "IID posterior: ${RUN_ROOT}/iid/posterior.h5"
echo "Spatial posterior: ${RUN_ROOT}/spatial/posterior.h5"
echo "Validation report: ${RUN_ROOT}/spatial_validation_report.txt"
