#!/bin/bash -l
#SBATCH --job-name=pyhmsc-spat-slope-val
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

# End-to-end full/GPP/NNGP spatial random-slope validation run.
# Submit from the hmsc-hpc repository root:
#
#   RUN_NAME=spatial_random_slope_validation sbatch docs/lumi_spatial_random_slope_validation_sbatch.sh
#
# Optional overrides:
#   SAMPLES=1000 TRANSIENT=500 THIN=10 VERBOSE=100
#   SKIP_EXISTING=1
#   MODELS="spatial_gpp spatial_nngp"

PROJECT_ID="project_462000131"
USER_WORK="/scratch/${PROJECT_ID}/anisrahm"
VENV="${USER_WORK}/venvs/hmsc_tf_env"
PYTHON="${VENV}/bin/python3"
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
RUN_NAME="${RUN_NAME:-spatial_random_slope_validation_${SLURM_JOB_ID:-manual}}"
RUN_ROOT="${USER_WORK}/hmsc-hpc-runs/${RUN_NAME}"
PROJECT_DIR="${REPO_DIR}/examples/projects/simulated_spatial_random_slope_validation"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
MODELS="${MODELS:-spatial_full spatial_gpp spatial_nngp}"

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

  if [[ "${SKIP_EXISTING}" == "1" && -s "${compiled}/init.json" ]]; then
    echo "Compiled init exists; skipping compile: ${compiled}/init.json"
  else
    "${PYTHON}" -m pyhmsc compile "${config}" --output "${compiled}"
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

  "${PYTHON}" -m pyhmsc summarize "${posterior}" --param Beta \
    > "${model_run_root}/beta_summary.txt"
  "${PYTHON}" -m pyhmsc diagnostics "${posterior}" --param Beta \
    --output "${model_run_root}/beta_diagnostics.txt"

  "${PYTHON}" -m pyhmsc summarize "${posterior}" --param Eta \
    > "${model_run_root}/eta_summary.txt"
  "${PYTHON}" -m pyhmsc diagnostics "${posterior}" --param Eta \
    --output "${model_run_root}/eta_diagnostics.txt"
  "${PYTHON}" -m pyhmsc diagnostics "${posterior}" --param Eta --align-factors \
    --output "${model_run_root}/eta_aligned_diagnostics.txt"

  "${PYTHON}" -m pyhmsc summarize "${posterior}" --param Lambda --x-index 0 \
    > "${model_run_root}/lambda_intercept_summary.txt"
  "${PYTHON}" -m pyhmsc summarize "${posterior}" --param Lambda --x-index 1 \
    > "${model_run_root}/lambda_slope_summary.txt"
  "${PYTHON}" -m pyhmsc diagnostics "${posterior}" --param Lambda --x-index 0 \
    --output "${model_run_root}/lambda_intercept_diagnostics.txt"
  "${PYTHON}" -m pyhmsc diagnostics "${posterior}" --param Lambda --x-index 1 \
    --output "${model_run_root}/lambda_slope_diagnostics.txt"
}

for model_name in ${MODELS}; do
  case "${model_name}" in
    spatial_full)
      run_model "spatial_full" "${PROJECT_DIR}/model_spatial_full.yaml"
      ;;
    spatial_gpp)
      run_model "spatial_gpp" "${PROJECT_DIR}/model_spatial_gpp.yaml"
      ;;
    spatial_nngp)
      run_model "spatial_nngp" "${PROJECT_DIR}/model_spatial_nngp.yaml"
      ;;
    *)
      echo "Unknown model in MODELS: ${model_name}" >&2
      exit 2
      ;;
  esac
done

if [[ -s "${RUN_ROOT}/spatial_full/posterior.h5" \
   && -s "${RUN_ROOT}/spatial_gpp/posterior.h5" \
   && -s "${RUN_ROOT}/spatial_nngp/posterior.h5" ]]; then
  "${PYTHON}" examples/analyze_spatial_random_slope_validation.py \
    --project "${PROJECT_DIR}" \
    --spatial-full-posterior "${RUN_ROOT}/spatial_full/posterior.h5" \
    --spatial-gpp-posterior "${RUN_ROOT}/spatial_gpp/posterior.h5" \
    --spatial-nngp-posterior "${RUN_ROOT}/spatial_nngp/posterior.h5" \
    --output "${RUN_ROOT}/spatial_random_slope_validation_report.txt"
else
  echo "Skipping analyzer because not all posterior files are present yet."
fi

echo
echo "Spatial full posterior: ${RUN_ROOT}/spatial_full/posterior.h5"
echo "Spatial GPP posterior: ${RUN_ROOT}/spatial_gpp/posterior.h5"
echo "Spatial NNGP posterior: ${RUN_ROOT}/spatial_nngp/posterior.h5"
echo "Validation report: ${RUN_ROOT}/spatial_random_slope_validation_report.txt"
