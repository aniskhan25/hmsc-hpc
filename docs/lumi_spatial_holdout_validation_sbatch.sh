#!/bin/bash -l
#SBATCH --job-name=pyhmsc-spat-holdout
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

# Training-only fit plus held-out CLI prediction validation.
# Submit from the hmsc-hpc repository root:
#
#   RUN_NAME=spatial_holdout_validation \
#     sbatch docs/lumi_spatial_holdout_validation_sbatch.sh
#
# Resume selected models with the same RUN_NAME:
#
#   RUN_NAME=spatial_holdout_validation MODELS="spatial_gpp spatial_nngp" \
#     sbatch docs/lumi_spatial_holdout_validation_sbatch.sh

PROJECT_ID="project_462000131"
USER_WORK="/scratch/${PROJECT_ID}/anisrahm"
VENV="${USER_WORK}/venvs/hmsc_tf_env"
PYTHON="${VENV}/bin/python3"
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
REPO_IMPORT_DIR="${PYHMSC_REPO_ROOT:-${USER_WORK}/hmsc-hpc}"
RUN_NAME="${RUN_NAME:-spatial_holdout_validation_${SLURM_JOB_ID:-manual}}"
RUN_ROOT="${USER_WORK}/hmsc-hpc-runs/${RUN_NAME}"
PROJECT_DIR="${PROJECT_DIR:-${REPO_DIR}/examples/projects/simulated_spatial_holdout_validation}"
MODELS="${MODELS:-fixed spatial_full spatial_gpp spatial_nngp}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
PREDICTION_SEED="${PREDICTION_SEED:-17}"

mkdir -p output "${RUN_ROOT}"

module use /appl/local/csc/modulefiles
module load tensorflow/2.16
source "${VENV}/bin/activate"
cd "${REPO_DIR}"
export PYHMSC_REPO_ROOT="${REPO_IMPORT_DIR}"
export PYTHONPATH="${REPO_IMPORT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "Repository: ${REPO_DIR}"
echo "Run root: ${RUN_ROOT}"
echo "Project: ${PROJECT_DIR}"
echo "Python: ${PYTHON}"
"${PYTHON}" -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"
"${PYTHON}" -c "import h5py, pyhmsc; print('h5py:', h5py.__version__); print('pyhmsc import: ok')"
"${PYTHON}" -c "import inspect, pyhmsc.posterior as p; print('pyhmsc posterior:', p.__file__); print('predict_ci:', inspect.signature(p.HmscFit.predict_ci))"

run_model() {
  local name="$1"
  local config="${PROJECT_DIR}/model_${name}.yaml"
  local model_root="${RUN_ROOT}/${name}"
  local compiled="${model_root}/compiled"
  local posterior="${model_root}/posterior.h5"
  local prediction="${model_root}/heldout_predictions.csv"
  local conditional_prediction="${model_root}/heldout_predictions_conditional.csv"

  mkdir -p "${model_root}"
  echo
  echo "== ${name} =="

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

  if [[ "${name}" == "fixed" ]]; then
    "${PYTHON}" -m pyhmsc predict "${posterior}" \
      --X "${PROJECT_DIR}/data/test/X.csv" \
      --model-config "${config}" \
      --output "${prediction}"
  else
    "${PYTHON}" -m pyhmsc predict "${posterior}" \
      --X "${PROJECT_DIR}/data/test/X.csv" \
      --model-config "${config}" \
      --study-design "${PROJECT_DIR}/data/test/study_design.csv" \
      --coords "${PROJECT_DIR}/data/test/coords.csv" \
      --random-effects known \
      --unseen-groups nearest \
      --output "${prediction}"
    if [[ "${name}" == "spatial_full" ]]; then
      "${PYTHON}" -m pyhmsc predict "${posterior}" \
        --X "${PROJECT_DIR}/data/test/X.csv" \
        --model-config "${config}" \
        --study-design "${PROJECT_DIR}/data/test/study_design.csv" \
        --coords "${PROJECT_DIR}/data/test/coords.csv" \
        --random-effects known \
        --spatial-prediction conditional \
        --seed "${PREDICTION_SEED}" \
        --output "${conditional_prediction}"
    fi
  fi
}

for model_name in ${MODELS}; do
  case "${model_name}" in
    fixed|spatial_full|spatial_gpp|spatial_nngp)
      run_model "${model_name}"
      ;;
    *)
      echo "Unknown model in MODELS: ${model_name}" >&2
      exit 2
      ;;
  esac
done

analyzer_args=()
for model_name in fixed spatial_full spatial_gpp spatial_nngp; do
  prediction="${RUN_ROOT}/${model_name}/heldout_predictions.csv"
  posterior="${RUN_ROOT}/${model_name}/posterior.h5"
  if [[ -s "${prediction}" ]]; then
    analyzer_args+=(--prediction "${model_name}=${prediction}")
  fi
  if [[ -s "${posterior}" ]]; then
    analyzer_args+=(--posterior "${model_name}=${posterior}")
  fi
done

conditional_prediction="${RUN_ROOT}/spatial_full/heldout_predictions_conditional.csv"
conditional_posterior="${RUN_ROOT}/spatial_full/posterior.h5"
if [[ -s "${conditional_prediction}" ]]; then
  analyzer_args+=(--prediction "spatial_full_conditional=${conditional_prediction}")
fi
if [[ -s "${conditional_posterior}" ]]; then
  analyzer_args+=(--posterior "spatial_full_conditional=${conditional_posterior}")
fi

if [[ "${#analyzer_args[@]}" -gt 0 ]]; then
  "${PYTHON}" examples/analyze_spatial_holdout_validation.py \
    --project "${PROJECT_DIR}" \
    --seed "${PREDICTION_SEED}" \
    "${analyzer_args[@]}" \
    --output "${RUN_ROOT}/spatial_holdout_validation_report.txt"
else
  echo "Skipping analyzer because no prediction files are present."
fi

echo
echo "Validation report: ${RUN_ROOT}/spatial_holdout_validation_report.txt"
