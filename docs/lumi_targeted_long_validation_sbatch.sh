#!/bin/bash -l
#SBATCH --job-name=pyhmsc-long-val
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

# Targeted longer validation for cases where diagnostics justify extra work.
# Default profile targets residual species associations for the compact
# big-spatial full-spatial model. Submit from the repository root:
#
#   RUN_NAME=big_spatial_assoc_long \
#     sbatch docs/lumi_targeted_long_validation_sbatch.sh
#
# Common overrides:
#   MODEL_CONFIG=examples/projects/big_spatial_plants_validation/model_spatial_full.yaml
#   DIAGNOSTIC_PROFILE=associations   # associations | beta | latent | all
#   CHAINS=4 SAMPLES=2500 TRANSIENT=1000 THIN=10
#   SKIP_SAMPLE=1                     # reuse existing ${RUN_ROOT}/posterior.h5

PROJECT_ID="project_462000131"
USER_WORK="/scratch/${PROJECT_ID}/anisrahm"
VENV="${USER_WORK}/venvs/hmsc_tf_env"
PYTHON="${VENV}/bin/python3"
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
RUN_NAME="${RUN_NAME:-targeted_long_validation_${SLURM_JOB_ID:-manual}}"
RUN_ROOT="${USER_WORK}/hmsc-hpc-runs/${RUN_NAME}"
MODEL_CONFIG="${MODEL_CONFIG:-${REPO_DIR}/examples/projects/big_spatial_plants_validation/model_spatial_full.yaml}"
COMPILED_DIR="${RUN_ROOT}/compiled"
POSTERIOR="${POSTERIOR:-${RUN_ROOT}/posterior.h5}"
DIAGNOSTIC_PROFILE="${DIAGNOSTIC_PROFILE:-associations}"
CHAINS="${CHAINS:-4}"
SAMPLES="${SAMPLES:-2500}"
TRANSIENT="${TRANSIENT:-1000}"
THIN="${THIN:-10}"
VERBOSE="${VERBOSE:-200}"

mkdir -p output "${RUN_ROOT}"

module use /appl/local/csc/modulefiles
module load tensorflow/2.16
source "${VENV}/bin/activate"
cd "${REPO_DIR}"

echo "Repository: ${REPO_DIR}"
echo "Run root: ${RUN_ROOT}"
echo "Model config: ${MODEL_CONFIG}"
echo "Posterior: ${POSTERIOR}"
echo "Profile: ${DIAGNOSTIC_PROFILE}"
echo "Chains/samples/transient/thin: ${CHAINS}/${SAMPLES}/${TRANSIENT}/${THIN}"
echo "Python: ${PYTHON}"
"${PYTHON}" -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"
"${PYTHON}" -c "import h5py; print('h5py:', h5py.__version__)"
"${PYTHON}" -c "import hmsc, pyhmsc; print('hmsc-hpc import: ok')"

if [[ "${SKIP_SAMPLE:-0}" != "1" ]]; then
  "${PYTHON}" -m pyhmsc compile "${MODEL_CONFIG}" \
    --chains "${CHAINS}" \
    --output "${COMPILED_DIR}"
  "${PYTHON}" -m pyhmsc validate-init "${COMPILED_DIR}/init.json" --strict

  srun "${PYTHON}" -m pyhmsc sample \
    "${COMPILED_DIR}/init.json" \
    --output "${POSTERIOR}" \
    --samples "${SAMPLES}" \
    --transient "${TRANSIENT}" \
    --thin "${THIN}" \
    --verbose "${VERBOSE}"
fi

run_beta() {
  "${PYTHON}" -m pyhmsc summarize "${POSTERIOR}" --param Beta \
    > "${RUN_ROOT}/beta_summary.txt"
  "${PYTHON}" -m pyhmsc diagnostics "${POSTERIOR}" --param Beta \
    --output "${RUN_ROOT}/beta_diagnostics.txt"
}

run_associations() {
  "${PYTHON}" -m pyhmsc diagnostics "${POSTERIOR}" --param Associations \
    --output "${RUN_ROOT}/association_diagnostics.txt"
  "${PYTHON}" -m pyhmsc associations "${POSTERIOR}" \
    > "${RUN_ROOT}/species_associations.txt"
}

run_latent() {
  "${PYTHON}" -m pyhmsc summarize "${POSTERIOR}" --param Eta \
    > "${RUN_ROOT}/eta_summary.txt"
  "${PYTHON}" -m pyhmsc summarize "${POSTERIOR}" --param Lambda \
    > "${RUN_ROOT}/lambda_summary.txt"
  "${PYTHON}" -m pyhmsc diagnostics "${POSTERIOR}" --param Eta \
    --output "${RUN_ROOT}/eta_diagnostics.txt"
  "${PYTHON}" -m pyhmsc diagnostics "${POSTERIOR}" --param Lambda \
    --output "${RUN_ROOT}/lambda_diagnostics.txt"
  "${PYTHON}" -m pyhmsc diagnostics "${POSTERIOR}" --param Eta --align-factors \
    --output "${RUN_ROOT}/eta_aligned_diagnostics.txt"
  "${PYTHON}" -m pyhmsc diagnostics "${POSTERIOR}" --param Lambda --align-factors \
    --output "${RUN_ROOT}/lambda_aligned_diagnostics.txt"
}

case "${DIAGNOSTIC_PROFILE}" in
  beta)
    run_beta
    ;;
  associations)
    run_associations
    ;;
  latent)
    run_latent
    ;;
  all)
    run_beta
    run_associations
    run_latent
    ;;
  *)
    echo "Unsupported DIAGNOSTIC_PROFILE=${DIAGNOSTIC_PROFILE}" >&2
    exit 2
    ;;
esac

"${PYTHON}" examples/plan_long_validation.py "${POSTERIOR}" \
  --include-latent \
  --output "${RUN_ROOT}/targeted_long_validation_plan.txt" \
  --csv-output "${RUN_ROOT}/targeted_long_validation_plan.csv"

echo
echo "Posterior: ${POSTERIOR}"
echo "Plan: ${RUN_ROOT}/targeted_long_validation_plan.txt"
echo "Profile outputs are in ${RUN_ROOT}"
