#!/bin/bash -l
#SBATCH --job-name=neural-m54-candidate-eval
#SBATCH --account=project_462000131
#SBATCH --partition=dev-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=output/%x-%j.out
#SBATCH --error=output/%x-%j.err

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-project_462000131}"
USER_WORK="${USER_WORK:-/scratch/${PROJECT_ID}/anisrahm}"
VENV="${VENV:-${USER_WORK}/venvs/hmsc_tf_env}"
PYTHON="${PYTHON:-${VENV}/bin/python3}"
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
FREEZE_ROOT="${FREEZE_ROOT:-${USER_WORK}/hmsc-hpc-runs/neural_hmsc_m54_candidate_train_calibration_20129822}"
RUN_NAME="${RUN_NAME:-neural_hmsc_m54_candidate_evaluation_${SLURM_JOB_ID:-manual}}"
RUN_ROOT="${RUN_ROOT:-${USER_WORK}/hmsc-hpc-runs/${RUN_NAME}}"
REGISTRY_ROOT="${REGISTRY_ROOT:-${USER_WORK}/hmsc-hpc-deployments}"
VARIABLE_BASELINE="${VARIABLE_BASELINE:-${REGISTRY_ROOT}/neural_hmsc_variable_probit_v1}"
CONFIRMATION="${CONFIRMATION:-}"
EXPECTED_CONFIRMATION="OPEN_M54_CANDIDATE_EVALUATION"
EXPECTED_FREEZE_SHA256="021488d1868b773232112bfa9199aad74602e26ef119bcd7a7f38bb2ea90728e"

if [[ "${CONFIRMATION}" != "${EXPECTED_CONFIRMATION}" ]]; then
  echo "Refusing to open the Milestone 54 candidate evaluation block." >&2
  echo "Set CONFIRMATION=${EXPECTED_CONFIRMATION}." >&2
  exit 2
fi

if [[ ! -f "${FREEZE_ROOT}/m54_train_calibration_freeze.json" ]]; then
  echo "Candidate freeze is missing: ${FREEZE_ROOT}" >&2
  exit 2
fi
OBSERVED_FREEZE_SHA256="$(sha256sum "${FREEZE_ROOT}/m54_train_calibration_freeze.json" | awk '{print $1}')"
if [[ "${OBSERVED_FREEZE_SHA256}" != "${EXPECTED_FREEZE_SHA256}" ]]; then
  echo "Candidate freeze hash differs: ${OBSERVED_FREEZE_SHA256}" >&2
  exit 2
fi
if [[ -e "${RUN_ROOT}" ]]; then
  echo "Refusing to reuse an existing Milestone 54 evaluation root: ${RUN_ROOT}" >&2
  exit 2
fi

mkdir -p output
module use /appl/local/csc/modulefiles
module load tensorflow/2.16
source "${VENV}/bin/activate"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export SINGULARITYENV_PYTHONPATH="${PYTHONPATH}"
export APPTAINERENV_PYTHONPATH="${PYTHONPATH}"

echo "Repository: ${REPO_DIR}"
echo "Frozen candidate: ${FREEZE_ROOT}"
echo "Frozen candidate SHA-256: ${OBSERVED_FREEZE_SHA256}"
echo "Evaluation root: ${RUN_ROOT}"
echo "Protocol action: candidate reserved evaluation only"
echo "Reserved candidate evaluation block: 103000001-103000243 (authorized)"
"${PYTHON}" -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"

SECONDS=0
"${PYTHON}" examples/qualify_neural_hmsc_variable_design.py evaluate \
  --role candidate \
  --freeze-root "${FREEZE_ROOT}" \
  --output "${RUN_ROOT}" \
  --confirmation "${CONFIRMATION}" \
  --fixed-registry "${REGISTRY_ROOT}" \
  --variable-baseline "${VARIABLE_BASELINE}" \
  > "${RUN_ROOT}.stdout.json"

sha256sum "${RUN_ROOT}/m54_role_evaluation.json" \
  > "${RUN_ROOT}/m54_role_evaluation.sha256"
printf 'wall_time_seconds=%s\n' "${SECONDS}" | tee "${RUN_ROOT}/wall_time.txt"
echo "Milestone 54 candidate reserved evaluation completed: ${RUN_ROOT}"
