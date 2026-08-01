#!/bin/bash -l
#SBATCH --job-name=gen-iid-v2-disposable-retry
#SBATCH --account=project_462000131
#SBATCH --partition=dev-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=03:00:00

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-project_462000131}"
USER_WORK="${USER_WORK:-/scratch/${PROJECT_ID}/anisrahm}"
VENV="${VENV:-${USER_WORK}/venvs/hmsc_tf_env}"
PYTHON="${PYTHON:-${VENV}/bin/python3}"
SOURCE_ROOT="${SOURCE_ROOT:?SOURCE_ROOT is required}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
EXPECTED_SOURCE_COMMIT="${EXPECTED_SOURCE_COMMIT:?EXPECTED_SOURCE_COMMIT is required}"
CONFIRMATION="${CONFIRMATION:-}"
EXPECTED_COMMIT="940d73d6de6e032797e4d695bd9799a74ef0b943"
EXPECTED_CONFIRMATION="GENERATE_593M_594M_DISPOSABLE_ONLY"
OPENING_ENV="OPEN_GENERATIVE_IID_V2_593M_594M_DISPOSABLE_SMOKE"

if [[ "${EXPECTED_SOURCE_COMMIT}" != "${EXPECTED_COMMIT}" ]]; then
  echo "Refusing a source commit other than ${EXPECTED_COMMIT}." >&2
  exit 2
fi
if [[ "${CONFIRMATION}" != "${EXPECTED_CONFIRMATION}" ]]; then
  echo "Refusing to open 593M-594M without the exact confirmation." >&2
  exit 2
fi
if [[ ! -f "${SOURCE_ROOT}/examples/run_generative_neural_hmsc_iid_v2.py" ]]; then
  echo "Isolated v2 source root is incomplete: ${SOURCE_ROOT}" >&2
  exit 2
fi
if [[ -e "${SOURCE_ROOT}/.git" ]]; then
  echo "Isolated v2 source must use host attestation, not a shared Git tree." >&2
  exit 2
fi
if [[ -e "${RUN_ROOT}" ]]; then
  echo "Refusing to reuse v2 disposable run root: ${RUN_ROOT}" >&2
  exit 2
fi

module use /appl/local/csc/modulefiles
module load tensorflow/2.16
source "${VENV}/bin/activate"
cd "${SOURCE_ROOT}"
# LUMI's TensorFlow Singularity image rewrites /scratch to /pfs/lustrep4/scratch.
# A relative entry survives that namespace translation and resolves this tree.
export PYTHONPATH=".:${PYTHONPATH:-}"
export GENERATIVE_IID_V2_HOST_SOURCE_COMMIT="${EXPECTED_SOURCE_COMMIT}"
export GENERATIVE_IID_V2_HOST_SOURCE_BRANCH="feature/generative-neural-hmsc"
export GENERATIVE_IID_V2_HOST_WORKTREE_CLEAN=1

echo "Source root: ${SOURCE_ROOT}"
echo "Source commit: ${EXPECTED_SOURCE_COMMIT}"
echo "Run root: ${RUN_ROOT}"
echo "Authorized disposable training: 593000001-593000018"
echo "Authorized disposable validation: 594000001-594000018"
echo "Sealed production and evaluation: 511000001-515000324"
"${PYTHON}" -c \
  "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"

env -u "${OPENING_ENV}" "${PYTHON}" \
  -m examples.run_generative_neural_hmsc_iid_v2 \
  --mode preflight \
  --expected-source-commit "${EXPECTED_SOURCE_COMMIT}" \
  > "${RUN_ROOT}.preflight.json"

export "${OPENING_ENV}=${CONFIRMATION}"
SECONDS=0
"${PYTHON}" -m examples.run_generative_neural_hmsc_iid_v2 \
  --mode disposable-smoke \
  --output "${RUN_ROOT}" \
  --expected-source-commit "${EXPECTED_SOURCE_COMMIT}"

"${PYTHON}" -m examples.run_generative_neural_hmsc_iid_v2 \
  --mode validate-disposable \
  --output "${RUN_ROOT}" \
  --expected-source-commit "${EXPECTED_SOURCE_COMMIT}" \
  > "${RUN_ROOT}/validator_stdout.json"

sha256sum "${RUN_ROOT}/freeze.json" > "${RUN_ROOT}/freeze.sha256"
printf 'wall_time_seconds=%s\n' "${SECONDS}" | tee "${RUN_ROOT}/wall_time.txt"
echo "Generative iid v2 disposable smoke and validation completed."
echo "511M-515M remain sealed."
