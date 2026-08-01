#!/bin/bash -l
#SBATCH --job-name=gen-iid-v2-final-disposable
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
EXPECTED_COMMIT="cca9e97518e77c5ca958dfdc3bee753997ed7ac5"
EXPECTED_CONFIRMATION="GENERATE_593M_594M_DISPOSABLE_ONLY"
OPENING_ENV="OPEN_GENERATIVE_IID_V2_593M_594M_DISPOSABLE_SMOKE"
EXPECTED_MODEL_SHA256="87828857ee1718a8825a1a15e7af99abe49a86ee4d179f6cbce6591162aa71bc"
EXPECTED_HARNESS_SHA256="b09e3e1eb743fe62a509876a284323e5bb151ce043cac7b60dba8a9a35f9300e"
EXPECTED_REVIEW_SHA256="e3b708a09b0c920676e592759f44f7457cc75decff66138b6b29c509254a6192"

if [[ "${EXPECTED_SOURCE_COMMIT}" != "${EXPECTED_COMMIT}" ]]; then
  echo "Refusing a source commit other than ${EXPECTED_COMMIT}." >&2
  exit 2
fi
if [[ "${CONFIRMATION}" != "${EXPECTED_CONFIRMATION}" ]]; then
  echo "Refusing to open 593M-594M without the exact confirmation." >&2
  exit 2
fi
if env | grep -q '^OPEN_GENERATIVE_IID'; then
  echo "Final disposable scheduler refuses inherited opening tokens." >&2
  exit 2
fi
if [[ ! -f "${SOURCE_ROOT}/examples/run_generative_neural_hmsc_iid_v2.py" ]]; then
  echo "Isolated repaired source root is incomplete: ${SOURCE_ROOT}" >&2
  exit 2
fi
if [[ -e "${SOURCE_ROOT}/.git" ]]; then
  echo "Repaired source must use host attestation, not a shared Git tree." >&2
  exit 2
fi
if [[ "$(sha256sum "${SOURCE_ROOT}/pyhmsc/neural/generative_iid_v2.py" | cut -d' ' -f1)" != "${EXPECTED_MODEL_SHA256}" ]]; then
  echo "Repaired model source hash differs." >&2
  exit 2
fi
if [[ "$(sha256sum "${SOURCE_ROOT}/examples/run_generative_neural_hmsc_iid_v2.py" | cut -d' ' -f1)" != "${EXPECTED_HARNESS_SHA256}" ]]; then
  echo "Repaired harness source hash differs." >&2
  exit 2
fi
if [[ "$(sha256sum "${SOURCE_ROOT}/docs/generative_neural_hmsc_iid_v2_numerical_review_2026-08-01.md" | cut -d' ' -f1)" != "${EXPECTED_REVIEW_SHA256}" ]]; then
  echo "Numerical-review evidence hash differs." >&2
  exit 2
fi
if [[ -e "${RUN_ROOT}" || -e "${RUN_ROOT}.preflight.json" ]]; then
  echo "Refusing to reuse final disposable output paths: ${RUN_ROOT}" >&2
  exit 2
fi

module use /appl/local/csc/modulefiles
module load tensorflow/2.16
source "${VENV}/bin/activate"
cd "${SOURCE_ROOT}"
export PYTHONPATH=".:${PYTHONPATH:-}"
export GENERATIVE_IID_V2_HOST_SOURCE_COMMIT="${EXPECTED_SOURCE_COMMIT}"
export GENERATIVE_IID_V2_HOST_SOURCE_BRANCH="feature/generative-neural-hmsc"
export GENERATIVE_IID_V2_HOST_WORKTREE_CLEAN=1
export MPLCONFIGDIR="${RUN_ROOT}.mpl"
export XDG_CACHE_HOME="${RUN_ROOT}.cache"

echo "Source root: ${SOURCE_ROOT}"
echo "Source commit: ${EXPECTED_SOURCE_COMMIT}"
echo "Run root: ${RUN_ROOT}"
echo "Authorized final disposable training: 593000001-593000018"
echo "Authorized final disposable validation: 594000001-594000018"
echo "Sealed production and evaluation: 511000001-515000324"
"${PYTHON}" -c \
  "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"

env -u "${OPENING_ENV}" "${PYTHON}" \
  -m examples.run_generative_neural_hmsc_iid_v2 \
  --mode preflight \
  --expected-source-commit "${EXPECTED_SOURCE_COMMIT}" \
  > "${RUN_ROOT}.preflight.json"

PREFLIGHT_PATH="${RUN_ROOT}.preflight.json" \
EXPECTED_SOURCE_COMMIT="${EXPECTED_SOURCE_COMMIT}" \
EXPECTED_MODEL_SHA256="${EXPECTED_MODEL_SHA256}" \
EXPECTED_REVIEW_SHA256="${EXPECTED_REVIEW_SHA256}" \
"${PYTHON}" -c '
import json, os
from pathlib import Path
payload = json.loads(Path(os.environ["PREFLIGHT_PATH"]).read_text())
inventory = {item["path"]: item["sha256"] for item in payload["source_files"]}
assert payload["source_commit"] == os.environ["EXPECTED_SOURCE_COMMIT"]
assert payload["simulation_generation_called"] is False
assert payload["output_created"] is False
assert payload["disposable_seed_ranges_opened"] is False
assert payload["production_511m_opened"] is False
assert payload["fixed_validation_512m_opened"] is False
assert payload["reserved_513m_515m_opened"] is False
assert inventory["pyhmsc/neural/generative_iid_v2.py"] == os.environ["EXPECTED_MODEL_SHA256"]
assert inventory["docs/generative_neural_hmsc_iid_v2_numerical_review_2026-08-01.md"] == os.environ["EXPECTED_REVIEW_SHA256"]
'

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
unset "${OPENING_ENV}"

sha256sum "${RUN_ROOT}/freeze.json" > "${RUN_ROOT}/freeze.sha256"
printf 'wall_time_seconds=%s\n' "${SECONDS}" | tee "${RUN_ROOT}/wall_time.txt"
echo "Final repaired generative iid v2 disposable verification completed."
echo "511M-515M remain sealed."
