#!/bin/bash -l
#SBATCH --job-name=gen-iid-v2-repair-preflight
#SBATCH --account=project_462000131
#SBATCH --partition=dev-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=00:15:00

set -euo pipefail

SOURCE_ROOT="${SOURCE_ROOT:?SOURCE_ROOT is required}"
OUTPUT_PATH="${OUTPUT_PATH:?OUTPUT_PATH is required}"
EXPECTED_SOURCE_COMMIT="${EXPECTED_SOURCE_COMMIT:?EXPECTED_SOURCE_COMMIT is required}"
VENV="${VENV:-/scratch/project_462000131/anisrahm/venvs/hmsc_tf_env}"
EXPECTED_MODEL_SHA256="87828857ee1718a8825a1a15e7af99abe49a86ee4d179f6cbce6591162aa71bc"
EXPECTED_REVIEW_SHA256="e3b708a09b0c920676e592759f44f7457cc75decff66138b6b29c509254a6192"

if [[ ! "${EXPECTED_SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "Expected source commit must be a full SHA-1." >&2
  exit 2
fi
if [[ -e "${SOURCE_ROOT}/.git" ]]; then
  echo "Preflight requires an isolated source archive." >&2
  exit 2
fi
if [[ ! -f "${SOURCE_ROOT}/examples/run_generative_neural_hmsc_iid_v2.py" ]]; then
  echo "Preflight source tree is incomplete." >&2
  exit 2
fi
if [[ -e "${OUTPUT_PATH}" ]]; then
  echo "Refusing to overwrite preflight output." >&2
  exit 2
fi
if env | grep -q '^OPEN_GENERATIVE_IID'; then
  echo "Preflight refuses every opening token." >&2
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
export MPLCONFIGDIR="${OUTPUT_PATH}.mpl"
export XDG_CACHE_HOME="${OUTPUT_PATH}.cache"

"${VENV}/bin/python3" -m examples.run_generative_neural_hmsc_iid_v2 \
  --mode preflight \
  --expected-source-commit "${EXPECTED_SOURCE_COMMIT}" \
  > "${OUTPUT_PATH}"

PREFLIGHT_PATH="${OUTPUT_PATH}" \
EXPECTED_MODEL_SHA256="${EXPECTED_MODEL_SHA256}" \
EXPECTED_REVIEW_SHA256="${EXPECTED_REVIEW_SHA256}" \
"${VENV}/bin/python3" -c '
import json, os
from pathlib import Path
payload = json.loads(Path(os.environ["PREFLIGHT_PATH"]).read_text())
inventory = {item["path"]: item["sha256"] for item in payload["source_files"]}
assert payload["status"] == "generative_iid_v2_disposable_preflight_sealed"
assert payload["simulation_generation_called"] is False
assert payload["output_created"] is False
assert payload["disposable_seed_ranges_opened"] is False
assert payload["production_511m_opened"] is False
assert payload["fixed_validation_512m_opened"] is False
assert payload["reserved_513m_515m_opened"] is False
assert inventory["pyhmsc/neural/generative_iid_v2.py"] == os.environ["EXPECTED_MODEL_SHA256"]
assert inventory["docs/generative_neural_hmsc_iid_v2_numerical_review_2026-08-01.md"] == os.environ["EXPECTED_REVIEW_SHA256"]
'

sha256sum "${OUTPUT_PATH}" > "${OUTPUT_PATH}.sha256"
echo "Repaired generative iid v2 token-free preflight passed."
echo "593M-594M and 511M-515M remain sealed."
