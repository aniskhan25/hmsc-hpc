#!/bin/bash -l
#SBATCH --job-name=gen-iid-v2-num-review
#SBATCH --account=project_462000131
#SBATCH --partition=dev-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=01:00:00

set -euo pipefail

SOURCE_ROOT="${SOURCE_ROOT:?SOURCE_ROOT is required}"
DIAGNOSTIC_SCRIPT="${DIAGNOSTIC_SCRIPT:?DIAGNOSTIC_SCRIPT is required}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
VENV="${VENV:-/scratch/project_462000131/anisrahm/venvs/hmsc_tf_env}"
EXPECTED_DIAGNOSTIC_SHA256="e3ca2876bdad56ce6c42d45720f88435742567e8977c292c530316cdd32e973b"

if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "Refusing to reuse numerical-review output: ${OUTPUT_ROOT}" >&2
  exit 2
fi
if [[ -e "${SOURCE_ROOT}/.git" ]]; then
  echo "Numerical review requires the isolated source tree." >&2
  exit 2
fi
if [[ "$(sha256sum "${DIAGNOSTIC_SCRIPT}" | cut -d' ' -f1)" != "${EXPECTED_DIAGNOSTIC_SHA256}" ]]; then
  echo "Numerical diagnostic hash differs." >&2
  exit 2
fi
if env | grep -q '^OPEN_GENERATIVE_IID'; then
  echo "Numerical review refuses every opening token." >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}"
module use /appl/local/csc/modulefiles
module load tensorflow/2.16
source "${VENV}/bin/activate"
cd "${SOURCE_ROOT}"
export PYTHONPATH=".:${PYTHONPATH:-}"
export MPLCONFIGDIR="${OUTPUT_ROOT}/mpl"
export XDG_CACHE_HOME="${OUTPUT_ROOT}/cache"

for mode in frozen symmetric_float64 symmetric_float64_cpu; do
  set +e
  DIAGNOSTIC_MODE="${mode}" "${VENV}/bin/python3" -c \
    'import os, runpy, sys; path = os.environ["DIAGNOSTIC_SCRIPT"]; sys.argv = [path, "--mode", os.environ["DIAGNOSTIC_MODE"]]; runpy.run_path(path, run_name="__main__")' \
    > "${OUTPUT_ROOT}/${mode}.json" \
    2> "${OUTPUT_ROOT}/${mode}.err"
  status=$?
  set -e
  printf '%s=%s\n' "${mode}" "${status}" | tee -a "${OUTPUT_ROOT}/exit_codes.txt"
done

sha256sum "${DIAGNOSTIC_SCRIPT}" > "${OUTPUT_ROOT}/diagnostic.sha256"
echo "No-ledger generative iid v2 numerical review completed."
echo "511M-515M remain sealed."
