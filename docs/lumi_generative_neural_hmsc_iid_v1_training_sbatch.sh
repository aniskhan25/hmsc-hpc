#!/bin/bash -l
#SBATCH --job-name=gen-neural-iid-v1-train
#SBATCH --account=project_462000131
#SBATCH --partition=dev-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --output=output/%x-%j.out
#SBATCH --error=output/%x-%j.err

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-project_462000131}"
USER_WORK="${USER_WORK:-/scratch/${PROJECT_ID}/anisrahm}"
VENV="${VENV:-${USER_WORK}/venvs/hmsc_tf_env}"
PYTHON="${PYTHON:-${VENV}/bin/python3}"
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
RUN_NAME="${RUN_NAME:-generative_neural_hmsc_iid_v1_501m_${SLURM_JOB_ID:-manual}}"
RUN_ROOT="${RUN_ROOT:-${USER_WORK}/hmsc-hpc-runs/${RUN_NAME}}"
EXPECTED_SOURCE_COMMIT="${EXPECTED_SOURCE_COMMIT:-}"
OPEN_GENERATIVE_IID_501M_TRAINING="${OPEN_GENERATIVE_IID_501M_TRAINING:-}"
EXPECTED_CONFIRMATION="GENERATE_501M_CANDIDATE_TRAINING_ONLY"
EXPECTED_PREREGISTRATION_SHA256="09c6a195ca139bdf168816b4f50db321c789bfdd061628e4f99a28cca81cea3f"
EXPECTED_SEED_AUDIT_SHA256="39e8763bf8a4fd525dc624570cd2f2b3392dbd1f62d7fa2e3c326f9340194cd6"
EXPECTED_DESIGN_REVIEW_SHA256="d271caed64dc1346b1f8d9e192534949adedd3122c1e311638e912ca868990cc"

if [[ "${OPEN_GENERATIVE_IID_501M_TRAINING}" != "${EXPECTED_CONFIRMATION}" ]]; then
  echo "Refusing to open 501M candidate training." >&2
  echo "Set OPEN_GENERATIVE_IID_501M_TRAINING=${EXPECTED_CONFIRMATION}." >&2
  exit 2
fi
if [[ -z "${EXPECTED_SOURCE_COMMIT}" ]]; then
  echo "EXPECTED_SOURCE_COMMIT must pin the reviewed clean commit." >&2
  exit 2
fi
if [[ -e "${RUN_ROOT}" ]]; then
  echo "Refusing to reuse production run root: ${RUN_ROOT}" >&2
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

if [[ "$(git rev-parse HEAD)" != "${EXPECTED_SOURCE_COMMIT}" ]]; then
  echo "Repository HEAD differs from EXPECTED_SOURCE_COMMIT." >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Production training requires a clean repository worktree." >&2
  exit 2
fi
if [[ "$(sha256sum docs/generative_neural_hmsc_iid_v1_preregistration_2026-07-27.md | awk '{print $1}')" != "${EXPECTED_PREREGISTRATION_SHA256}" ]]; then
  echo "Generative iid preregistration hash differs." >&2
  exit 2
fi
if [[ "$(sha256sum docs/generative_neural_hmsc_iid_v1_seed_audit_2026-07-27.json.md | awk '{print $1}')" != "${EXPECTED_SEED_AUDIT_SHA256}" ]]; then
  echo "Generative iid seed-audit hash differs." >&2
  exit 2
fi
if [[ "$(sha256sum docs/generative_neural_hmsc_iid_v1_design_review_2026-07-27.md | awk '{print $1}')" != "${EXPECTED_DESIGN_REVIEW_SHA256}" ]]; then
  echo "Generative iid design-review hash differs." >&2
  exit 2
fi

echo "Repository: ${REPO_DIR}"
echo "Pinned source commit: ${EXPECTED_SOURCE_COMMIT}"
echo "Run root: ${RUN_ROOT}"
echo "Authorized block: 501000001-501000324"
echo "Responses per owning context: 2"
echo "Fixed schedule: 200 epochs, batch size 4, model seed 501900001"
echo "Sealed blocks: 502000001-505000324 and 511000001-515000324"
"${PYTHON}" -c \
  "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"

SECONDS=0
"${PYTHON}" examples/run_generative_neural_hmsc_iid_v1_production.py \
  train-candidate \
  --output "${RUN_ROOT}" \
  --expected-source-commit "${EXPECTED_SOURCE_COMMIT}" \
  > "${RUN_ROOT}.stdout.json"

"${PYTHON}" examples/run_generative_neural_hmsc_iid_v1_production.py \
  validate-training \
  --freeze-root "${RUN_ROOT}" \
  --expected-source-commit "${EXPECTED_SOURCE_COMMIT}" \
  > "${RUN_ROOT}/read_only_validation.json"

printf 'wall_time_seconds=%s\n' "${SECONDS}" | tee "${RUN_ROOT}/wall_time.txt"
echo "501M training completed: ${RUN_ROOT}"
echo "502M-505M and 511M-515M remain sealed."
