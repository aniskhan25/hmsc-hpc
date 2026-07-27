#!/bin/bash -l
#SBATCH --job-name=gen-neural-iid-v1-val
#SBATCH --account=project_462000131
#SBATCH --partition=standard-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=output/%x-%j.out
#SBATCH --error=output/%x-%j.err

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-project_462000131}"
USER_WORK="${USER_WORK:-/scratch/${PROJECT_ID}/anisrahm}"
VENV="${VENV:-${USER_WORK}/venvs/hmsc_tf_env}"
PYTHON="${PYTHON:-${VENV}/bin/python3}"
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
TRAINING_ROOT="${TRAINING_ROOT:?Set TRAINING_ROOT to the frozen 501M run}"
RUN_NAME="${RUN_NAME:-generative_neural_hmsc_iid_v1_502m_${SLURM_JOB_ID:-manual}}"
RUN_ROOT="${RUN_ROOT:-${USER_WORK}/hmsc-hpc-runs/${RUN_NAME}}"
RELEASE_REGISTRY="${RELEASE_REGISTRY:-${USER_WORK}/hmsc-hpc-deployments}"
EXPECTED_SOURCE_COMMIT="${EXPECTED_SOURCE_COMMIT:?Pin the reviewed evaluator commit}"
EXPECTED_TRAINING_FREEZE_SHA256="${EXPECTED_TRAINING_FREEZE_SHA256:?Pin freeze.json}"
EXPECTED_CHECKPOINT_CONTENT_SHA256="${EXPECTED_CHECKPOINT_CONTENT_SHA256:?Pin candidate content}"
EXPECTED_ABLATION_CONTENT_SHA256="${EXPECTED_ABLATION_CONTENT_SHA256:?Pin ablation content}"
OPEN_GENERATIVE_IID_502M_FIXED_VALIDATION="${OPEN_GENERATIVE_IID_502M_FIXED_VALIDATION:-}"
EXPECTED_CONFIRMATION="EVALUATE_502M_FIXED_VALIDATION_ONCE"
EXPECTED_PREREGISTRATION_SHA256="09c6a195ca139bdf168816b4f50db321c789bfdd061628e4f99a28cca81cea3f"
EXPECTED_SEED_AUDIT_SHA256="39e8763bf8a4fd525dc624570cd2f2b3392dbd1f62d7fa2e3c326f9340194cd6"
EXPECTED_DESIGN_REVIEW_SHA256="d271caed64dc1346b1f8d9e192534949adedd3122c1e311638e912ca868990cc"

if [[ "${OPEN_GENERATIVE_IID_502M_FIXED_VALIDATION}" != "${EXPECTED_CONFIRMATION}" ]]; then
  echo "Refusing to open the one-shot 502M fixed-validation block." >&2
  exit 2
fi
if [[ -e "${RUN_ROOT}" ]]; then
  echo "Refusing to reuse fixed-validation run root: ${RUN_ROOT}" >&2
  exit 2
fi
if [[ ! -f "${TRAINING_ROOT}/freeze.json" ]]; then
  echo "Frozen 501M training root is missing." >&2
  exit 2
fi
if [[ ! -f "${RELEASE_REGISTRY}/neural_hmsc_v0_1/release.json" ]]; then
  echo "Immutable neural_hmsc_v0_1 release is missing." >&2
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
  echo "Fixed validation requires a clean repository worktree." >&2
  exit 2
fi
if [[ "$(sha256sum "${TRAINING_ROOT}/freeze.json" | awk '{print $1}')" != "${EXPECTED_TRAINING_FREEZE_SHA256}" ]]; then
  echo "501M training freeze hash differs." >&2
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
echo "Training root: ${TRAINING_ROOT}"
echo "Run root: ${RUN_ROOT}"
echo "Authorized block: 502000001-502000324"
echo "Exact/Python comparator subset: 36 contexts"
echo "Reserved blocks remain sealed: 503000001-505000324"
echo "Redesign blocks remain sealed: 511000001-515000324"

SAVED_CONFIRMATION="${OPEN_GENERATIVE_IID_502M_FIXED_VALIDATION}"
unset OPEN_GENERATIVE_IID_502M_FIXED_VALIDATION
"${PYTHON}" examples/run_generative_neural_hmsc_iid_v1_production.py \
  preflight-fixed-validation \
  --freeze-root "${TRAINING_ROOT}" \
  --expected-source-commit "${EXPECTED_SOURCE_COMMIT}" \
  --expected-checkpoint-content-sha256 "${EXPECTED_CHECKPOINT_CONTENT_SHA256}" \
  --expected-ablation-content-sha256 "${EXPECTED_ABLATION_CONTENT_SHA256}" \
  > "${RUN_ROOT}.preflight.json"
export OPEN_GENERATIVE_IID_502M_FIXED_VALIDATION="${SAVED_CONFIRMATION}"

SECONDS=0
"${PYTHON}" examples/run_generative_neural_hmsc_iid_v1_production.py \
  fixed-validation \
  --freeze-root "${TRAINING_ROOT}" \
  --output "${RUN_ROOT}" \
  --expected-source-commit "${EXPECTED_SOURCE_COMMIT}" \
  --expected-checkpoint-content-sha256 "${EXPECTED_CHECKPOINT_CONTENT_SHA256}" \
  --expected-ablation-content-sha256 "${EXPECTED_ABLATION_CONTENT_SHA256}" \
  --release-registry "${RELEASE_REGISTRY}" \
  --python "${PYTHON}" \
  > "${RUN_ROOT}.stdout.json"

"${PYTHON}" examples/run_generative_neural_hmsc_iid_v1_production.py \
  validate-fixed-validation \
  --root "${RUN_ROOT}" \
  --expected-source-commit "${EXPECTED_SOURCE_COMMIT}" \
  --expected-training-freeze-sha256 "${EXPECTED_TRAINING_FREEZE_SHA256}" \
  > "${RUN_ROOT}/read_only_validation.json"

printf 'wall_time_seconds=%s\n' "${SECONDS}" | tee "${RUN_ROOT}/wall_time.txt"
echo "502M fixed validation completed: ${RUN_ROOT}"
echo "503M-505M and 511M-515M remain sealed."
