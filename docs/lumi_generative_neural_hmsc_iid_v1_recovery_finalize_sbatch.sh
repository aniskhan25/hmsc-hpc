#!/bin/bash -l
#SBATCH --job-name=gen-iid-502-recovery-finalize
#SBATCH --account=project_462000131
#SBATCH --partition=dev-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=output/%x-%j.out
#SBATCH --error=output/%x-%j.err

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-project_462000131}"
USER_WORK="${USER_WORK:-/scratch/${PROJECT_ID}/anisrahm}"
VENV="${VENV:-${USER_WORK}/venvs/hmsc_tf_env}"
PYTHON="${PYTHON:-${VENV}/bin/python3}"
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
TRAINING_ROOT="${TRAINING_ROOT:?Set TRAINING_ROOT to the frozen 501M run}"
SHARD_ROOT="${SHARD_ROOT:?Set the complete 502M recovery shard root}"
RUN_ROOT="${RUN_ROOT:?Set a fresh final 502M recovery root}"
RELEASE_REGISTRY="${RELEASE_REGISTRY:-${USER_WORK}/hmsc-hpc-deployments}"
EXPECTED_TRAINING_SOURCE_COMMIT="${EXPECTED_TRAINING_SOURCE_COMMIT:?Pin the 501M source commit}"
EXPECTED_EVALUATOR_SOURCE_COMMIT="${EXPECTED_EVALUATOR_SOURCE_COMMIT:?Pin the recovery evaluator commit}"
EXPECTED_TRAINING_FREEZE_SHA256="${EXPECTED_TRAINING_FREEZE_SHA256:?Pin freeze.json}"
EXPECTED_CHECKPOINT_CONTENT_SHA256="${EXPECTED_CHECKPOINT_CONTENT_SHA256:?Pin candidate content}"
EXPECTED_ABLATION_CONTENT_SHA256="${EXPECTED_ABLATION_CONTENT_SHA256:?Pin ablation content}"
OPEN_GENERATIVE_IID_502M_TIMEOUT_RECOVERY_FINALIZER="${OPEN_GENERATIVE_IID_502M_TIMEOUT_RECOVERY_FINALIZER:-}"
EXPECTED_CONFIRMATION="FINALIZE_502M_TIMEOUT_RECOVERY_ONCE"

if [[ "${OPEN_GENERATIVE_IID_502M_TIMEOUT_RECOVERY_FINALIZER}" != "${EXPECTED_CONFIRMATION}" ]]; then
  echo "Refusing to finalize the 502M timeout recovery." >&2
  exit 2
fi
for name in \
  OPEN_GENERATIVE_IID_501M_TRAINING \
  OPEN_GENERATIVE_IID_502M_FIXED_VALIDATION \
  OPEN_GENERATIVE_IID_502M_TIMEOUT_RECOVERY
do
  if [[ -n "${!name:-}" ]]; then
    echo "Recovery finalizer refuses conflicting token ${name}." >&2
    exit 2
  fi
done
if [[ -e "${RUN_ROOT}" ]]; then
  echo "Refusing to reuse final recovery root: ${RUN_ROOT}" >&2
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

if [[ "$(git rev-parse HEAD)" != "${EXPECTED_EVALUATOR_SOURCE_COMMIT}" ]]; then
  echo "Repository HEAD differs from the recovery evaluator commit." >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Recovery finalization requires a clean repository worktree." >&2
  exit 2
fi
HOST_SOURCE_BRANCH="$(git branch --show-current)"
if [[ -z "${HOST_SOURCE_BRANCH}" ]]; then
  HOST_SOURCE_BRANCH="detached"
fi
export GENERATIVE_IID_HOST_SOURCE_COMMIT="${EXPECTED_EVALUATOR_SOURCE_COMMIT}"
export GENERATIVE_IID_HOST_SOURCE_BRANCH="${HOST_SOURCE_BRANCH}"
export GENERATIVE_IID_HOST_WORKTREE_CLEAN="1"
export SINGULARITYENV_GENERATIVE_IID_HOST_SOURCE_COMMIT="${EXPECTED_EVALUATOR_SOURCE_COMMIT}"
export SINGULARITYENV_GENERATIVE_IID_HOST_SOURCE_BRANCH="${HOST_SOURCE_BRANCH}"
export SINGULARITYENV_GENERATIVE_IID_HOST_WORKTREE_CLEAN="1"
export APPTAINERENV_GENERATIVE_IID_HOST_SOURCE_COMMIT="${EXPECTED_EVALUATOR_SOURCE_COMMIT}"
export APPTAINERENV_GENERATIVE_IID_HOST_SOURCE_BRANCH="${HOST_SOURCE_BRANCH}"
export APPTAINERENV_GENERATIVE_IID_HOST_WORKTREE_CLEAN="1"

"${PYTHON}" examples/run_generative_neural_hmsc_iid_v1_production.py \
  finalize-fixed-validation-recovery \
  --freeze-root "${TRAINING_ROOT}" \
  --shard-root "${SHARD_ROOT}" \
  --output "${RUN_ROOT}" \
  --expected-training-source-commit "${EXPECTED_TRAINING_SOURCE_COMMIT}" \
  --expected-evaluator-source-commit "${EXPECTED_EVALUATOR_SOURCE_COMMIT}" \
  --expected-training-freeze-sha256 "${EXPECTED_TRAINING_FREEZE_SHA256}" \
  --expected-checkpoint-content-sha256 "${EXPECTED_CHECKPOINT_CONTENT_SHA256}" \
  --expected-ablation-content-sha256 "${EXPECTED_ABLATION_CONTENT_SHA256}" \
  --release-registry "${RELEASE_REGISTRY}"

unset OPEN_GENERATIVE_IID_502M_TIMEOUT_RECOVERY_FINALIZER
"${PYTHON}" examples/run_generative_neural_hmsc_iid_v1_production.py \
  validate-fixed-validation-recovery \
  --root "${RUN_ROOT}" \
  --shard-root "${SHARD_ROOT}" \
  --expected-training-source-commit "${EXPECTED_TRAINING_SOURCE_COMMIT}" \
  --expected-evaluator-source-commit "${EXPECTED_EVALUATOR_SOURCE_COMMIT}" \
  --expected-training-freeze-sha256 "${EXPECTED_TRAINING_FREEZE_SHA256}" \
  --expected-checkpoint-content-sha256 "${EXPECTED_CHECKPOINT_CONTENT_SHA256}" \
  --expected-ablation-content-sha256 "${EXPECTED_ABLATION_CONTENT_SHA256}" \
  > "${RUN_ROOT}/read_only_recovery_validation.json"

echo "Sharded 502M timeout recovery finalized: ${RUN_ROOT}"
echo "503M-515M remain sealed."
