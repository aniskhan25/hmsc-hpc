#!/bin/bash -l
#SBATCH --job-name=gen-iid-502-recovery-shard
#SBATCH --account=project_462000131
#SBATCH --partition=standard-g
#SBATCH --array=0-35%12
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=output/%x-%A_%a.out
#SBATCH --error=output/%x-%A_%a.err

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-project_462000131}"
USER_WORK="${USER_WORK:-/scratch/${PROJECT_ID}/anisrahm}"
VENV="${VENV:-${USER_WORK}/venvs/hmsc_tf_env}"
PYTHON="${PYTHON:-${VENV}/bin/python3}"
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
TRAINING_ROOT="${TRAINING_ROOT:?Set TRAINING_ROOT to the frozen 501M run}"
SHARD_ROOT="${SHARD_ROOT:?Set the fresh 502M recovery shard root}"
EXPECTED_TRAINING_SOURCE_COMMIT="${EXPECTED_TRAINING_SOURCE_COMMIT:?Pin the 501M source commit}"
EXPECTED_EVALUATOR_SOURCE_COMMIT="${EXPECTED_EVALUATOR_SOURCE_COMMIT:?Pin the recovery evaluator commit}"
EXPECTED_TRAINING_FREEZE_SHA256="${EXPECTED_TRAINING_FREEZE_SHA256:?Pin freeze.json}"
EXPECTED_CHECKPOINT_CONTENT_SHA256="${EXPECTED_CHECKPOINT_CONTENT_SHA256:?Pin candidate content}"
EXPECTED_ABLATION_CONTENT_SHA256="${EXPECTED_ABLATION_CONTENT_SHA256:?Pin ablation content}"
OPEN_GENERATIVE_IID_502M_TIMEOUT_RECOVERY="${OPEN_GENERATIVE_IID_502M_TIMEOUT_RECOVERY:-}"
EXPECTED_CONFIRMATION="RECOVER_502M_TIMEOUT_SHARDS_ONLY"

if [[ "${OPEN_GENERATIVE_IID_502M_TIMEOUT_RECOVERY}" != "${EXPECTED_CONFIRMATION}" ]]; then
  echo "Refusing to open a 502M recovery shard." >&2
  exit 2
fi
for name in \
  OPEN_GENERATIVE_IID_501M_TRAINING \
  OPEN_GENERATIVE_IID_502M_FIXED_VALIDATION \
  OPEN_GENERATIVE_IID_502M_TIMEOUT_RECOVERY_FINALIZER
do
  if [[ -n "${!name:-}" ]]; then
    echo "Recovery shard refuses conflicting token ${name}." >&2
    exit 2
  fi
done
if [[ ! -f "${TRAINING_ROOT}/freeze.json" ]]; then
  echo "Frozen 501M training root is missing." >&2
  exit 2
fi

mkdir -p output "${SHARD_ROOT}"
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
  echo "Recovery shards require a clean repository worktree." >&2
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
  fixed-validation-recovery-shard \
  --freeze-root "${TRAINING_ROOT}" \
  --shard-root "${SHARD_ROOT}" \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --expected-training-source-commit "${EXPECTED_TRAINING_SOURCE_COMMIT}" \
  --expected-evaluator-source-commit "${EXPECTED_EVALUATOR_SOURCE_COMMIT}" \
  --expected-training-freeze-sha256 "${EXPECTED_TRAINING_FREEZE_SHA256}" \
  --expected-checkpoint-content-sha256 "${EXPECTED_CHECKPOINT_CONTENT_SHA256}" \
  --expected-ablation-content-sha256 "${EXPECTED_ABLATION_CONTENT_SHA256}" \
  --python "${PYTHON}"

unset OPEN_GENERATIVE_IID_502M_TIMEOUT_RECOVERY
"${PYTHON}" examples/run_generative_neural_hmsc_iid_v1_production.py \
  validate-fixed-validation-recovery-shard \
  --shard-root "${SHARD_ROOT}" \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --expected-training-source-commit "${EXPECTED_TRAINING_SOURCE_COMMIT}" \
  --expected-evaluator-source-commit "${EXPECTED_EVALUATOR_SOURCE_COMMIT}" \
  --expected-training-freeze-sha256 "${EXPECTED_TRAINING_FREEZE_SHA256}" \
  --expected-checkpoint-content-sha256 "${EXPECTED_CHECKPOINT_CONTENT_SHA256}" \
  --expected-ablation-content-sha256 "${EXPECTED_ABLATION_CONTENT_SHA256}"

echo "502M recovery shard ${SLURM_ARRAY_TASK_ID} completed."
echo "503M-515M remain sealed."
