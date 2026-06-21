#!/bin/bash -l

set -euo pipefail

PROJECT_ID="project_462000131"
USER_WORK="/scratch/${PROJECT_ID}/anisrahm"
VENV="${USER_WORK}/venvs/hmsc_tf_env"
PYTHON="${VENV}/bin/python3"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_NAME="${RUN_NAME:-replicated_spatial_holdout_validation}"
RUN_ROOT="${USER_WORK}/hmsc-hpc-runs/${RUN_NAME}"
PROJECT_ROOT="${PROJECT_ROOT:-${RUN_ROOT}/projects}"

mkdir -p "${REPO_DIR}/output" "${RUN_ROOT}"
module use /appl/local/csc/modulefiles
module load tensorflow/2.16
source "${VENV}/bin/activate"
cd "${REPO_DIR}"
export PYHMSC_REPO_ROOT="${USER_WORK}/hmsc-hpc"
export PYTHONPATH="${PYHMSC_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON}" examples/generate_replicated_spatial_holdout_validation.py \
  --output "${PROJECT_ROOT}" \
  --seeds ${SEEDS:-321 654 987}

task_count=$(($(wc -l < "${PROJECT_ROOT}/tasks.csv") - 1))
if [[ "${task_count}" -le 0 ]]; then
  echo "Generated manifest contains no tasks." >&2
  exit 2
fi
tasks_per_seed=6
if ((task_count % tasks_per_seed != 0)); then
  echo "Task count ${task_count} is not divisible by ${tasks_per_seed}." >&2
  exit 2
fi
seed_count=$((task_count / tasks_per_seed))
seed_array="${SEED_ARRAY:-0-$((seed_count > 1 ? 1 : 0))}"
array_job=$(sbatch --parsable \
  --array="${seed_array}%${MAX_CONCURRENT:-2}" \
  --export=ALL,RUN_NAME="${RUN_NAME}",PROJECT_ROOT="${PROJECT_ROOT}" \
  docs/lumi_replicated_spatial_holdout_array_sbatch.sh)

echo "Array job: ${array_job}"
if [[ "${SUBMIT_ANALYSIS:-0}" == "1" ]]; then
  analysis_job=$(sbatch --parsable \
    --dependency="afterok:${array_job}" \
    --export=ALL,RUN_NAME="${RUN_NAME}",PROJECT_ROOT="${PROJECT_ROOT}" \
    docs/lumi_replicated_spatial_holdout_analyze_sbatch.sh)
  echo "Analysis job: ${analysis_job}"
else
  echo "Analysis job not submitted. Submit the final seed wave with SUBMIT_ANALYSIS=1."
fi
echo "Run root: ${RUN_ROOT}"
