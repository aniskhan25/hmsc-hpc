#!/bin/bash -l
#SBATCH --job-name=pyhmsc-spat-repl
#SBATCH --account=project_462000131
#SBATCH --partition=dev-g
#SBATCH --array=0-2%3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=output/%x-%A_%a.out
#SBATCH --error=output/%x-%A_%a.err

set -euo pipefail

PROJECT_ID="project_462000131"
USER_WORK="/scratch/${PROJECT_ID}/anisrahm"
VENV="${USER_WORK}/venvs/hmsc_tf_env"
PYTHON="${VENV}/bin/python3"
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
RUN_NAME="${RUN_NAME:-replicated_spatial_holdout_validation}"
RUN_ROOT="${USER_WORK}/hmsc-hpc-runs/${RUN_NAME}"
PROJECT_ROOT="${PROJECT_ROOT:-${RUN_ROOT}/projects}"
MANIFEST="${MANIFEST:-${PROJECT_ROOT}/tasks.csv}"
SEED_TASK_INDEX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
TASKS_PER_SEED=6

mkdir -p output "${RUN_ROOT}/tasks"
if [[ ! -s "${MANIFEST}" ]]; then
  echo "Missing task manifest: ${MANIFEST}" >&2
  exit 2
fi

module use /appl/local/csc/modulefiles
module load tensorflow/2.16
source "${VENV}/bin/activate"
cd "${REPO_DIR}"
export PYHMSC_REPO_ROOT="${USER_WORK}/hmsc-hpc"
export PYTHONPATH="${PYHMSC_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

run_task() {
  local task_index="$1"
  local task_id seed ordering model project
  local task_root config compiled posterior prediction
  IFS=, read -r task_id seed ordering model project < <(sed -n "$((task_index + 2))p" "${MANIFEST}")
  if [[ -z "${task_id:-}" || "${task_id}" != "${task_index}" ]]; then
    echo "Manifest task ${task_id:-missing} does not match task index ${task_index}" >&2
    exit 2
  fi

  task_root="${RUN_ROOT}/tasks/task_$(printf '%03d' "${task_index}")"
  config="${project}/model_${model}.yaml"
  compiled="${task_root}/compiled"
  posterior="${task_root}/posterior.h5"
  prediction="${task_root}/prediction.csv"
  mkdir -p "${task_root}"

  echo
  echo "Task: ${task_id}"
  echo "Seed: ${seed}"
  echo "Ordering: ${ordering}"
  echo "Model: ${model}"
  echo "Project: ${project}"
  echo "Task root: ${task_root}"

  if [[ "${SKIP_EXISTING:-1}" == "1" && -s "${compiled}/init.json" ]]; then
    echo "Compiled init exists; skipping compile."
  else
    "${PYTHON}" -m pyhmsc compile "${config}" --output "${compiled}"
  fi
  "${PYTHON}" -m pyhmsc validate-init "${compiled}/init.json" --strict

  if [[ "${SKIP_EXISTING:-1}" == "1" && -s "${posterior}" ]]; then
    echo "Posterior exists; skipping sample."
  else
    srun "${PYTHON}" -m pyhmsc sample \
      "${compiled}/init.json" \
      --output "${posterior}" \
      --samples "${SAMPLES:-1000}" \
      --transient "${TRANSIENT:-500}" \
      --thin "${THIN:-10}" \
      --verbose "${VERBOSE:-100}"
  fi

  if [[ "${model}" == "fixed" ]]; then
    "${PYTHON}" -m pyhmsc predict "${posterior}" \
      --X "${project}/data/test/X.csv" \
      --model-config "${config}" \
      --output "${prediction}"
  else
    "${PYTHON}" -m pyhmsc predict "${posterior}" \
      --X "${project}/data/test/X.csv" \
      --model-config "${config}" \
      --study-design "${project}/data/test/study_design.csv" \
      --coords "${project}/data/test/coords.csv" \
      --random-effects known \
      --spatial-prediction conditional \
      --seed "${PREDICTION_SEED:-17}" \
      --output "${prediction}"
  fi
  echo "Completed task ${task_id}: ${prediction}"
}

first_task=$((SEED_TASK_INDEX * TASKS_PER_SEED))
for task_index in $(seq "${first_task}" "$((first_task + TASKS_PER_SEED - 1))"); do
  run_task "${task_index}"
done
