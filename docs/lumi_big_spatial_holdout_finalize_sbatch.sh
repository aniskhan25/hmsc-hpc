#!/bin/bash -l
#SBATCH --job-name=pyhmsc-big-final
#SBATCH --account=project_462000131
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=output/%x-%j.out
#SBATCH --error=output/%x-%j.err

set -euo pipefail

PROJECT_ID="project_462000131"
USER_WORK="/scratch/${PROJECT_ID}/anisrahm"
VENV="${USER_WORK}/venvs/hmsc_tf_env"
PYTHON="${VENV}/bin/python3"
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
RUN_NAME="${RUN_NAME:-big_spatial_holdout_validation_real}"
RUN_ROOT="${USER_WORK}/hmsc-hpc-runs/${RUN_NAME}"
MODEL_ROOT="${RUN_ROOT}/spatial_nngp"
CHAIN_DIR="${MODEL_ROOT}/chains"
POSTERIOR="${MODEL_ROOT}/posterior.h5"
RESOURCE="${MODEL_ROOT}/resource_metrics.txt"

mkdir -p output
module use /appl/local/csc/modulefiles
module load tensorflow/2.16
source "${VENV}/bin/activate"
cd "${REPO_DIR}"
export PYTHONPATH="${USER_WORK}/hmsc-hpc${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON}" -m pyhmsc chain-status "${CHAIN_DIR}" \
  --expected-chains 0 1 --expected-draws "${SAMPLES:-1000}" \
  --run-name "${RUN_NAME}" --strict
"${PYTHON}" -m pyhmsc merge \
  "${CHAIN_DIR}/posterior_chain_0.h5" \
  "${CHAIN_DIR}/posterior_chain_1.h5" \
  --expected-chains 0 1 --output "${POSTERIOR}"

elapsed_seconds=$(awk -F= '/^elapsed_seconds=/{sum += $2} END {print sum + 0}' "${CHAIN_DIR}"/resource_chain_*.txt)
max_rss_kb=$(awk -F= '/^max_rss_kb=/{if ($2 > max) max = $2} END {print max + 0}' "${CHAIN_DIR}"/resource_chain_*.txt)
compiled_bytes=$(du -cb "${MODEL_ROOT}/compiled/init.json" "${MODEL_ROOT}/compiled/init_arrays.h5" | tail -1 | cut -f1)
posterior_bytes=$(stat -c %s "${POSTERIOR}")
printf 'elapsed_seconds=%s\nmax_rss_kb=%s\ncompiled_bytes=%s\nposterior_bytes=%s\n' \
  "${elapsed_seconds}" "${max_rss_kb}" "${compiled_bytes}" "${posterior_bytes}" \
  > "${RESOURCE}"

"${PYTHON}" examples/analyze_big_spatial_holdout_validation.py \
  --project "${RUN_ROOT}/project" \
  --run-root "${RUN_ROOT}" \
  --output "${RUN_ROOT}/big_spatial_holdout_report.txt"

echo "Report: ${RUN_ROOT}/big_spatial_holdout_report.txt"
