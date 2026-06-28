#!/bin/bash -l
#SBATCH --job-name=neural-hmsc-bench
#SBATCH --account=project_462000131
#SBATCH --partition=dev-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=output/%x-%j.out
#SBATCH --error=output/%x-%j.err

set -euo pipefail

# End-to-end Neural-HMSC benchmark workflow for LUMI.
# Submit from the hmsc-hpc repository root:
#
#   sbatch docs/lumi_neural_hmsc_benchmark_sbatch.sh
#
# This workflow generates fixed-effect benchmark data, trains neural posterior
# prototypes, runs Python-native MCMC references, and writes benchmark reports.
# Large artifacts are kept under ${RUN_ROOT} on scratch.

PROJECT_ID="${PROJECT_ID:-project_462000131}"
USER_WORK="${USER_WORK:-/scratch/${PROJECT_ID}/anisrahm}"
VENV="${VENV:-${USER_WORK}/venvs/hmsc_tf_env}"
PYTHON="${PYTHON:-${VENV}/bin/python3}"
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
RUN_NAME="${RUN_NAME:-neural_hmsc_benchmark_${SLURM_JOB_ID:-manual}}"
RUN_ROOT="${RUN_ROOT:-${USER_WORK}/hmsc-hpc-runs/${RUN_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_ROOT}/benchmark}"
SUITE="${SUITE:-normal probit poisson}"
N_SITES="${N_SITES:-64}"
N_SPECIES="${N_SPECIES:-4}"
TRAIN_DATASETS="${TRAIN_DATASETS:-128}"
CALIBRATION_DATASETS="${CALIBRATION_DATASETS:-32}"
EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NEURAL_CHAINS="${NEURAL_CHAINS:-4}"
NEURAL_DRAWS="${NEURAL_DRAWS:-500}"
MCMC_CHAINS="${MCMC_CHAINS:-2}"
MCMC_SAMPLES="${MCMC_SAMPLES:-500}"
MCMC_TRANSIENT="${MCMC_TRANSIENT:-250}"
MCMC_THIN="${MCMC_THIN:-5}"
MCMC_VERBOSE="${MCMC_VERBOSE:-100}"
SEED="${SEED:-20260626}"
MODEL_SEED="${MODEL_SEED:-${SEED}}"
RUN_MCMC_REFERENCE="${RUN_MCMC_REFERENCE:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
GPU_LOG_INTERVAL="${GPU_LOG_INTERVAL:-60}"

mkdir -p output "${RUN_ROOT}" "${OUTPUT_DIR}"

module use /appl/local/csc/modulefiles
module load tensorflow/2.16

if [[ ! -f "${VENV}/bin/activate" ]]; then
  echo "Missing virtual environment: ${VENV}" >&2
  echo "Create it with the LUMI TensorFlow module, then resubmit." >&2
  exit 2
fi

source "${VENV}/bin/activate"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export SINGULARITYENV_PYTHONPATH="${PYTHONPATH}"
export APPTAINERENV_PYTHONPATH="${PYTHONPATH}"

echo "Repository: ${REPO_DIR}"
echo "Run root: ${RUN_ROOT}"
echo "Benchmark output: ${OUTPUT_DIR}"
echo "Suite: ${SUITE}"
echo "Shape: sites=${N_SITES}, species=${N_SPECIES}"
echo "Training datasets: ${TRAIN_DATASETS}; calibration datasets: ${CALIBRATION_DATASETS}"
echo "Epochs/batch size: ${EPOCHS}/${BATCH_SIZE}"
echo "MCMC chains/samples/transient/thin: ${MCMC_CHAINS}/${MCMC_SAMPLES}/${MCMC_TRANSIENT}/${MCMC_THIN}"
echo "Simulation/model seed: ${SEED}/${MODEL_SEED}"
echo "Python: ${PYTHON}"
"${PYTHON}" -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"
"${PYTHON}" -c "import tensorflow_probability as tfp; print('TFP:', tfp.__version__)"
"${PYTHON}" -c "import h5py; print('h5py:', h5py.__version__)"
"${PYTHON}" -c "import hmsc, pyhmsc; print('hmsc-hpc import: ok')"

GPU_LOG="${RUN_ROOT}/gpu_utilization.log"
GPU_LOG_PID=""
if command -v rocm-smi >/dev/null 2>&1; then
  (
    while true; do
      date -u +"timestamp=%Y-%m-%dT%H:%M:%SZ"
      rocm-smi --showuse --showmemuse --csv
      sleep "${GPU_LOG_INTERVAL}"
    done
  ) > "${GPU_LOG}" 2>&1 &
  GPU_LOG_PID="$!"
  trap 'if [[ -n "${GPU_LOG_PID}" ]]; then kill "${GPU_LOG_PID}" 2>/dev/null || true; fi' EXIT
  echo "GPU utilization log: ${GPU_LOG}"
else
  echo "rocm-smi not found; GPU utilization logging disabled"
fi

SECONDS=0
args=(
  examples/run_neural_hmsc_benchmark.py
  --output "${OUTPUT_DIR}"
  --suite ${SUITE}
  --n-sites "${N_SITES}"
  --n-species "${N_SPECIES}"
  --train-datasets "${TRAIN_DATASETS}"
  --calibration-datasets "${CALIBRATION_DATASETS}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --neural-chains "${NEURAL_CHAINS}"
  --neural-draws "${NEURAL_DRAWS}"
  --seed "${SEED}"
  --model-seed "${MODEL_SEED}"
  --mcmc-chains "${MCMC_CHAINS}"
  --mcmc-samples "${MCMC_SAMPLES}"
  --mcmc-transient "${MCMC_TRANSIENT}"
  --mcmc-thin "${MCMC_THIN}"
  --mcmc-verbose "${MCMC_VERBOSE}"
)
if [[ "${RUN_MCMC_REFERENCE}" == "1" ]]; then
  args+=(--run-mcmc-reference)
fi
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  args+=(--skip-existing)
fi

"${PYTHON}" "${args[@]}"
printf 'wall_time_seconds=%s\n' "${SECONDS}" | tee "${RUN_ROOT}/wall_time.txt"

echo "Benchmark workflow complete."
echo "Outputs are in ${OUTPUT_DIR}"
