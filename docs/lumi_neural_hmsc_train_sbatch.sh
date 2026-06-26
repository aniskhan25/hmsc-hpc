#!/bin/bash -l
#SBATCH --job-name=neural-hmsc-train
#SBATCH --account=project_462000131
#SBATCH --partition=dev-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=output/%x-%j.out
#SBATCH --error=output/%x-%j.err

set -euo pipefail

# Neural-HMSC prototype training workflow for LUMI.
# Submit from the hmsc-hpc repository root:
#
#   sbatch docs/lumi_neural_hmsc_train_sbatch.sh
#
# Large generated datasets and posterior artifacts are written under
# ${RUN_ROOT}, which defaults to scratch outside the repository. Common
# overrides:
#
#   RUN_NAME=neural_train_large EPOCHS=120 TRAIN_DATASETS=512 \
#     sbatch docs/lumi_neural_hmsc_train_sbatch.sh

PROJECT_ID="${PROJECT_ID:-project_462000131}"
USER_WORK="${USER_WORK:-/scratch/${PROJECT_ID}/anisrahm}"
VENV="${VENV:-${USER_WORK}/venvs/hmsc_tf_env}"
PYTHON="${PYTHON:-${VENV}/bin/python3}"
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
RUN_NAME="${RUN_NAME:-neural_hmsc_train_${SLURM_JOB_ID:-manual}}"
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
SEED="${SEED:-20260626}"
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

echo "Repository: ${REPO_DIR}"
echo "Run root: ${RUN_ROOT}"
echo "Benchmark output: ${OUTPUT_DIR}"
echo "Suite: ${SUITE}"
echo "Shape: sites=${N_SITES}, species=${N_SPECIES}"
echo "Training datasets: ${TRAIN_DATASETS}; calibration datasets: ${CALIBRATION_DATASETS}"
echo "Epochs/batch size: ${EPOCHS}/${BATCH_SIZE}"
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
)
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  args+=(--skip-existing)
fi

srun "${PYTHON}" "${args[@]}"
printf 'wall_time_seconds=%s\n' "${SECONDS}" | tee "${RUN_ROOT}/wall_time.txt"

echo "Training workflow complete."
echo "Outputs are in ${OUTPUT_DIR}"
