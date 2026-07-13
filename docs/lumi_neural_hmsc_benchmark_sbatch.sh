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
SBC_DATASETS="${SBC_DATASETS:-32}"
SBC_DRAWS="${SBC_DRAWS:-256}"
SBC_BINS="${SBC_BINS:-10}"
OOD_REGIMES="${OOD_REGIMES:-covariate_shift effect_size_shift combined_shift}"
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
NEURAL_CHECKPOINT="${NEURAL_CHECKPOINT:-}"
PROBIT_ANCHOR="${PROBIT_ANCHOR:-auto}"
PROBIT_ANCHOR_ITERATIONS="${PROBIT_ANCHOR_ITERATIONS:-8}"
PROBIT_ANCHOR_PRIOR_PRECISION="${PROBIT_ANCHOR_PRIOR_PRECISION:-1.0}"
PROBIT_ANCHOR_ETA_CLIP="${PROBIT_ANCHOR_ETA_CLIP:-6.0}"
RUN_MCMC_REFERENCE="${RUN_MCMC_REFERENCE:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
COEFFICIENT_CALIBRATION="${COEFFICIENT_CALIBRATION:-scalar}"
CONDITIONAL_CALIBRATION_EPOCHS="${CONDITIONAL_CALIBRATION_EPOCHS:-400}"
CONDITIONAL_CALIBRATION_LEARNING_RATE="${CONDITIONAL_CALIBRATION_LEARNING_RATE:-0.03}"
CONDITIONAL_CALIBRATION_REGULARIZATION="${CONDITIONAL_CALIBRATION_REGULARIZATION:-0.001}"
CONDITIONAL_CALIBRATION_RANK_PENALTY_WEIGHT="${CONDITIONAL_CALIBRATION_RANK_PENALTY_WEIGHT:-0.02}"
CONDITIONAL_CALIBRATION_RARE_WEIGHT="${CONDITIONAL_CALIBRATION_RARE_WEIGHT:-4.0}"
CONDITIONAL_CALIBRATION_INTERMEDIATE_WEIGHT="${CONDITIONAL_CALIBRATION_INTERMEDIATE_WEIGHT:-2.0}"
CONDITIONAL_CALIBRATION_COMMON_WEIGHT="${CONDITIONAL_CALIBRATION_COMMON_WEIGHT:-1.0}"
CONDITIONAL_CALIBRATION_SUPPORT_QUANTILE="${CONDITIONAL_CALIBRATION_SUPPORT_QUANTILE:-0.99}"
CONDITIONAL_CALIBRATION_FALLBACK_STRENGTH="${CONDITIONAL_CALIBRATION_FALLBACK_STRENGTH:-2.0}"
CONDITIONAL_CALIBRATION_OOD_UNCERTAINTY_STRENGTH="${CONDITIONAL_CALIBRATION_OOD_UNCERTAINTY_STRENGTH:-0.75}"
CONDITIONAL_CALIBRATION_OOD_UNCERTAINTY_MAX_MULTIPLIER="${CONDITIONAL_CALIBRATION_OOD_UNCERTAINTY_MAX_MULTIPLIER:-4.0}"
CONDITIONAL_CALIBRATION_OOD_OBJECTIVE="${CONDITIONAL_CALIBRATION_OOD_OBJECTIVE:-none}"
CONDITIONAL_CALIBRATION_OOD_DATASETS="${CONDITIONAL_CALIBRATION_OOD_DATASETS:-0}"
CONDITIONAL_CALIBRATION_OOD_OBJECTIVE_WEIGHT="${CONDITIONAL_CALIBRATION_OOD_OBJECTIVE_WEIGHT:-1.0}"
CONDITIONAL_CALIBRATION_OOD_IN_DOMAIN_GATE_WEIGHT="${CONDITIONAL_CALIBRATION_OOD_IN_DOMAIN_GATE_WEIGHT:-10.0}"
CONDITIONAL_CALIBRATION_OOD_OBJECTIVE_EPOCHS="${CONDITIONAL_CALIBRATION_OOD_OBJECTIVE_EPOCHS:-200}"
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
echo "SBC datasets/draws/bins: ${SBC_DATASETS}/${SBC_DRAWS}/${SBC_BINS}; OOD: ${OOD_REGIMES}"
echo "Epochs/batch size: ${EPOCHS}/${BATCH_SIZE}"
echo "Coefficient calibration: ${COEFFICIENT_CALIBRATION}"
echo "Conditional calibration epochs/learning rate/regularization: ${CONDITIONAL_CALIBRATION_EPOCHS}/${CONDITIONAL_CALIBRATION_LEARNING_RATE}/${CONDITIONAL_CALIBRATION_REGULARIZATION}"
echo "Conditional rank penalty/prevalence weights: ${CONDITIONAL_CALIBRATION_RANK_PENALTY_WEIGHT}/${CONDITIONAL_CALIBRATION_RARE_WEIGHT},${CONDITIONAL_CALIBRATION_INTERMEDIATE_WEIGHT},${CONDITIONAL_CALIBRATION_COMMON_WEIGHT}"
echo "Conditional support quantile/fallback strength: ${CONDITIONAL_CALIBRATION_SUPPORT_QUANTILE}/${CONDITIONAL_CALIBRATION_FALLBACK_STRENGTH}"
echo "Conditional OOD uncertainty strength/max multiplier: ${CONDITIONAL_CALIBRATION_OOD_UNCERTAINTY_STRENGTH}/${CONDITIONAL_CALIBRATION_OOD_UNCERTAINTY_MAX_MULTIPLIER}"
echo "Conditional OOD objective/datasets/epochs: ${CONDITIONAL_CALIBRATION_OOD_OBJECTIVE}/${CONDITIONAL_CALIBRATION_OOD_DATASETS}/${CONDITIONAL_CALIBRATION_OOD_OBJECTIVE_EPOCHS}"
echo "Conditional OOD objective/gate weights: ${CONDITIONAL_CALIBRATION_OOD_OBJECTIVE_WEIGHT}/${CONDITIONAL_CALIBRATION_OOD_IN_DOMAIN_GATE_WEIGHT}"
echo "MCMC chains/samples/transient/thin: ${MCMC_CHAINS}/${MCMC_SAMPLES}/${MCMC_TRANSIENT}/${MCMC_THIN}"
echo "Simulation/model seed: ${SEED}/${MODEL_SEED}"
echo "Frozen checkpoint: ${NEURAL_CHECKPOINT:-none}"
echo "Probit anchor/iterations/prior precision/eta clip: ${PROBIT_ANCHOR}/${PROBIT_ANCHOR_ITERATIONS}/${PROBIT_ANCHOR_PRIOR_PRECISION}/${PROBIT_ANCHOR_ETA_CLIP}"
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
  --sbc-datasets "${SBC_DATASETS}"
  --sbc-draws "${SBC_DRAWS}"
  --sbc-bins "${SBC_BINS}"
  --ood-regimes ${OOD_REGIMES}
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --coefficient-calibration "${COEFFICIENT_CALIBRATION}"
  --conditional-calibration-epochs "${CONDITIONAL_CALIBRATION_EPOCHS}"
  --conditional-calibration-learning-rate "${CONDITIONAL_CALIBRATION_LEARNING_RATE}"
  --conditional-calibration-regularization "${CONDITIONAL_CALIBRATION_REGULARIZATION}"
  --conditional-calibration-rank-penalty-weight "${CONDITIONAL_CALIBRATION_RANK_PENALTY_WEIGHT}"
  --conditional-calibration-rare-weight "${CONDITIONAL_CALIBRATION_RARE_WEIGHT}"
  --conditional-calibration-intermediate-weight "${CONDITIONAL_CALIBRATION_INTERMEDIATE_WEIGHT}"
  --conditional-calibration-common-weight "${CONDITIONAL_CALIBRATION_COMMON_WEIGHT}"
  --conditional-calibration-support-quantile "${CONDITIONAL_CALIBRATION_SUPPORT_QUANTILE}"
  --conditional-calibration-fallback-strength "${CONDITIONAL_CALIBRATION_FALLBACK_STRENGTH}"
  --conditional-calibration-ood-uncertainty-strength "${CONDITIONAL_CALIBRATION_OOD_UNCERTAINTY_STRENGTH}"
  --conditional-calibration-ood-uncertainty-max-multiplier "${CONDITIONAL_CALIBRATION_OOD_UNCERTAINTY_MAX_MULTIPLIER}"
  --conditional-calibration-ood-objective "${CONDITIONAL_CALIBRATION_OOD_OBJECTIVE}"
  --conditional-calibration-ood-datasets "${CONDITIONAL_CALIBRATION_OOD_DATASETS}"
  --conditional-calibration-ood-objective-weight "${CONDITIONAL_CALIBRATION_OOD_OBJECTIVE_WEIGHT}"
  --conditional-calibration-ood-in-domain-gate-weight "${CONDITIONAL_CALIBRATION_OOD_IN_DOMAIN_GATE_WEIGHT}"
  --conditional-calibration-ood-objective-epochs "${CONDITIONAL_CALIBRATION_OOD_OBJECTIVE_EPOCHS}"
  --neural-chains "${NEURAL_CHAINS}"
  --neural-draws "${NEURAL_DRAWS}"
  --seed "${SEED}"
  --model-seed "${MODEL_SEED}"
  --probit-anchor "${PROBIT_ANCHOR}"
  --probit-anchor-iterations "${PROBIT_ANCHOR_ITERATIONS}"
  --probit-anchor-prior-precision "${PROBIT_ANCHOR_PRIOR_PRECISION}"
  --probit-anchor-eta-clip "${PROBIT_ANCHOR_ETA_CLIP}"
  --mcmc-chains "${MCMC_CHAINS}"
  --mcmc-samples "${MCMC_SAMPLES}"
  --mcmc-transient "${MCMC_TRANSIENT}"
  --mcmc-thin "${MCMC_THIN}"
  --mcmc-verbose "${MCMC_VERBOSE}"
)
if [[ "${RUN_MCMC_REFERENCE}" == "1" ]]; then
  args+=(--run-mcmc-reference)
fi
if [[ -n "${NEURAL_CHECKPOINT}" ]]; then
  args+=(--checkpoint "${NEURAL_CHECKPOINT}")
fi
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  args+=(--skip-existing)
fi

"${PYTHON}" "${args[@]}"
printf 'wall_time_seconds=%s\n' "${SECONDS}" | tee "${RUN_ROOT}/wall_time.txt"

echo "Benchmark workflow complete."
echo "Outputs are in ${OUTPUT_DIR}"
