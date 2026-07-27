#!/bin/bash -l
#SBATCH --job-name=neural-real-sens
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

PROJECT_ID="${PROJECT_ID:-project_462000131}"
USER_WORK="${USER_WORK:-/scratch/${PROJECT_ID}/anisrahm}"
VENV="${VENV:-${USER_WORK}/venvs/hmsc_tf_env}"
PYTHON="${PYTHON:-${VENV}/bin/python3}"
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
RUN_NAME="${RUN_NAME:-neural_realdata_sensitivity_${SLURM_JOB_ID:-manual}}"
RUN_ROOT="${RUN_ROOT:-${USER_WORK}/hmsc-hpc-runs/${RUN_NAME}}"
SEEDS="${SEEDS:-20260721 20260722 20260723}"

TRAIN_DATASETS="${TRAIN_DATASETS:-512}"
CALIBRATION_DATASETS="${CALIBRATION_DATASETS:-128}"
SBC_DATASETS="${SBC_DATASETS:-128}"
SBC_DRAWS="${SBC_DRAWS:-512}"
EPOCHS="${EPOCHS:-120}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NEURAL_CHAINS="${NEURAL_CHAINS:-4}"
NEURAL_DRAWS="${NEURAL_DRAWS:-1000}"
MCMC_CHAINS="${MCMC_CHAINS:-2}"
MCMC_SAMPLES="${MCMC_SAMPLES:-1000}"
MCMC_TRANSIENT="${MCMC_TRANSIENT:-500}"
MCMC_THIN="${MCMC_THIN:-5}"
MCMC_VERBOSE="${MCMC_VERBOSE:-500}"
COEFFICIENT_CALIBRATION="${COEFFICIENT_CALIBRATION:-external_monotone}"
CONDITIONAL_CALIBRATION_EPOCHS="${CONDITIONAL_CALIBRATION_EPOCHS:-400}"
CONDITIONAL_CALIBRATION_LEARNING_RATE="${CONDITIONAL_CALIBRATION_LEARNING_RATE:-0.03}"
CONDITIONAL_CALIBRATION_REGULARIZATION="${CONDITIONAL_CALIBRATION_REGULARIZATION:-0.001}"
CONDITIONAL_CALIBRATION_RANK_PENALTY_WEIGHT="${CONDITIONAL_CALIBRATION_RANK_PENALTY_WEIGHT:-0.02}"
EXTERNAL_MONOTONE_DATASETS="${EXTERNAL_MONOTONE_DATASETS:-4}"
EXTERNAL_MONOTONE_MAX_MULTIPLIER="${EXTERNAL_MONOTONE_MAX_MULTIPLIER:-2.0}"
EXTERNAL_MONOTONE_MIN_OOD_GAIN="${EXTERNAL_MONOTONE_MIN_OOD_GAIN:-0.005}"
EXTERNAL_MONOTONE_MIN_COMBINED_GAIN="${EXTERNAL_MONOTONE_MIN_COMBINED_GAIN:-0.005}"
PREDICTIVE_MEAN_CALIBRATION="${PREDICTIVE_MEAN_CALIBRATION:-probit_response_affine}"
PREDICTIVE_MEAN_CALIBRATION_VALIDATION_DATASETS="${PREDICTIVE_MEAN_CALIBRATION_VALIDATION_DATASETS:-128}"
PREDICTIVE_MEAN_CALIBRATION_MAX_BRIER_RATIO="${PREDICTIVE_MEAN_CALIBRATION_MAX_BRIER_RATIO:-1.0}"
PREDICTIVE_MEAN_CALIBRATION_MAX_LOG_LOSS_RATIO="${PREDICTIVE_MEAN_CALIBRATION_MAX_LOG_LOSS_RATIO:-1.0}"
PREDICTIVE_MEAN_CALIBRATION_MIN_IMPROVEMENT="${PREDICTIVE_MEAN_CALIBRATION_MIN_IMPROVEMENT:-0.0001}"
PREDICTIVE_MEAN_SOURCE_MIN_IMPROVEMENT="${PREDICTIVE_MEAN_SOURCE_MIN_IMPROVEMENT:-0.0005}"
PREDICTIVE_MEAN_TRANSFER_BRANCH_MIN_IMPROVEMENT="${PREDICTIVE_MEAN_TRANSFER_BRANCH_MIN_IMPROVEMENT:-0.0001}"
PREDICTIVE_MEAN_SELECTION_POLICY="${PREDICTIVE_MEAN_SELECTION_POLICY:-domain_conditional}"
PREDICTIVE_MEAN_TRANSFER_MIN_BRIER_GAIN="${PREDICTIVE_MEAN_TRANSFER_MIN_BRIER_GAIN:-0.0001}"
PREDICTIVE_MEAN_TRANSFER_MIN_LOG_LOSS_GAIN="${PREDICTIVE_MEAN_TRANSFER_MIN_LOG_LOSS_GAIN:-0.0005}"
PREDICTIVE_MEAN_TRANSFER_MAX_SLOPE_DELTA="${PREDICTIVE_MEAN_TRANSFER_MAX_SLOPE_DELTA:-0.05}"
PREDICTIVE_MEAN_TRANSFER_MAX_ABS_INTERCEPT="${PREDICTIVE_MEAN_TRANSFER_MAX_ABS_INTERCEPT:-0.025}"
TARGET_CONTEXT_GATE="${TARGET_CONTEXT_GATE:-none}"
TARGET_CONTEXT_GATE_DATASETS="${TARGET_CONTEXT_GATE_DATASETS:-12}"
TARGET_CONTEXT_GATE_MAX_BRIER_RATIO="${TARGET_CONTEXT_GATE_MAX_BRIER_RATIO:-1.0}"
TARGET_CONTEXT_GATE_MAX_LOG_LOSS_RATIO="${TARGET_CONTEXT_GATE_MAX_LOG_LOSS_RATIO:-1.0}"
TARGET_CONTEXT_GATE_MIN_IMPROVEMENT="${TARGET_CONTEXT_GATE_MIN_IMPROVEMENT:-0.0001}"
PREDICTIVE_PROMOTION_MAX_BRIER_RATIO="${PREDICTIVE_PROMOTION_MAX_BRIER_RATIO:-1.0}"
PREDICTIVE_PROMOTION_MAX_LOG_LOSS_RATIO="${PREDICTIVE_PROMOTION_MAX_LOG_LOSS_RATIO:-1.0}"
PREDICTIVE_PROMOTION_MAX_RMSE_RATIO="${PREDICTIVE_PROMOTION_MAX_RMSE_RATIO:-1.0}"
PREDICTIVE_PROMOTION_MAX_RICHNESS_RATIO="${PREDICTIVE_PROMOTION_MAX_RICHNESS_RATIO:-1.0}"
PREDICTIVE_PROMOTION_MIN_MEAN_BRIER_GAIN="${PREDICTIVE_PROMOTION_MIN_MEAN_BRIER_GAIN:-0.000001}"
PREDICTIVE_PROMOTION_MIN_MEAN_LOG_LOSS_GAIN="${PREDICTIVE_PROMOTION_MIN_MEAN_LOG_LOSS_GAIN:-0.000001}"
SIMULATED_PREDICTIVE_SUMMARY="${SIMULATED_PREDICTIVE_SUMMARY:-${USER_WORK}/hmsc-hpc-runs/neural_hmsc_transfer_response_mean_production_eval_20260720/response_mean_production_predictive_summary.csv}"
QUALIFIED_REFERENCE_LABEL="${QUALIFIED_REFERENCE_LABEL:-qualified_python_mcmc_fixed}"
WHITTAKER_REFERENCE_PARITY_METRICS="${WHITTAKER_REFERENCE_PARITY_METRICS:-${USER_WORK}/hmsc-hpc-runs/whittaker_r_python_parity_scaled_20260718_082539/whittaker_r_python_parity_metrics.json}"
BIG_SPATIAL_REFERENCE_PARITY_METRICS="${BIG_SPATIAL_REFERENCE_PARITY_METRICS:-${USER_WORK}/hmsc-hpc-runs/direct_r_python_big_spatial_full_parity_20260719/big_spatial_plants_validation_model_spatial_full/direct_r_python_parity_metrics.json}"

mkdir -p output "${RUN_ROOT}"
module use /appl/local/csc/modulefiles
module load tensorflow/2.16
source "${VENV}/bin/activate"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export SINGULARITYENV_PYTHONPATH="${PYTHONPATH}"
export APPTAINERENV_PYTHONPATH="${PYTHONPATH}"

echo "Repository: ${REPO_DIR}"
echo "Run root: ${RUN_ROOT}"
echo "Seeds: ${SEEDS}"
echo "Predictive mean calibration: ${PREDICTIVE_MEAN_CALIBRATION}"
echo "Predictive mean selection policy: ${PREDICTIVE_MEAN_SELECTION_POLICY}"
echo "Predictive mean branch margins: source=${PREDICTIVE_MEAN_SOURCE_MIN_IMPROVEMENT}, transfer=${PREDICTIVE_MEAN_TRANSFER_BRANCH_MIN_IMPROVEMENT}"
echo "Predictive mean transfer guard: brier>=${PREDICTIVE_MEAN_TRANSFER_MIN_BRIER_GAIN}, log_loss>=${PREDICTIVE_MEAN_TRANSFER_MIN_LOG_LOSS_GAIN}, slope_delta<=${PREDICTIVE_MEAN_TRANSFER_MAX_SLOPE_DELTA}, abs_intercept<=${PREDICTIVE_MEAN_TRANSFER_MAX_ABS_INTERCEPT}"
echo "Target-context gate: ${TARGET_CONTEXT_GATE}, datasets=${TARGET_CONTEXT_GATE_DATASETS}, brier<=${TARGET_CONTEXT_GATE_MAX_BRIER_RATIO}, log_loss<=${TARGET_CONTEXT_GATE_MAX_LOG_LOSS_RATIO}, improvement>=${TARGET_CONTEXT_GATE_MIN_IMPROVEMENT}"
echo "Cross-dataset gates: brier<=${PREDICTIVE_PROMOTION_MAX_BRIER_RATIO}, log_loss<=${PREDICTIVE_PROMOTION_MAX_LOG_LOSS_RATIO}, rmse<=${PREDICTIVE_PROMOTION_MAX_RMSE_RATIO}, richness<=${PREDICTIVE_PROMOTION_MAX_RICHNESS_RATIO}"
echo "Simulated predictive summary: ${SIMULATED_PREDICTIVE_SUMMARY}"
echo "Whittaker parity metrics: ${WHITTAKER_REFERENCE_PARITY_METRICS}"
echo "Big Spatial parity metrics: ${BIG_SPATIAL_REFERENCE_PARITY_METRICS}"
"${PYTHON}" -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"

SECONDS=0
for SEED in ${SEEDS}; do
  SEED_ROOT="${RUN_ROOT}/seed_${SEED}"
  WHITTAKER_ROOT="${SEED_ROOT}/whittaker"
  BIG_ROOT="${SEED_ROOT}/big_spatial"
  PROMOTION_ROOT="${SEED_ROOT}/promotion_gate"
  mkdir -p "${SEED_ROOT}" "${WHITTAKER_ROOT}" "${BIG_ROOT}" "${PROMOTION_ROOT}"

  echo "Running Whittaker seed ${SEED}"
  "${PYTHON}" examples/run_neural_hmsc_whittaker.py \
    --output "${WHITTAKER_ROOT}" \
    --train-datasets "${TRAIN_DATASETS}" \
    --calibration-datasets "${CALIBRATION_DATASETS}" \
    --sbc-datasets "${SBC_DATASETS}" \
    --sbc-draws "${SBC_DRAWS}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --neural-chains "${NEURAL_CHAINS}" \
    --neural-draws "${NEURAL_DRAWS}" \
    --mcmc-chains "${MCMC_CHAINS}" \
    --mcmc-samples "${MCMC_SAMPLES}" \
    --mcmc-transient "${MCMC_TRANSIENT}" \
    --mcmc-thin "${MCMC_THIN}" \
    --mcmc-verbose "${MCMC_VERBOSE}" \
    --coefficient-calibration "${COEFFICIENT_CALIBRATION}" \
    --conditional-calibration-epochs "${CONDITIONAL_CALIBRATION_EPOCHS}" \
    --conditional-calibration-learning-rate "${CONDITIONAL_CALIBRATION_LEARNING_RATE}" \
    --conditional-calibration-regularization "${CONDITIONAL_CALIBRATION_REGULARIZATION}" \
    --conditional-calibration-rank-penalty-weight "${CONDITIONAL_CALIBRATION_RANK_PENALTY_WEIGHT}" \
    --external-monotone-datasets "${EXTERNAL_MONOTONE_DATASETS}" \
    --external-monotone-max-multiplier "${EXTERNAL_MONOTONE_MAX_MULTIPLIER}" \
    --external-monotone-min-ood-gain "${EXTERNAL_MONOTONE_MIN_OOD_GAIN}" \
    --external-monotone-min-combined-gain "${EXTERNAL_MONOTONE_MIN_COMBINED_GAIN}" \
    --predictive-mean-calibration "${PREDICTIVE_MEAN_CALIBRATION}" \
    --predictive-mean-calibration-validation-datasets "${PREDICTIVE_MEAN_CALIBRATION_VALIDATION_DATASETS}" \
    --predictive-mean-calibration-max-brier-ratio "${PREDICTIVE_MEAN_CALIBRATION_MAX_BRIER_RATIO}" \
    --predictive-mean-calibration-max-log-loss-ratio "${PREDICTIVE_MEAN_CALIBRATION_MAX_LOG_LOSS_RATIO}" \
    --predictive-mean-calibration-min-improvement "${PREDICTIVE_MEAN_CALIBRATION_MIN_IMPROVEMENT}" \
    --predictive-mean-source-min-improvement "${PREDICTIVE_MEAN_SOURCE_MIN_IMPROVEMENT}" \
    --predictive-mean-transfer-branch-min-improvement "${PREDICTIVE_MEAN_TRANSFER_BRANCH_MIN_IMPROVEMENT}" \
    --predictive-mean-selection-policy "${PREDICTIVE_MEAN_SELECTION_POLICY}" \
    --predictive-mean-transfer-min-brier-gain "${PREDICTIVE_MEAN_TRANSFER_MIN_BRIER_GAIN}" \
    --predictive-mean-transfer-min-log-loss-gain "${PREDICTIVE_MEAN_TRANSFER_MIN_LOG_LOSS_GAIN}" \
    --predictive-mean-transfer-max-slope-delta "${PREDICTIVE_MEAN_TRANSFER_MAX_SLOPE_DELTA}" \
    --predictive-mean-transfer-max-abs-intercept "${PREDICTIVE_MEAN_TRANSFER_MAX_ABS_INTERCEPT}" \
    --reference-parity-metrics "${WHITTAKER_REFERENCE_PARITY_METRICS}" \
    --qualified-reference-label "${QUALIFIED_REFERENCE_LABEL}" \
    --seed "${SEED}"

  echo "Running Big Spatial transfer seed ${SEED}"
  "${PYTHON}" examples/run_neural_hmsc_big_spatial_transfer.py \
    --frozen-run "${WHITTAKER_ROOT}" \
    --output "${BIG_ROOT}" \
    --neural-chains "${NEURAL_CHAINS}" \
    --neural-draws "${NEURAL_DRAWS}" \
    --mcmc-chains "${MCMC_CHAINS}" \
    --mcmc-samples "${MCMC_SAMPLES}" \
    --mcmc-transient "${MCMC_TRANSIENT}" \
    --mcmc-thin "${MCMC_THIN}" \
    --mcmc-verbose "${MCMC_VERBOSE}" \
    --target-context-gate "${TARGET_CONTEXT_GATE}" \
    --target-context-gate-datasets "${TARGET_CONTEXT_GATE_DATASETS}" \
    --target-context-gate-max-brier-ratio "${TARGET_CONTEXT_GATE_MAX_BRIER_RATIO}" \
    --target-context-gate-max-log-loss-ratio "${TARGET_CONTEXT_GATE_MAX_LOG_LOSS_RATIO}" \
    --target-context-gate-min-improvement "${TARGET_CONTEXT_GATE_MIN_IMPROVEMENT}" \
    --reference-parity-metrics "${BIG_SPATIAL_REFERENCE_PARITY_METRICS}" \
    --qualified-reference-label "${QUALIFIED_REFERENCE_LABEL}" \
    --seed "${SEED}"

  echo "Running cross-dataset predictive promotion gate seed ${SEED}"
  set +e
  "${PYTHON}" examples/evaluate_neural_hmsc_predictive_promotion.py \
    --dataset "whittaker=${WHITTAKER_ROOT}/whittaker_heldout_metrics.csv" \
    --dataset "big_spatial=${BIG_ROOT}/big_spatial_transfer_heldout_metrics.csv" \
    --max-brier-ratio "${PREDICTIVE_PROMOTION_MAX_BRIER_RATIO}" \
    --max-log-loss-ratio "${PREDICTIVE_PROMOTION_MAX_LOG_LOSS_RATIO}" \
    --max-predictive-rmse-ratio "${PREDICTIVE_PROMOTION_MAX_RMSE_RATIO}" \
    --max-richness-mae-ratio "${PREDICTIVE_PROMOTION_MAX_RICHNESS_RATIO}" \
    --min-mean-brier-gain "${PREDICTIVE_PROMOTION_MIN_MEAN_BRIER_GAIN}" \
    --min-mean-log-loss-gain "${PREDICTIVE_PROMOTION_MIN_MEAN_LOG_LOSS_GAIN}" \
    --simulated-summary "${SIMULATED_PREDICTIVE_SUMMARY}" \
    --min-simulated-brier-gain 0.000001 \
    --min-simulated-log-loss-gain 0.000001 \
    --output "${PROMOTION_ROOT}"
  PROMOTION_EXIT_CODE=$?
  set -e
  printf 'promotion_gate_exit_code=%s\n' "${PROMOTION_EXIT_CODE}" \
    | tee "${PROMOTION_ROOT}/promotion_gate_exit_code.txt"
done

"${PYTHON}" examples/aggregate_neural_hmsc_realdata_sensitivity.py \
  --run-root "${RUN_ROOT}" \
  --seeds ${SEEDS} \
  --output-prefix "${RUN_ROOT}/realdata_sensitivity" \
  --strict

printf 'wall_time_seconds=%s\n' "${SECONDS}" | tee "${RUN_ROOT}/wall_time.txt"
echo "Neural-HMSC real-data sensitivity complete: ${RUN_ROOT}"
