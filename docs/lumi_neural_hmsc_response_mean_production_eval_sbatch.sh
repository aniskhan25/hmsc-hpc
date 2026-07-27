#!/bin/bash -l
#SBATCH --job-name=nhmsc-respmean
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
BASELINE_ROOT="${BASELINE_ROOT:-${USER_WORK}/hmsc-hpc-runs/neural_hmsc_external_monotone_production_confirm_20260716}"
RUN_NAME="${RUN_NAME:-neural_hmsc_response_mean_production_eval_${SLURM_JOB_ID:-manual}}"
RUN_ROOT="${RUN_ROOT:-${USER_WORK}/hmsc-hpc-runs/${RUN_NAME}}"
SEEDS="${SEEDS:-20260716 20260717 20260718 20260719 20260720}"
N_SITES="${N_SITES:-40}"
N_SPECIES="${N_SPECIES:-75}"
TRAIN_DATASETS="${TRAIN_DATASETS:-8}"
CALIBRATION_DATASETS="${CALIBRATION_DATASETS:-8}"
RARE_CALIBRATION_DATASETS="${RARE_CALIBRATION_DATASETS:-8}"
RARE_VALIDATION_DATASETS="${RARE_VALIDATION_DATASETS:-8}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-4}"
CONDITIONAL_CALIBRATION_EPOCHS="${CONDITIONAL_CALIBRATION_EPOCHS:-30}"
EXTERNAL_MONOTONE_DATASETS="${EXTERNAL_MONOTONE_DATASETS:-4}"
EXTERNAL_MONOTONE_MAX_MULTIPLIER="${EXTERNAL_MONOTONE_MAX_MULTIPLIER:-2}"
EXTERNAL_MONOTONE_MIN_OOD_GAIN="${EXTERNAL_MONOTONE_MIN_OOD_GAIN:-0.005}"
EXTERNAL_MONOTONE_MIN_COMBINED_GAIN="${EXTERNAL_MONOTONE_MIN_COMBINED_GAIN:-0.005}"
PREDICTIVE_MEAN_VALIDATION_DATASETS="${PREDICTIVE_MEAN_VALIDATION_DATASETS:-8}"
PREDICTIVE_MEAN_MIN_IMPROVEMENT="${PREDICTIVE_MEAN_MIN_IMPROVEMENT:-0.0001}"
PREDICTIVE_MEAN_CALIBRATION="${PREDICTIVE_MEAN_CALIBRATION:-probit_response_affine}"
NEURAL_CHAINS="${NEURAL_CHAINS:-1}"
NEURAL_DRAWS="${NEURAL_DRAWS:-16}"
SBC_DATASETS="${SBC_DATASETS:-32}"
SBC_DRAWS="${SBC_DRAWS:-256}"
SBC_BINS="${SBC_BINS:-10}"
OOD_REGIMES="${OOD_REGIMES:-covariate_shift effect_size_shift combined_shift}"

mkdir -p output "${RUN_ROOT}"

module use /appl/local/csc/modulefiles
module load tensorflow/2.16

if [[ ! -f "${VENV}/bin/activate" ]]; then
  echo "Missing virtual environment: ${VENV}" >&2
  exit 2
fi

source "${VENV}/bin/activate"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export SINGULARITYENV_PYTHONPATH="${PYTHONPATH}"
export APPTAINERENV_PYTHONPATH="${PYTHONPATH}"

echo "Repository: ${REPO_DIR}"
echo "Qualified baseline root: ${BASELINE_ROOT}"
echo "Run root: ${RUN_ROOT}"
echo "Seeds: ${SEEDS}"
echo "Shape: sites=${N_SITES}, species=${N_SPECIES}"
echo "SBC datasets/draws/bins: ${SBC_DATASETS}/${SBC_DRAWS}/${SBC_BINS}"
echo "Predictive mean calibration: ${PREDICTIVE_MEAN_CALIBRATION}"
echo "Response mean validation datasets/min improvement: ${PREDICTIVE_MEAN_VALIDATION_DATASETS}/${PREDICTIVE_MEAN_MIN_IMPROVEMENT}"
"${PYTHON}" -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"
"${PYTHON}" -c "import hmsc, pyhmsc; print('hmsc-hpc import: ok')"

SECONDS=0
for seed in ${SEEDS}; do
  seed_root="${RUN_ROOT}/seed_${seed}"
  baseline_seed_root="${BASELINE_ROOT}/seed_${seed}"
  baseline_external="${baseline_seed_root}/external_monotone"
  checkpoint="${baseline_seed_root}/scalar/probit/neural_checkpoint"
  response_dir="${seed_root}/external_monotone_response"
  comparison_dir="${seed_root}/comparison"
  predictive_dir="${seed_root}/predictive"

  if [[ ! -d "${baseline_external}" ]]; then
    echo "Missing baseline external_monotone directory: ${baseline_external}" >&2
    exit 3
  fi
  if [[ ! -d "${checkpoint}" ]]; then
    echo "Missing baseline scalar checkpoint: ${checkpoint}" >&2
    exit 3
  fi

  echo "=== seed ${seed}: external_monotone + ${PREDICTIVE_MEAN_CALIBRATION} ==="
  "${PYTHON}" examples/run_neural_hmsc_benchmark.py \
    --output "${response_dir}" \
    --suite probit \
    --n-sites "${N_SITES}" \
    --n-species "${N_SPECIES}" \
    --train-datasets "${TRAIN_DATASETS}" \
    --calibration-datasets "${CALIBRATION_DATASETS}" \
    --rare-calibration-datasets "${RARE_CALIBRATION_DATASETS}" \
    --rare-validation-datasets "${RARE_VALIDATION_DATASETS}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --checkpoint "${checkpoint}" \
    --coefficient-calibration external_monotone \
    --conditional-calibration-epochs "${CONDITIONAL_CALIBRATION_EPOCHS}" \
    --external-monotone-datasets "${EXTERNAL_MONOTONE_DATASETS}" \
    --external-monotone-max-multiplier "${EXTERNAL_MONOTONE_MAX_MULTIPLIER}" \
    --external-monotone-min-ood-gain "${EXTERNAL_MONOTONE_MIN_OOD_GAIN}" \
    --external-monotone-min-combined-gain "${EXTERNAL_MONOTONE_MIN_COMBINED_GAIN}" \
    --predictive-mean-calibration "${PREDICTIVE_MEAN_CALIBRATION}" \
    --predictive-mean-calibration-validation-datasets "${PREDICTIVE_MEAN_VALIDATION_DATASETS}" \
    --predictive-mean-calibration-min-improvement "${PREDICTIVE_MEAN_MIN_IMPROVEMENT}" \
    --neural-chains "${NEURAL_CHAINS}" \
    --neural-draws "${NEURAL_DRAWS}" \
    --sbc-datasets "${SBC_DATASETS}" \
    --sbc-draws "${SBC_DRAWS}" \
    --sbc-bins "${SBC_BINS}" \
    --ood-regimes ${OOD_REGIMES} \
    --seed "${seed}" \
    --model-seed "${seed}"

  echo "=== seed ${seed}: fixed evaluation comparison ==="
  "${PYTHON}" examples/compare_neural_hmsc_fixed_evaluation.py \
    --run "external_monotone=${baseline_external}" \
    --run "external_monotone_response=${response_dir}" \
    --baseline external_monotone \
    --output "${comparison_dir}" \
    --min-mean-ood-delta 0.0 \
    --min-combined-delta 0.0

  echo "=== seed ${seed}: predictive score comparison ==="
  "${PYTHON}" examples/compare_neural_hmsc_predictive_scores.py \
    --run "external_monotone=${baseline_external}" \
    --run "external_monotone_response=${response_dir}" \
    --baseline external_monotone \
    --output "${predictive_dir}"
done

"${PYTHON}" - "${RUN_ROOT}" <<'PY'
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])

fixed_rows = []
for summary in sorted(root.glob("seed_*/comparison/fixed_evaluation_summary.csv")):
    seed = summary.parts[-3].removeprefix("seed_")
    with summary.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            fixed_rows.append({"seed": seed, **row})

predictive_rows = []
for summary in sorted(root.glob("seed_*/predictive/predictive_score_summary.csv")):
    seed = summary.parts[-3].removeprefix("seed_")
    with summary.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            predictive_rows.append({"seed": seed, **row})

def summarize(rows, metrics):
    by_run = defaultdict(list)
    for row in rows:
        by_run[row["run"]].append(row)
    out = []
    for run, run_rows in sorted(by_run.items()):
        item = {"run": run, "n_seeds": len(run_rows)}
        if "fixed_evaluation_acceptance_passed" in run_rows[0]:
            item["accepted"] = [
                row["fixed_evaluation_acceptance_passed"] for row in run_rows
            ]
        for metric in metrics:
            values = [float(row[metric]) for row in run_rows]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            item[f"{metric}_mean"] = mean
            item[f"{metric}_sd"] = variance ** 0.5
        out.append(item)
    return out

fixed_metrics = [
    "in_domain_coverage_95",
    "rare_prevalence_coverage_95",
    "mean_ood_coverage_95",
    "worst_ood_coverage_95",
    "effect_size_shift_coverage_95",
    "combined_shift_coverage_95",
]
predictive_metrics = [
    "brier_score",
    "brier_score_ratio_vs_baseline",
    "log_loss",
    "log_loss_ratio_vs_baseline",
    "predictive_rmse",
    "predictive_rmse_ratio_vs_baseline",
    "prevalence_mae",
    "richness_mae",
]

aggregate = {
    "fixed_rows": fixed_rows,
    "predictive_rows": predictive_rows,
    "fixed_summary": summarize(fixed_rows, fixed_metrics),
    "predictive_summary": summarize(predictive_rows, predictive_metrics),
}
with (root / "response_mean_production_aggregate.json").open("w", encoding="utf-8") as handle:
    json.dump(aggregate, handle, indent=2)
for key in ["fixed_summary", "predictive_summary"]:
    with (root / f"response_mean_production_{key}.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({name for row in aggregate[key] for name in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(aggregate[key])
print(json.dumps({k: aggregate[k] for k in ["fixed_summary", "predictive_summary"]}, indent=2))
PY

printf 'wall_time_seconds=%s\n' "${SECONDS}" | tee "${RUN_ROOT}/wall_time.txt"
echo "Response-mean production-shape evaluation complete."
echo "Outputs are in ${RUN_ROOT}"
