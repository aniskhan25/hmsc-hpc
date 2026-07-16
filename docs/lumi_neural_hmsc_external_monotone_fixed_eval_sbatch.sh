#!/bin/bash -l
#SBATCH --job-name=nhmsc-extmono
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
RUN_NAME="${RUN_NAME:-neural_hmsc_external_monotone_fixed_eval_${SLURM_JOB_ID:-manual}}"
RUN_ROOT="${RUN_ROOT:-${USER_WORK}/hmsc-hpc-runs/${RUN_NAME}}"
SEEDS="${SEEDS:-20260716 20260717 20260718 20260719 20260720}"
N_SITES="${N_SITES:-32}"
N_SPECIES="${N_SPECIES:-45}"
TRAIN_DATASETS="${TRAIN_DATASETS:-8}"
CALIBRATION_DATASETS="${CALIBRATION_DATASETS:-8}"
RARE_CALIBRATION_DATASETS="${RARE_CALIBRATION_DATASETS:-8}"
RARE_VALIDATION_DATASETS="${RARE_VALIDATION_DATASETS:-8}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-4}"
CONDITIONAL_CALIBRATION_EPOCHS="${CONDITIONAL_CALIBRATION_EPOCHS:-30}"
EXTERNAL_MONOTONE_DATASETS="${EXTERNAL_MONOTONE_DATASETS:-4}"
EXTERNAL_MONOTONE_MAX_MULTIPLIER="${EXTERNAL_MONOTONE_MAX_MULTIPLIER:-2}"
NEURAL_CHAINS="${NEURAL_CHAINS:-1}"
NEURAL_DRAWS="${NEURAL_DRAWS:-16}"
SBC_DATASETS="${SBC_DATASETS:-8}"
SBC_DRAWS="${SBC_DRAWS:-64}"
SBC_BINS="${SBC_BINS:-8}"
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
echo "Run root: ${RUN_ROOT}"
echo "Seeds: ${SEEDS}"
echo "Shape: sites=${N_SITES}, species=${N_SPECIES}"
echo "SBC datasets/draws/bins: ${SBC_DATASETS}/${SBC_DRAWS}/${SBC_BINS}"
echo "OOD regimes: ${OOD_REGIMES}"
echo "External monotone datasets/max multiplier: ${EXTERNAL_MONOTONE_DATASETS}/${EXTERNAL_MONOTONE_MAX_MULTIPLIER}"
"${PYTHON}" -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"
"${PYTHON}" -c "import hmsc, pyhmsc; print('hmsc-hpc import: ok')"

SECONDS=0
for seed in ${SEEDS}; do
  seed_root="${RUN_ROOT}/seed_${seed}"
  scalar_dir="${seed_root}/scalar"
  default_dir="${seed_root}/default"
  external_dir="${seed_root}/external_monotone"
  comparison_dir="${seed_root}/comparison"
  checkpoint="${scalar_dir}/probit/neural_checkpoint"

  echo "=== seed ${seed}: scalar ==="
  "${PYTHON}" examples/run_neural_hmsc_benchmark.py \
    --output "${scalar_dir}" \
    --suite probit \
    --n-sites "${N_SITES}" \
    --n-species "${N_SPECIES}" \
    --train-datasets "${TRAIN_DATASETS}" \
    --calibration-datasets "${CALIBRATION_DATASETS}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --coefficient-calibration scalar \
    --neural-chains "${NEURAL_CHAINS}" \
    --neural-draws "${NEURAL_DRAWS}" \
    --sbc-datasets "${SBC_DATASETS}" \
    --sbc-draws "${SBC_DRAWS}" \
    --sbc-bins "${SBC_BINS}" \
    --ood-regimes ${OOD_REGIMES} \
    --seed "${seed}" \
    --model-seed "${seed}"

  echo "=== seed ${seed}: default conditional ==="
  "${PYTHON}" examples/run_neural_hmsc_benchmark.py \
    --output "${default_dir}" \
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
    --coefficient-calibration conditional \
    --conditional-calibration-epochs "${CONDITIONAL_CALIBRATION_EPOCHS}" \
    --neural-chains "${NEURAL_CHAINS}" \
    --neural-draws "${NEURAL_DRAWS}" \
    --sbc-datasets "${SBC_DATASETS}" \
    --sbc-draws "${SBC_DRAWS}" \
    --sbc-bins "${SBC_BINS}" \
    --ood-regimes ${OOD_REGIMES} \
    --seed "${seed}" \
    --model-seed "${seed}"

  echo "=== seed ${seed}: external_monotone ==="
  "${PYTHON}" examples/run_neural_hmsc_benchmark.py \
    --output "${external_dir}" \
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
    --run "scalar=${scalar_dir}" \
    --run "default=${default_dir}" \
    --run "external_monotone=${external_dir}" \
    --baseline scalar \
    --output "${comparison_dir}"
done

"${PYTHON}" - "${RUN_ROOT}" <<'PY'
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for summary in sorted(root.glob("seed_*/comparison/fixed_evaluation_summary.csv")):
    seed = summary.parts[-3].removeprefix("seed_")
    with summary.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append({"seed": seed, **row})

metrics = [
    "in_domain_coverage_95",
    "rare_prevalence_coverage_95",
    "mean_ood_coverage_95",
    "worst_ood_coverage_95",
    "effect_size_shift_coverage_95",
    "combined_shift_coverage_95",
    "mean_ood_coverage_95_delta_vs_baseline",
    "combined_shift_coverage_95_delta_vs_baseline",
]
by_run = defaultdict(list)
for row in rows:
    by_run[row["run"]].append(row)

aggregate = {"rows": rows, "summary": []}
for run, run_rows in sorted(by_run.items()):
    item = {"run": run, "n_seeds": len(run_rows)}
    for metric in metrics:
        values = [float(row[metric]) for row in run_rows]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        item[f"{metric}_mean"] = mean
        item[f"{metric}_sd"] = variance ** 0.5
    item["accepted"] = [row["fixed_evaluation_acceptance_passed"] for row in run_rows]
    aggregate["summary"].append(item)

with (root / "five_seed_fixed_evaluation_aggregate.json").open("w", encoding="utf-8") as handle:
    json.dump(aggregate, handle, indent=2)
with (root / "five_seed_fixed_evaluation_aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
    fieldnames = sorted({key for item in aggregate["summary"] for key in item})
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(aggregate["summary"])
print(json.dumps(aggregate["summary"], indent=2))
PY

printf 'wall_time_seconds=%s\n' "${SECONDS}" | tee "${RUN_ROOT}/wall_time.txt"
echo "Five-seed external monotone fixed-evaluation workflow complete."
echo "Outputs are in ${RUN_ROOT}"
