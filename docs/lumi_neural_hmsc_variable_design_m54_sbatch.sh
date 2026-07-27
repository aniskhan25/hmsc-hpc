#!/bin/bash -l
#SBATCH --job-name=neural-m54-candidate
#SBATCH --account=project_462000131
#SBATCH --partition=dev-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=output/%x-%j.out
#SBATCH --error=output/%x-%j.err

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-project_462000131}"
USER_WORK="${USER_WORK:-/scratch/${PROJECT_ID}/anisrahm}"
VENV="${VENV:-${USER_WORK}/venvs/hmsc_tf_env}"
PYTHON="${PYTHON:-${VENV}/bin/python3}"
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
RUN_NAME="${RUN_NAME:-neural_hmsc_m54_candidate_train_calibration_${SLURM_JOB_ID:-manual}}"
RUN_ROOT="${RUN_ROOT:-${USER_WORK}/hmsc-hpc-runs/${RUN_NAME}}"
REGISTRY_ROOT="${REGISTRY_ROOT:-${USER_WORK}/hmsc-hpc-deployments}"
VARIABLE_BASELINE="${VARIABLE_BASELINE:-${REGISTRY_ROOT}/neural_hmsc_variable_probit_v1}"
CONFIRMATION="${CONFIRMATION:-}"
EXPECTED_CONFIRMATION="GENERATE_M54_CANDIDATE_TRAIN_CALIBRATION"

if [[ "${CONFIRMATION}" != "${EXPECTED_CONFIRMATION}" ]]; then
  echo "Refusing to open the Milestone 54 candidate train/calibration blocks." >&2
  echo "Set CONFIRMATION=${EXPECTED_CONFIRMATION}." >&2
  exit 2
fi

mkdir -p output
if [[ -e "${RUN_ROOT}" ]]; then
  echo "Refusing to reuse an existing Milestone 54 run root: ${RUN_ROOT}" >&2
  exit 2
fi

module use /appl/local/csc/modulefiles
module load tensorflow/2.16
source "${VENV}/bin/activate"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export SINGULARITYENV_PYTHONPATH="${PYTHONPATH}"
export APPTAINERENV_PYTHONPATH="${PYTHONPATH}"

echo "Repository: ${REPO_DIR}"
echo "Run root: ${RUN_ROOT}"
echo "Fixed release registry: ${REGISTRY_ROOT}"
echo "Variable-shape baseline: ${VARIABLE_BASELINE}"
echo "Protocol action: candidate train-calibrate only"
echo "Reserved candidate evaluation block: 103000001-103000243 (sealed)"
"${PYTHON}" -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"

SECONDS=0
"${PYTHON}" examples/qualify_neural_hmsc_variable_design.py train-calibrate \
  --role candidate \
  --output "${RUN_ROOT}" \
  --confirmation "${CONFIRMATION}" \
  --fixed-registry "${REGISTRY_ROOT}" \
  --variable-baseline "${VARIABLE_BASELINE}" \
  > "${RUN_ROOT}.stdout.json"

"${PYTHON}" -c '
import hashlib
import importlib.util
import json
import pathlib
import sys

repo = pathlib.Path(sys.argv[1]).resolve()
root = pathlib.Path(sys.argv[2]).resolve()
spec = importlib.util.spec_from_file_location(
    "m54_qualification", repo / "examples/qualify_neural_hmsc_variable_design.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
freeze = module.validate_freeze(root, role="candidate")
train = freeze["seeds"]["train"]
calibration = freeze["seeds"]["calibration"]
checks = {
    "protocol_id": freeze["protocol_id"] == module.PROTOCOL_ID,
    "candidate_role": freeze["role"] == "candidate",
    "frozen_before_reserved_evaluation": (
        freeze["status"] == "frozen_before_reserved_evaluation"
    ),
    "reserved_evaluation_sealed": freeze["reserved_evaluation_opened"] is False,
    "train_block_101m": train == list(range(101000001, 101000244)),
    "calibration_block_102m": calibration == list(range(102000001, 102000244)),
    "reserved_evaluation_block_103m": (
        freeze["seeds"]["reserved_evaluation_start"] == 103000001
        and freeze["seeds"]["reserved_evaluation_count"] == 243
    ),
    "seed_roles_disjoint": not (
        set(train) & set(calibration)
        or set(train) & set(range(103000001, 103000244))
        or set(calibration) & set(range(103000001, 103000244))
    ),
    "immutable_baseline_hashes": freeze["baseline_hashes"]["all_valid"] is True,
}
if not all(checks.values()):
    raise SystemExit(f"Milestone 54 post-freeze validation failed: {checks}")
report = {
    "schema_version": 1,
    "kind": "neural_hmsc_variable_design_m54_postfreeze_validation",
    "validated": True,
    "checks": checks,
    "freeze_sha256": hashlib.sha256(
        (root / "m54_train_calibration_freeze.json").read_bytes()
    ).hexdigest(),
    "checkpoint": freeze["checkpoint"],
    "baseline_hashes": freeze["baseline_hashes"],
    "seed_roles": freeze["seeds"],
}
(root / "m54_postfreeze_validation.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2, sort_keys=True))
' "${REPO_DIR}" "${RUN_ROOT}"

printf 'wall_time_seconds=%s\n' "${SECONDS}" | tee "${RUN_ROOT}/wall_time.txt"
echo "Milestone 54 candidate train/calibration freeze validated: ${RUN_ROOT}"
echo "Reserved 103M evaluation remains sealed."
