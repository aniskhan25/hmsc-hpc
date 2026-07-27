#!/bin/bash -l
#SBATCH --job-name=neural-m54-v21-freeze
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
RUN_NAME="${RUN_NAME:-neural_hmsc_m54_v2_1_train_calibration_${SLURM_JOB_ID:-manual}}"
RUN_ROOT="${RUN_ROOT:-${USER_WORK}/hmsc-hpc-runs/${RUN_NAME}}"
REGISTRY_ROOT="${REGISTRY_ROOT:-${USER_WORK}/hmsc-hpc-deployments}"
VARIABLE_BASELINE="${VARIABLE_BASELINE:-${REGISTRY_ROOT}/neural_hmsc_variable_probit_v1}"
CONFIRMATION="${CONFIRMATION:-}"
EXPECTED_CONFIRMATION="GENERATE_M54_V2_1_TRAIN_AUX_CALIBRATION"

if [[ "${CONFIRMATION}" != "${EXPECTED_CONFIRMATION}" ]]; then
  echo "Refusing to open the Milestone 54 v2.1 production train blocks." >&2
  echo "Set CONFIRMATION=${EXPECTED_CONFIRMATION}." >&2
  exit 2
fi

mkdir -p output
if [[ -e "${RUN_ROOT}" ]]; then
  echo "Refusing to reuse an existing Milestone 54 v2.1 run root: ${RUN_ROOT}" >&2
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
echo "Protocol action: v2.1 train/auxiliary/calibration freeze only"
echo "Authorized blocks: 111M coefficient, 112M context, 113M heldout, 114M calibration"
echo "Reserved evaluation block: 115000001-115000243 (sealed)"
"${PYTHON}" -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"

SECONDS=0
"${PYTHON}" examples/qualify_neural_hmsc_variable_design_v2_1.py train-calibrate \
  --output "${RUN_ROOT}" \
  --confirmation "${CONFIRMATION}" \
  --fixed-registry "${REGISTRY_ROOT}" \
  --variable-baseline "${VARIABLE_BASELINE}" \
  > "${RUN_ROOT}.stdout.json"

"${PYTHON}" - "${REPO_DIR}" "${RUN_ROOT}" <<'PY'
import hashlib
import importlib.util
import json
import pathlib
import sys

repo = pathlib.Path(sys.argv[1]).resolve()
root = pathlib.Path(sys.argv[2]).resolve()
spec = importlib.util.spec_from_file_location(
    "m54_v2_1_qualification",
    repo / "examples/qualify_neural_hmsc_variable_design_v2_1.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
freeze = module.validate_freeze(root)
expected = module._seed_blocks(module.PRODUCTION_STARTS, module.PRODUCTION_COUNT)
reserved = set(expected["evaluation"])
opened_roles = (
    "coefficient_train",
    "predictive_context",
    "predictive_heldout",
    "calibration",
)
opened = [seed for role in opened_roles for seed in freeze["seeds"][role]]

def balanced(record):
    cells = record["factorial_cell_counts"]
    marginals = record["marginal_counts"]
    return (
        len(cells) == 81
        and set(cells.values()) == {3}
        and all(
            max(counts.values()) - min(counts.values()) <= 1
            for counts in marginals.values()
        )
    )

checks = {
    "protocol_id": freeze["protocol_id"] == module.PROTOCOL_ID,
    "frozen_before_reserved_evaluation": (
        freeze["status"] == "frozen_before_reserved_evaluation"
    ),
    "production_train_blocks_opened": freeze["production_seed_opened"] is True,
    "reserved_evaluation_sealed": freeze["reserved_evaluation_opened"] is False,
    "coefficient_train_111m": freeze["seeds"]["coefficient_train"] == expected["coefficient_train"],
    "predictive_context_112m": freeze["seeds"]["predictive_context"] == expected["predictive_context"],
    "predictive_heldout_113m": freeze["seeds"]["predictive_heldout"] == expected["predictive_heldout"],
    "calibration_114m": freeze["seeds"]["calibration"] == expected["calibration"],
    "reserved_evaluation_115m": (
        freeze["seeds"]["reserved_evaluation_start"] == 115000001
        and freeze["seeds"]["reserved_evaluation_count"] == 243
    ),
    "model_seed": freeze["seeds"]["model"] == 111900001,
    "opened_roles_disjoint": len(opened) == len(set(opened)),
    "opened_roles_exclude_115m": not (set(opened) & reserved),
    "predictive_heldouts_independent": freeze["predictive_heldout_independence"] is True,
    "coefficient_train_balanced": balanced(freeze["corpus_balance"]["coefficient_train"]),
    "predictive_context_balanced": balanced(freeze["corpus_balance"]["predictive_context"]),
    "calibration_balanced": balanced(freeze["corpus_balance"]["calibration"]),
    "immutable_baseline_hashes": freeze["baseline_hashes"]["all_valid"] is True,
    "preregistration_hash": freeze["preregistration_sha256"] == module.PREREGISTRATION_SHA256,
}
if not all(checks.values()):
    raise SystemExit(f"Milestone 54 v2.1 post-freeze validation failed: {checks}")

freeze_path = root / "m54_v2_1_freeze.json"
report = {
    "schema_version": 1,
    "kind": "neural_hmsc_variable_design_m54_v2_1_postfreeze_validation",
    "validated": True,
    "checks": checks,
    "freeze_sha256": hashlib.sha256(freeze_path.read_bytes()).hexdigest(),
    "checkpoint": freeze["checkpoint"],
    "baseline_hashes": freeze["baseline_hashes"],
    "preregistration_sha256": freeze["preregistration_sha256"],
    "seed_roles": freeze["seeds"],
}
(root / "m54_v2_1_postfreeze_validation.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2, sort_keys=True))
PY

sha256sum "${RUN_ROOT}/m54_v2_1_freeze.json" \
  > "${RUN_ROOT}/m54_v2_1_freeze.sha256"
printf 'wall_time_seconds=%s\n' "${SECONDS}" | tee "${RUN_ROOT}/wall_time.txt"
echo "Milestone 54 v2.1 production freeze validated: ${RUN_ROOT}"
echo "Reserved 115M evaluation remains sealed."
