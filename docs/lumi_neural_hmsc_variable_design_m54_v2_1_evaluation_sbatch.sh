#!/bin/bash -l
#SBATCH --job-name=neural-m54-v21-eval
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
FREEZE_ROOT="${FREEZE_ROOT:-${USER_WORK}/hmsc-hpc-runs/neural_hmsc_m54_v2_1_train_calibration_20144482}"
RUN_NAME="${RUN_NAME:-neural_hmsc_m54_v2_1_evaluation_${SLURM_JOB_ID:-manual}}"
RUN_ROOT="${RUN_ROOT:-${USER_WORK}/hmsc-hpc-runs/${RUN_NAME}}"
REGISTRY_ROOT="${REGISTRY_ROOT:-${USER_WORK}/hmsc-hpc-deployments}"
VARIABLE_BASELINE="${VARIABLE_BASELINE:-${REGISTRY_ROOT}/neural_hmsc_variable_probit_v1}"
CONFIRMATION="${CONFIRMATION:-}"
EXPECTED_CONFIRMATION="OPEN_M54_V2_1_RESERVED_EVALUATION"
EXPECTED_FREEZE_SHA256="bb32afd655db277064c5c6fcbdf53e2d89a9f42c24a0690c50a494967f46d816"

if [[ "${CONFIRMATION}" != "${EXPECTED_CONFIRMATION}" ]]; then
  echo "Refusing to open the Milestone 54 v2.1 reserved evaluation." >&2
  echo "Set CONFIRMATION=${EXPECTED_CONFIRMATION}." >&2
  exit 2
fi

if [[ ! -f "${FREEZE_ROOT}/m54_v2_1_freeze.json" ]]; then
  echo "Milestone 54 v2.1 freeze is missing: ${FREEZE_ROOT}" >&2
  exit 2
fi
OBSERVED_FREEZE_SHA256="$(sha256sum "${FREEZE_ROOT}/m54_v2_1_freeze.json" | awk '{print $1}')"
if [[ "${OBSERVED_FREEZE_SHA256}" != "${EXPECTED_FREEZE_SHA256}" ]]; then
  echo "Milestone 54 v2.1 freeze hash differs: ${OBSERVED_FREEZE_SHA256}" >&2
  exit 2
fi
if [[ -e "${RUN_ROOT}" ]]; then
  echo "Refusing to reuse an existing Milestone 54 v2.1 evaluation root: ${RUN_ROOT}" >&2
  exit 2
fi

mkdir -p output
module use /appl/local/csc/modulefiles
module load tensorflow/2.16
source "${VENV}/bin/activate"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export SINGULARITYENV_PYTHONPATH="${PYTHONPATH}"
export APPTAINERENV_PYTHONPATH="${PYTHONPATH}"

echo "Repository: ${REPO_DIR}"
echo "Frozen candidate: ${FREEZE_ROOT}"
echo "Frozen candidate SHA-256: ${OBSERVED_FREEZE_SHA256}"
echo "Evaluation root: ${RUN_ROOT}"
echo "Protocol action: v2.1 one-shot reserved evaluation only"
echo "Authorized evaluation block: 115000001-115000243"
"${PYTHON}" -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"

"${PYTHON}" - "${REPO_DIR}" "${FREEZE_ROOT}" "${EXPECTED_FREEZE_SHA256}" <<'PY'
import hashlib
import importlib.util
import pathlib
import sys

repo = pathlib.Path(sys.argv[1]).resolve()
root = pathlib.Path(sys.argv[2]).resolve()
expected_sha256 = sys.argv[3]
spec = importlib.util.spec_from_file_location(
    "m54_v2_1_qualification",
    repo / "examples/qualify_neural_hmsc_variable_design_v2_1.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
freeze = module.validate_freeze(root)
freeze_path = root / "m54_v2_1_freeze.json"
observed_sha256 = hashlib.sha256(freeze_path.read_bytes()).hexdigest()
checks = {
    "freeze_sha256": observed_sha256 == expected_sha256,
    "protocol_id": freeze["protocol_id"] == module.PROTOCOL_ID,
    "frozen_before_reserved_evaluation": (
        freeze["status"] == "frozen_before_reserved_evaluation"
    ),
    "reserved_evaluation_sealed": freeze["reserved_evaluation_opened"] is False,
    "reserved_evaluation_start": (
        freeze["seeds"]["reserved_evaluation_start"] == 115000001
    ),
    "reserved_evaluation_count": (
        freeze["seeds"]["reserved_evaluation_count"] == 243
    ),
    "checkpoint_hashes": all(
        freeze["checkpoint"].get(name)
        for name in (
            "manifest_sha256",
            "weights_sha256",
            "calibration_sha256",
        )
    ),
    "immutable_baseline_hashes": freeze["baseline_hashes"]["all_valid"] is True,
    "preregistration_hash": (
        freeze["preregistration_sha256"] == module.PREREGISTRATION_SHA256
    ),
}
if not all(checks.values()):
    raise SystemExit(f"Milestone 54 v2.1 pre-evaluation validation failed: {checks}")
print(f"Pre-evaluation validation passed: {checks}")
PY

SECONDS=0
"${PYTHON}" examples/qualify_neural_hmsc_variable_design_v2_1.py evaluate \
  --freeze-root "${FREEZE_ROOT}" \
  --output "${RUN_ROOT}" \
  --confirmation "${CONFIRMATION}" \
  --fixed-registry "${REGISTRY_ROOT}" \
  --variable-baseline "${VARIABLE_BASELINE}" \
  > "${RUN_ROOT}.stdout.json"

"${PYTHON}" - "${RUN_ROOT}" "${EXPECTED_FREEZE_SHA256}" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
expected_freeze_sha256 = sys.argv[2]
report_path = root / "m54_v2_1_evaluation.json"
report = json.loads(report_path.read_text(encoding="utf-8"))
evaluation_seeds = [
    int(row["seed"]) for row in report["evaluation"]["dataset_rows"]
]
expected_evaluation_seeds = list(range(115000001, 115000244))
expected_mcmc_seeds = [
    115000109,
    115000148,
    115000133,
    115000178,
    115000211,
    115000217,
]
mcmc_seeds = [int(row["seed"]) for row in report["mcmc"]["rows"]]
gate_consistency = report["all_gates_passed"] == all(
    bool(value) for value in report["gates"].values()
)
expected_decision = (
    "variable_design_v2_1_simulated_passed_realdata_pending"
    if report["all_gates_passed"]
    else "variable_design_v2_1_terminal_failure"
)
checks = {
    "protocol_id": (
        report["protocol_id"] == "neural_hmsc_variable_design_m54_v2_1"
    ),
    "reserved_evaluation_opened": report["reserved_evaluation_opened"] is True,
    "freeze_sha256": report["freeze_sha256"] == expected_freeze_sha256,
    "evaluation_block_exact": evaluation_seeds == expected_evaluation_seeds,
    "mcmc_seed_subset_exact": mcmc_seeds == expected_mcmc_seeds,
    "gate_consistency": gate_consistency,
    "decision_consistency": report["decision"] == expected_decision,
    "immutable_baseline_hashes": report["baseline_hashes"]["all_valid"] is True,
    "preregistration_hash": (
        report["preregistration_sha256"]
        == "900af8719fc73947cd7addf3b7dc9fe2f233eadbbd2bf9f37bac1286fc15e54d"
    ),
}
if not all(checks.values()):
    raise SystemExit(f"Milestone 54 v2.1 post-evaluation validation failed: {checks}")
validation = {
    "schema_version": 1,
    "kind": "neural_hmsc_variable_design_m54_v2_1_postevaluation_validation",
    "validated": True,
    "checks": checks,
    "decision": report["decision"],
    "all_gates_passed": report["all_gates_passed"],
    "failed_gates": [
        name for name, value in report["gates"].items() if not bool(value)
    ],
    "evaluation_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
    "freeze_sha256": report["freeze_sha256"],
}
(root / "m54_v2_1_postevaluation_validation.json").write_text(
    json.dumps(validation, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(validation, indent=2, sort_keys=True))
PY

sha256sum "${RUN_ROOT}/m54_v2_1_evaluation.json" \
  > "${RUN_ROOT}/m54_v2_1_evaluation.sha256"
printf 'wall_time_seconds=%s\n' "${SECONDS}" | tee "${RUN_ROOT}/wall_time.txt"
echo "Milestone 54 v2.1 reserved evaluation completed: ${RUN_ROOT}"
