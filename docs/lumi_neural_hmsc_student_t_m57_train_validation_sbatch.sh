#!/bin/bash -l
#SBATCH --job-name=neural-m57-train-val
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
RUN_NAME="${RUN_NAME:-neural_hmsc_m57_train_validation_${SLURM_JOB_ID:-manual}}"
RUN_ROOT="${RUN_ROOT:-${USER_WORK}/hmsc-hpc-runs/${RUN_NAME}}"
REGISTRY_ROOT="${REGISTRY_ROOT:-${USER_WORK}/hmsc-hpc-deployments}"
VARIABLE_REGISTRY_ROOT="${VARIABLE_REGISTRY_ROOT:-${REGISTRY_ROOT}}"
M56_ROOT="${M56_ROOT:-${USER_WORK}/hmsc-hpc-runs/neural_hmsc_m56_train_validation_20192218}"
CONFIRMATION="${CONFIRMATION:-}"
EXPECTED_CONFIRMATION="GENERATE_M57_STUDENT_T_TRAIN_VALIDATION"
EXPECTED_DECISION_SHA256="a1a7bc4a54eca4c78f6b32537f1afff662a524557accbd99d7267a28bc2cb2ba"
EXPECTED_AUDIT_SHA256="1e1150a04cd17643db37988bfc010b611f8f49d638dbd40ead49cd5329b9b25c"
EXPECTED_PREREGISTRATION_SHA256="10878c65bb16746a4a9c57fa91d6a4fd3cbcc753739a816f6cc8b9b738f1a388"

if [[ "${CONFIRMATION}" != "${EXPECTED_CONFIRMATION}" ]]; then
  echo "Refusing to open the Milestone 57 production train-validation blocks." >&2
  echo "Set CONFIRMATION=${EXPECTED_CONFIRMATION}." >&2
  exit 2
fi

if [[ -e "${RUN_ROOT}" ]]; then
  echo "Refusing to reuse an existing Milestone 57 run root: ${RUN_ROOT}" >&2
  exit 2
fi
if [[ ! -f "${REGISTRY_ROOT}/neural_hmsc_v0_1/release.json" ]]; then
  echo "Immutable neural_hmsc_v0_1 release is missing." >&2
  exit 2
fi
if [[ ! -f \
  "${VARIABLE_REGISTRY_ROOT}/neural_hmsc_variable_probit_v1/baseline.json" ]]; then
  echo "Immutable variable-v1 regression baseline is missing." >&2
  exit 2
fi
if [[ ! -f "${M56_ROOT}/freeze.json" ]]; then
  echo "Frozen failed-M56 negative reference is missing." >&2
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

OBSERVED_DECISION_SHA256="$(
  sha256sum docs/neural_hmsc_post_m56_capability_decision_2026-07-24.md |
    awk '{print $1}'
)"
OBSERVED_AUDIT_SHA256="$(
  sha256sum docs/neural_hmsc_m57_artifact_seed_audit_2026-07-24.json.md |
    awk '{print $1}'
)"
OBSERVED_PREREGISTRATION_SHA256="$(
  sha256sum docs/neural_hmsc_m57_student_t_preregistration_2026-07-24.md |
    awk '{print $1}'
)"
if [[ "${OBSERVED_DECISION_SHA256}" != "${EXPECTED_DECISION_SHA256}" ]]; then
  echo "Milestone 57 capability-decision hash differs." >&2
  exit 2
fi
if [[ "${OBSERVED_AUDIT_SHA256}" != "${EXPECTED_AUDIT_SHA256}" ]]; then
  echo "Milestone 57 artifact/seed audit hash differs." >&2
  exit 2
fi
if [[ "${OBSERVED_PREREGISTRATION_SHA256}" != "${EXPECTED_PREREGISTRATION_SHA256}" ]]; then
  echo "Milestone 57 preregistration hash differs." >&2
  exit 2
fi

echo "Repository: ${REPO_DIR}"
echo "Run root: ${RUN_ROOT}"
echo "Release registry: ${REGISTRY_ROOT}"
echo "Variable-v1 registry: ${VARIABLE_REGISTRY_ROOT}"
echo "Failed-M56 reference: ${M56_ROOT}"
echo "Protocol action: one-shot Student-t training and fixed validation"
echo "Authorized training block: 321000001-321000324"
echo "Authorized validation block: 322000001-322000324"
echo "Reserved blocks remain sealed: 323000001-325000324"
"${PYTHON}" -c \
  "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"

SECONDS=0
"${PYTHON}" examples/qualify_neural_hmsc_student_t.py \
  --release-registry "${REGISTRY_ROOT}" \
  --variable-registry "${VARIABLE_REGISTRY_ROOT}" \
  --m56-root "${M56_ROOT}" \
  train-validate \
  --output "${RUN_ROOT}" \
  --confirmation "${CONFIRMATION}" \
  > "${RUN_ROOT}.stdout.json"

"${PYTHON}" - \
  "${RUN_ROOT}" \
  "${EXPECTED_DECISION_SHA256}" \
  "${EXPECTED_AUDIT_SHA256}" \
  "${EXPECTED_PREREGISTRATION_SHA256}" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
decision_sha256, audit_sha256, preregistration_sha256 = sys.argv[2:]
freeze_path = root / "freeze.json"
freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
gates = freeze["validation_gates"]
checks = {
    "protocol_id": (
        freeze["protocol_id"] == "neural_hmsc_fixed_probit_student_t_m57_v1"
    ),
    "mode": freeze["mode"] == "production_train_validation",
    "training_seed_range": freeze["training_seed_range"]
    == [321000001, 321000324],
    "validation_seed_range": freeze["validation_seed_range"]
    == [322000001, 322000324],
    "reserved_seed_blocks_opened": freeze["reserved_seed_blocks_opened"] is False,
    "model_seed": freeze["training"]["seed"] == 321900001,
    "owning_context_count": freeze["training"]["owning_context_count"] == 324,
    "training_realization_count": freeze["training"]["realization_count"] == 648,
    "responses_per_context": freeze["training"]["responses_per_context"] == 2,
    "training_epochs": freeze["training"]["epochs"] == 150,
    "training_batch_contexts": (
        freeze["training"]["batch_owning_contexts"] == 9
    ),
    "learning_rate": freeze["training"]["learning_rate"] == 0.0005,
    "decision_hash": (
        freeze["protocol_hashes"]["decision_sha256"] == decision_sha256
    ),
    "audit_hash": freeze["protocol_hashes"]["audit_sha256"] == audit_sha256,
    "preregistration_hash": (
        freeze["protocol_hashes"]["preregistration_sha256"]
        == preregistration_sha256
    ),
    "bound_release": (
        freeze["bindings"]["v0_1"]["release_content_sha256"]
        == "affcfe10d2f9586e97432ae07754ab829e929ffdb62f41fb71779ad5f3ed12c8"
    ),
    "bound_variable_v1": (
        freeze["bindings"]["variable_v1"]["content_sha256"]
        == "badaf8b8244cbd850693723147c6094bf196482237c52d2cc279ce42b286d2f9"
    ),
    "bound_failed_m56": (
        freeze["bindings"]["m56_negative"]["freeze_sha256"]
        == "c4fcb04cf1ebd7123be12144803de319ce1ff16a31e4fc5a1fb3e224f361a526"
    ),
    "gate_decision_consistent": freeze["validation_passed"]
    == all(bool(value) for value in gates.values()),
    "overlay_manifest_present": (
        root / "overlay/student_t_overlay.json"
    ).is_file(),
    "overlay_weights_present": (
        root / "overlay/student_t_head.weights.h5"
    ).is_file(),
}
if not all(checks.values()):
    raise SystemExit(f"Milestone 57 post-freeze validation failed: {checks}")

validation = {
    "schema_version": 1,
    "kind": "neural_hmsc_m57_train_validation_postfreeze_validation",
    "validated": True,
    "checks": checks,
    "validation_passed": freeze["validation_passed"],
    "failed_gates": [
        name for name, value in gates.items() if not bool(value)
    ],
    "freeze_sha256": hashlib.sha256(freeze_path.read_bytes()).hexdigest(),
    "overlay_manifest_sha256": freeze["overlay_manifest_sha256"],
    "overlay_weights_sha256": freeze["overlay_weights_sha256"],
    "reserved_seed_blocks_opened": False,
}
(root / "postfreeze_validation.json").write_text(
    json.dumps(validation, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(validation, indent=2, sort_keys=True))
PY

"${PYTHON}" examples/qualify_neural_hmsc_student_t.py \
  --release-registry "${REGISTRY_ROOT}" \
  --variable-registry "${VARIABLE_REGISTRY_ROOT}" \
  --m56-root "${M56_ROOT}" \
  validate \
  --freeze-root "${RUN_ROOT}" \
  > "${RUN_ROOT}/read_only_validation.json"

sha256sum "${RUN_ROOT}/freeze.json" > "${RUN_ROOT}/freeze.sha256"
printf 'wall_time_seconds=%s\n' "${SECONDS}" | tee "${RUN_ROOT}/wall_time.txt"
echo "Milestone 57 train-validation completed: ${RUN_ROOT}"
echo "Reserved 323M-325M evaluation remains sealed."
