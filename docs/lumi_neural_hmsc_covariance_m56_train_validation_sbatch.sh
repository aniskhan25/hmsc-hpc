#!/bin/bash -l
#SBATCH --job-name=neural-m56-train-val
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
RUN_NAME="${RUN_NAME:-neural_hmsc_m56_train_validation_${SLURM_JOB_ID:-manual}}"
RUN_ROOT="${RUN_ROOT:-${USER_WORK}/hmsc-hpc-runs/${RUN_NAME}}"
REGISTRY_ROOT="${REGISTRY_ROOT:-${USER_WORK}/hmsc-hpc-deployments}"
CONFIRMATION="${CONFIRMATION:-}"
EXPECTED_CONFIRMATION="GENERATE_M56_CORRELATION_TRAIN_VALIDATION"
EXPECTED_PREREGISTRATION_SHA256="d99b63da87103c3d8891cb2fab5bb7ffad30a188ed7be920950345581f8b2d4b"
EXPECTED_AUDIT_SHA256="5bb9236967afb5a2a1adc166781f4a34359a7469150aa2e19117752dd1fce29c"

if [[ "${CONFIRMATION}" != "${EXPECTED_CONFIRMATION}" ]]; then
  echo "Refusing to open the Milestone 56 production train-validation blocks." >&2
  echo "Set CONFIRMATION=${EXPECTED_CONFIRMATION}." >&2
  exit 2
fi

if [[ -e "${RUN_ROOT}" ]]; then
  echo "Refusing to reuse an existing Milestone 56 run root: ${RUN_ROOT}" >&2
  exit 2
fi
if [[ ! -f "${REGISTRY_ROOT}/neural_hmsc_v0_1/release.json" ]]; then
  echo "Immutable neural_hmsc_v0_1 release is missing under ${REGISTRY_ROOT}." >&2
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

OBSERVED_PREREGISTRATION_SHA256="$(
  sha256sum docs/neural_hmsc_m56_covariance_preregistration_2026-07-23.md |
    awk '{print $1}'
)"
OBSERVED_AUDIT_SHA256="$(
  sha256sum docs/neural_hmsc_m56_artifact_seed_audit_2026-07-23.json.md |
    awk '{print $1}'
)"
if [[ "${OBSERVED_PREREGISTRATION_SHA256}" != "${EXPECTED_PREREGISTRATION_SHA256}" ]]; then
  echo "Milestone 56 preregistration hash differs." >&2
  exit 2
fi
if [[ "${OBSERVED_AUDIT_SHA256}" != "${EXPECTED_AUDIT_SHA256}" ]]; then
  echo "Milestone 56 artifact/seed audit hash differs." >&2
  exit 2
fi

echo "Repository: ${REPO_DIR}"
echo "Run root: ${RUN_ROOT}"
echo "Release registry: ${REGISTRY_ROOT}"
echo "Protocol action: one-shot correlation training and fixed validation"
echo "Authorized training block: 211000001-211000324"
echo "Authorized validation block: 212000001-212000324"
echo "Reserved blocks remain sealed: 213000001-215000324"
"${PYTHON}" -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"

SECONDS=0
"${PYTHON}" examples/qualify_neural_hmsc_covariance.py \
  --release-registry "${REGISTRY_ROOT}" \
  train-validate \
  --output "${RUN_ROOT}" \
  --confirmation "${CONFIRMATION}" \
  > "${RUN_ROOT}.stdout.json"

"${PYTHON}" - "${RUN_ROOT}" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
freeze_path = root / "freeze.json"
freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
gates = freeze["validation_gates"]
checks = {
    "protocol_id": freeze["protocol_id"] == "neural_hmsc_fixed_probit_covariance_m56_v1",
    "mode": freeze["mode"] == "production_train_validation",
    "training_seed_range": freeze["training_seed_range"] == [211000001, 211000324],
    "validation_seed_range": freeze["validation_seed_range"] == [212000001, 212000324],
    "reserved_seed_blocks_opened": freeze["reserved_seed_blocks_opened"] is False,
    "model_seed": freeze["training"]["seed"] == 211900001,
    "training_count": freeze["training"]["community_count"] == 324,
    "training_epochs": freeze["training"]["epochs"] == 100,
    "training_batch_size": freeze["training"]["batch_size"] == 9,
    "preregistration_hash": (
        freeze["protocol_hashes"]["preregistration_sha256"]
        == "d99b63da87103c3d8891cb2fab5bb7ffad30a188ed7be920950345581f8b2d4b"
    ),
    "audit_hash": (
        freeze["protocol_hashes"]["artifact_seed_audit_sha256"]
        == "5bb9236967afb5a2a1adc166781f4a34359a7469150aa2e19117752dd1fce29c"
    ),
    "bound_release": (
        freeze["base_binding"]["release_content_sha256"]
        == "affcfe10d2f9586e97432ae07754ab829e929ffdb62f41fb71779ad5f3ed12c8"
    ),
    "bound_weights": (
        freeze["base_binding"]["weights_sha256"]
        == "bb6e76d3ec9bc5e294ceac3051c3b2d7e5273db5053cfa5ceac676913d6265d9"
    ),
    "bound_calibration": (
        freeze["base_binding"]["calibration_sha256"]
        == "595fc0796d36802002cee09b270d53162f1fce100b83aecd32476e0958a0fd94"
    ),
    "gate_decision_consistent": freeze["validation_passed"] == all(gates.values()),
    "overlay_manifest_present": (root / "overlay/correlation_overlay.json").is_file(),
    "overlay_weights_present": (root / "overlay/correlation_head.weights.h5").is_file(),
}
if not all(checks.values()):
    raise SystemExit(f"Milestone 56 post-freeze validation failed: {checks}")

validation = {
    "schema_version": 1,
    "kind": "neural_hmsc_m56_train_validation_postfreeze_validation",
    "validated": True,
    "checks": checks,
    "validation_passed": freeze["validation_passed"],
    "failed_gates": [name for name, value in gates.items() if not bool(value)],
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

sha256sum "${RUN_ROOT}/freeze.json" > "${RUN_ROOT}/freeze.sha256"
printf 'wall_time_seconds=%s\n' "${SECONDS}" | tee "${RUN_ROOT}/wall_time.txt"
echo "Milestone 56 train-validation completed: ${RUN_ROOT}"
echo "Reserved 213M-215M evaluation remains sealed."
