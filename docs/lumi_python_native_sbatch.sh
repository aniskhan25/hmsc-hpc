#!/bin/bash -l
#SBATCH --job-name=pyhmsc-native
#SBATCH --account=project_462000131
#SBATCH --partition=small-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=output/%x-%j.out
#SBATCH --error=output/%x-%j.err

set -euo pipefail

# Submit this script from the hmsc-hpc repository root:
#   sbatch docs/lumi_python_native_sbatch.sh
#
# The script expects a pre-created virtual environment. Create it once on LUMI
# before submitting jobs:
#
#   cd /scratch/project_462000131/anisrahm
#   module use /appl/local/csc/modulefiles
#   module load tensorflow/2.16
#   mkdir -p venvs
#   python3 -m venv --system-site-packages venvs/hmsc_tf_env
#   source venvs/hmsc_tf_env/bin/activate
#   python3 -m pip install --upgrade pip
#   python3 -m pip install --no-deps tf-keras==2.16.* tensorflow-probability==0.24.*
#   python3 -m pip install --no-deps /path/to/hmsc-hpc
#
# If your LUMI module environment uses a different TensorFlow/AI stack, adjust
# the module lines below but keep the GPU sanity check.

PROJECT_ID="project_462000131"
USER_WORK="/scratch/${PROJECT_ID}/anisrahm"
VENV="${USER_WORK}/venvs/hmsc_tf_env"
PYTHON="${VENV}/bin/python3"
RUN_ROOT="${USER_WORK}/hmsc-hpc-runs/${SLURM_JOB_ID:-manual}"
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
MODEL_CONFIG="${MODEL_CONFIG:-${REPO_DIR}/examples/projects/fixed_poisson/model.yaml}"

mkdir -p output "${USER_WORK}/hmsc-hpc-runs" "${RUN_ROOT}"

module use /appl/local/csc/modulefiles
module load tensorflow/2.16

if [[ ! -f "${VENV}/bin/activate" ]]; then
  cat >&2 <<EOF
Missing virtual environment:
  ${VENV}

Create it once on LUMI, then resubmit:

  cd ${USER_WORK}
  module use /appl/local/csc/modulefiles
  module load tensorflow/2.16
  mkdir -p venvs
  python3 -m venv --system-site-packages venvs/hmsc_tf_env
  source venvs/hmsc_tf_env/bin/activate
  python3 -m pip install --upgrade pip
  python3 -m pip install --no-deps tf-keras==2.16.* tensorflow-probability==0.24.*
  python3 -m pip install --no-deps ${REPO_DIR}

If TensorFlow is provided through a different LUMI AI/container stack, create
the venv from that stack and update docs/lumi_python_native_sbatch.sh.
EOF
  exit 2
fi

source "${VENV}/bin/activate"
cd "${REPO_DIR}"

echo "Repository: ${REPO_DIR}"
echo "Run root: ${RUN_ROOT}"
echo "Model config: ${MODEL_CONFIG}"
echo "Python: ${PYTHON}"
"${PYTHON}" -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"
"${PYTHON}" -c "import tf_keras; print('tf_keras:', tf_keras.__version__)"
"${PYTHON}" -c "import tensorflow_probability as tfp; print('TFP:', tfp.__version__)"
"${PYTHON}" -c "import hmsc, pyhmsc; print('hmsc-hpc import: ok')"

"${PYTHON}" -m pyhmsc compile "${MODEL_CONFIG}" --output "${RUN_ROOT}/compiled"
"${PYTHON}" -m pyhmsc validate-init "${RUN_ROOT}/compiled/init.json" --strict

srun "${PYTHON}" -m pyhmsc sample \
  "${RUN_ROOT}/compiled/init.json" \
  --output "${RUN_ROOT}/posterior.h5" \
  --samples "${SAMPLES:-1000}" \
  --transient "${TRANSIENT:-500}" \
  --thin "${THIN:-10}" \
  --verbose "${VERBOSE:-100}"

"${PYTHON}" -m pyhmsc summarize "${RUN_ROOT}/posterior.h5" --param Beta \
  > "${RUN_ROOT}/beta_summary.txt"

echo "Done. Outputs are in ${RUN_ROOT}"
