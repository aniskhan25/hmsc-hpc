# Python-Native Tutorial

Compile and sample without R:

```bash
python -m pyhmsc compile examples/projects/fixed_poisson/model.yaml --output run
python -m pyhmsc validate-init run/init.json --strict
python -m pyhmsc sample run/init.json --output run/posterior.h5 --samples 100 --transient 100 --thin 1
python -m pyhmsc summarize run/posterior.h5 --param Beta
```

Run the supported no-R example projects from one command:

```bash
python examples/run_python_native_smoke.py --clean
```

For a fast compile/validation-only check:

```bash
python examples/run_python_native_smoke.py --skip-sample --clean
```

On LUMI, use the Slurm template in
[`docs/lumi_python_native_sbatch.sh`](lumi_python_native_sbatch.sh). It uses
`project_462000131`, writes runs under
`/scratch/project_462000131/anisrahm/hmsc-hpc-runs`, and expects a virtual
environment at `/scratch/project_462000131/anisrahm/venvs/hmsc_tf_env`.
By default it runs the supported no-R examples: `fixed_poisson`,
`traits_phylogeny`, `iid_random_intercept`, and `spatial_full`.

To run one custom model config instead:

```bash
MODEL_CONFIG=/path/to/model.yaml sbatch docs/lumi_python_native_sbatch.sh
```

To run a subset of bundled examples:

```bash
EXAMPLE_PROJECTS="traits_phylogeny iid_random_intercept spatial_full" \
  sbatch docs/lumi_python_native_sbatch.sh
```

For one-chain-per-array-task sampling, use:

```bash
RUN_NAME=fixed_poisson_array sbatch docs/lumi_python_native_compile_sbatch.sh
RUN_NAME=fixed_poisson_array sbatch docs/lumi_python_native_array_sbatch.sh
RUN_NAME=fixed_poisson_array sbatch docs/lumi_python_native_merge_sbatch.sh
```

The compile script writes one shared native model under
`/scratch/project_462000131/anisrahm/hmsc-hpc-runs/<run_name>/compiled`. The
array script reads that shared `init.json` and writes per-chain files under
`/scratch/project_462000131/anisrahm/hmsc-hpc-runs/<run_name>/chains`, and the
merge script writes one metadata-preserving `posterior.h5`.

Inspect chain status before merging or after a failed array run:

```bash
python -m pyhmsc chain-status \
  /scratch/project_462000131/anisrahm/hmsc-hpc-runs/fixed_poisson_array/chains \
  --expected-chains 0 1 \
  --run-name fixed_poisson_array
```

Rerun failed chains only:

```bash
RUN_NAME=fixed_poisson_array sbatch --array=1 docs/lumi_python_native_array_sbatch.sh
```

To safely resubmit an array without overwriting completed chains:

```bash
RUN_NAME=fixed_poisson_array SKIP_EXISTING=1 sbatch docs/lumi_python_native_array_sbatch.sh
```

Known working LUMI environment:

```text
module load tensorflow/2.16
Python: /scratch/project_462000131/anisrahm/venvs/hmsc_tf_env/bin/python3
TensorFlow: 2.16.1
GPU visible to TensorFlow: yes
tf_keras: 2.16.0
tensorflow_probability: 0.24.0
```

The fixed Poisson example has been verified on LUMI with 2 chains, 1000 saved
samples, 500 transient iterations, and thin 10, writing `posterior.h5` under the
scratch run directory.

The sampler consumes the compiled `init.json` + `init_arrays.h5` artifact, not
raw CSV files directly. This keeps file loading, formula expansion, prior setup,
and parameter initialization outside the TensorFlow Gibbs sampler.

Supported no-R sampler inputs are fixed effects, traits, phylogenetic covariance
or Newick-derived covariance, iid random intercepts, and full spatial random
intercepts. Random-slope metadata can be compiled for schema work, but
`validate-init --strict` marks it as not sampler-ready.

From Python:

```python
import pandas as pd
from pyhmsc import HmscModel

Y = pd.read_csv("examples/projects/fixed_poisson/data/Y.csv", index_col=0)
X = pd.read_csv("examples/projects/fixed_poisson/data/X.csv", index_col=0)

model = HmscModel(Y=Y, X=X, x_formula="~ forest_cover + elevation", distr="poisson")
fit = model.sample(samples=100, transient=100, thin=1, chains=2, init="python-native")
print(fit.beta_mean())
```
