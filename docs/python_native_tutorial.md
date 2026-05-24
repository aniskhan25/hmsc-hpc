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
environment at `/scratch/project_462000131/anisrahm/hmsc_tf_env`.

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
