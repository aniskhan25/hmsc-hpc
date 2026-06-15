# Python-Native Tutorial

Compile and sample without R:

```bash
python -m pyhmsc compile examples/projects/fixed_poisson/model.yaml --output run
python -m pyhmsc validate-init run/init.json --strict
python -m pyhmsc sample run/init.json --output run/posterior.h5 --samples 100 --transient 100 --thin 1
python -m pyhmsc summarize run/posterior.h5 --param Beta
python -m pyhmsc ppc run/posterior.h5 \
  --X examples/projects/fixed_poisson/data/X.csv \
  --Y examples/projects/fixed_poisson/data/Y.csv \
  --output run/ppc.csv
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
GPU scripts use the `dev-g` partition with a 30 minute default walltime for the
bundled smoke-scale runs.
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

Run the opt-in slow recovery suite on LUMI:

```bash
sbatch docs/lumi_python_native_recovery_tests_sbatch.sh
```

Known working LUMI environment:

```text
module load tensorflow/2.16
Python: /scratch/project_462000131/anisrahm/venvs/hmsc_tf_env/bin/python3
TensorFlow: 2.16.1
GPU visible to TensorFlow: yes
tf_keras: 2.16.0
tensorflow_probability: 0.24.0
h5py: required for init_arrays.h5 and posterior.h5
```

For a fresh LUMI venv, install the non-TensorFlow Python dependencies with:

```bash
python3 -m pip install --upgrade-strategy only-if-needed -r /path/to/hmsc-hpc/requirements_lumi.txt
python3 -m pip install --no-deps /path/to/hmsc-hpc
```

The LUMI requirements file intentionally avoids core system-site packages such
as NumPy, pandas, SciPy, and TensorFlow so the CSC module/base container is not
upgraded or replaced inside the venv.

The fixed Poisson example has been verified on LUMI with 2 chains, 1000 saved
samples, 500 transient iterations, and thin 10, writing `posterior.h5` under the
scratch run directory.

The deterministic simulated spatial validation can be run as:

```bash
RUN_NAME=spatial_validation_test sbatch docs/lumi_spatial_validation_sbatch.sh
```

This runs fixed, iid random-intercept, and full spatial random-intercept models
for `examples/projects/simulated_spatial_validation`, then writes
`spatial_validation_report.txt` under the scratch run directory.

Validated LUMI run `spatial_validation_full_codex` completed in 7 minutes 15
seconds on `dev-g` with TensorFlow 2.16 and an MI250X GPU. All three models
recovered the nonzero beta signs (`4 / 4`) and covered all species and sites in
posterior predictive checks (`5 / 5` species, `36 / 36` site richness). The
full spatial random-intercept model had stronger Eta/truth recovery
(`0.868592`) than the iid random-intercept model (`0.675717`).

The deterministic spatial random-slope validation can be run as:

```bash
RUN_NAME=spatial_random_slope_validation sbatch docs/lumi_spatial_random_slope_validation_sbatch.sh
```

This runs full spatial, GPP spatial, and NNGP spatial random-slope models for
`examples/projects/simulated_spatial_random_slope_validation`, then writes
`spatial_random_slope_validation_report.txt` under the scratch run directory.
The analyzer reports beta sign recovery, posterior predictive coverage, Eta
recovery, and both Lambda intercept and slope recovery.

Validated LUMI run `spatial_random_slope_validation_cli_fixed` completed the
full/GPP/NNGP spatial random-slope validation on `dev-g`. The run recovered beta
signs for all three models (`4 / 4`) and covered all species and site richness
checks (`5 / 5` species, `49 / 49` sites). Eta/truth recovery was strongest for
full spatial (`0.831875`) and GPP (`0.766687`) and weak for NNGP (`0.107724`);
Lambda intercept recovery was strong for all three (`0.886723`, `0.921705`,
`0.978135`), while Lambda slope recovery was weak to moderate (`0.022262`,
`0.384912`, `0.331418`).

A stronger companion scenario is available for testing whether weak Lambda slope
recovery is signal-limited:

```bash
RUN_NAME=spatial_random_slope_strong_validation \
PROJECT_DIR="${PWD}/examples/projects/simulated_spatial_random_slope_strong_validation" \
SAMPLES=2000 TRANSIENT=1000 THIN=10 \
sbatch docs/lumi_spatial_random_slope_validation_sbatch.sh
```

This uses a normal response, 81 sites, lower observation noise, and stronger
random-slope loadings.

Validated LUMI run `spatial_random_slope_strong_validation_real` completed on
`dev-g` as job `19272750` in 20 minutes 41 seconds with 2 chains, 2000 saved
samples, 1000 transient iterations, and thin 10. All three spatial random-slope
models recovered beta signs (`6 / 6`), species PPC (`6 / 6`), and site richness
PPC (`81 / 81`). Lambda slope recovery was strong for full spatial
(`0.999992`), GPP (`0.999573`), and NNGP (`0.999988`), confirming that the weak
baseline Lambda slope result is signal-limited. Eta recovery remained weaker
for NNGP (`0.674563`) than full spatial (`0.962924`) and GPP (`0.895142`).

A focused spatial Eta validation is available for checking whether NNGP latent
recovery improves as neighbor count increases:

```bash
RUN_NAME=spatial_eta_validation \
SAMPLES=1500 TRANSIENT=750 THIN=10 \
sbatch docs/lumi_spatial_eta_validation_sbatch.sh
```

This uses `examples/projects/simulated_spatial_eta_validation`, a normal-response
100-site simulation with one known spatial latent factor. It compares full
spatial, GPP, and NNGP models with 5, 10, and 20 neighbors, then writes
`spatial_eta_validation_report.txt` under the scratch run directory.
The full five-model run can exceed one 30-minute `dev-g` allocation; reuse the
same `RUN_NAME` with `MODELS=spatial_nngp_20` or another subset to resume
missing models.

Validated LUMI run `spatial_eta_validation_real` completed after one resume. The
first job, `19273473`, timed out after completing full spatial, GPP, NNGP-5,
and NNGP-10; resume job `19275812` completed NNGP-20 in 9 minutes 27 seconds
and generated the combined report. Full spatial recovered Eta/truth correlation
`0.988845`, GPP recovered `0.894420`, while NNGP remained weak at neighbor
counts 5, 10, and 20 (`0.162178`, `0.101626`, `0.199853`). All models covered
species PPC `6 / 6` and site richness PPC `100 / 100`.

The compact real-data big-spatial plant validation can be run as:

```bash
RUN_NAME=big_spatial_real_validation sbatch docs/lumi_big_spatial_validation_sbatch.sh
```

This uses `examples/projects/big_spatial_plants_validation`, a 400-site,
40-species subset derived from `examples/big_spatial`, and compares fixed, iid
site-level, and full spatial site-level random-intercept models. The analyzer
reports posterior predictive coverage/errors and nearest-neighbor residual
correlation.

Validated LUMI run `big_spatial_real_validation_codex` completed in 9 minutes
43 seconds on `dev-g` with TensorFlow 2.16 and an MI250X GPU. The fixed model
covered site richness for `309 / 400` sites with nearest-neighbor residual
correlation `0.427027`; the iid model covered `400 / 400` sites with residual
correlation `0.299458`; the full spatial model covered `400 / 400` sites and
reduced residual correlation to `-0.291249`.

The Whittaker plant real-data validation can be run without R as:

```bash
RUN_NAME=whittaker_iid_long \
MODEL_CONFIG=examples/projects/whittaker_plants_hmsc_book/model_iid_site.yaml \
SAMPLES=3000 \
TRANSIENT=1000 \
THIN=10 \
VERBOSE=500 \
sbatch docs/lumi_whittaker_real_data_sbatch.sh
```

This uses a probit model with TMG, species CN traits, phylogenetic covariance,
and an iid site-level random intercept. The validated LUMI run completed in
under 10 minutes on `dev-g` with TensorFlow 2.16 and an MI250X GPU. It produced
species occupancy PPC coverage `75 / 75` and site richness PPC coverage
`52 / 52`, improving the fixed-effect Whittaker baseline site richness coverage
of `40 / 52`.

For models with random-level `Lambda` samples, export residual species
associations as a pair table:

```bash
python -m pyhmsc associations run/posterior.h5 \
  --output run/species_associations.csv
```

Or export the mean association matrix:

```bash
python -m pyhmsc associations run/posterior.h5 \
  --matrix \
  --output run/species_association_matrix.csv
```

Random-level effects and species loadings can be summarized directly:

```bash
python -m pyhmsc summarize run/posterior.h5 --param Eta --random-level 0
python -m pyhmsc summarize run/posterior.h5 --param Lambda --random-level 0
python -m pyhmsc diagnostics run/posterior.h5 --param Eta --random-level 0
python -m pyhmsc diagnostics run/posterior.h5 --param Lambda --random-level 0
python -m pyhmsc diagnostics run/posterior.h5 --param Lambda --random-level 0 --align-factors
python -m pyhmsc diagnostics run/posterior.h5 --param Associations --random-level 0
```

The sampler consumes the compiled `init.json` + `init_arrays.h5` artifact, not
raw CSV files directly. This keeps file loading, formula expansion, prior setup,
and parameter initialization outside the TensorFlow Gibbs sampler.

Supported no-R sampler inputs are fixed effects, traits, phylogenetic covariance
or Newick-derived covariance, iid random intercepts, iid random slopes, full
spatial random intercepts, GPP spatial random intercepts, and NNGP spatial
random intercepts. Full spatial, GPP spatial, and NNGP spatial random slopes
are supported with `type: spatial_full`, `type: spatial_gpp`, or
`type: spatial_nngp` plus `x_formula`.

NNGP random levels use deterministic previous-nearest-neighbor selection in the
compiled random-level order:

```yaml
random_levels:
  plot:
    column: plot
    type: spatial_nngp
    coords: [xcoord, ycoord]
    n_neighbors: 10
```

From Python:

```python
import pandas as pd
from pyhmsc import HmscModel

Y = pd.read_csv("examples/projects/fixed_poisson/data/Y.csv", index_col=0)
X = pd.read_csv("examples/projects/fixed_poisson/data/X.csv", index_col=0)

model = HmscModel(Y=Y, X=X, x_formula="~ forest_cover + elevation", distr="poisson")
fit = model.sample(samples=100, transient=100, thin=1, chains=2, init="python-native")
print(fit.beta_mean())
print(fit.ppc_summary(Y=Y, X=X))
print(fit.species_association_summary())
print(fit.eta_summary(level=0))
print(fit.lambda_summary(level=0))
print(fit.diagnostics("Eta", level=0))
print(fit.diagnostics("Lambda", level=0))
print(fit.diagnostics("Lambda", level=0, align=True))
print(fit.diagnostics("Associations", level=0))
```
