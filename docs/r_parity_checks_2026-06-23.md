# R Parity Check Archive: 2026-06-23

This is the archived one-time model-construction parity check for the
Python-native compiler. The check verifies that selected Python-native compiled
artifacts match base R formula and factor encodings.

## Environment

- Host: LUMI
- Repository: `/scratch/project_462000131/anisrahm/hmsc-hpc`
- Output artifacts:
  `/scratch/project_462000131/anisrahm/hmsc-hpc-runs/r_parity_checks_2026-06-23`
- Python module: `cray-python/3.11.7`
- R module: `cray-R/4.4.0`
- R version: `Rscript (R) version 4.4.0 (2024-04-24)`
- Scratch parity venv:
  `/scratch/project_462000131/anisrahm/venvs/hmsc_r_parity_env`

The scratch parity venv was used only for this host-side parity check because
the TensorFlow Python module runs inside a container and cannot call host
`Rscript`. The venv uses system site packages and adds only `patsy`.

## Command

```bash
module load cray-python/3.11.7 cray-R/4.4.0
source /scratch/project_462000131/anisrahm/venvs/hmsc_r_parity_env/bin/activate
python examples/run_r_parity_checks.py \
  --output /scratch/project_462000131/anisrahm/hmsc-hpc-runs/r_parity_checks_2026-06-23
```

## Result

All default parity cases passed.

### `tests/fixtures/fixed_effect/model.yaml`

- `native_compiled_validation`: passed
- `X_design`: passed
  - observed shape: `(6, 3)`
  - expected shape: `(6, 3)`
  - observed names: `Intercept`, `forest_cover`, `elevation`
  - expected names: `Intercept`, `forest_cover`, `elevation`
  - max absolute difference: `0.0`

### `tests/fixtures/fixed_effect/model_traits_phylo.yaml`

- `native_compiled_validation`: passed
- `X_design`: passed
  - observed shape: `(6, 3)`
  - expected shape: `(6, 3)`
  - observed names: `Intercept`, `forest_cover`, `elevation`
  - expected names: `Intercept`, `forest_cover`, `elevation`
  - max absolute difference: `0.0`
- `trait_design`: passed
  - observed shape: `(3, 3)`
  - expected shape: `(3, 3)`
  - observed names: `Intercept`, `body_size`, `forest_specialist`
  - expected names: `Intercept`, `body_size`, `forest_specialist`
  - max absolute difference: `0.0`
- `phylo_cov`: passed
  - observed shape: `(3, 3)`
  - expected shape: `(3, 3)`
  - observed names: `sparrow`, `owl`, `woodpecker`
  - expected names: `sparrow`, `owl`, `woodpecker`
  - max absolute difference: `0.0`

### `examples/projects/iid_random_intercept/model.yaml`

- `native_compiled_validation`: passed
- `X_design`: passed
  - observed shape: `(4, 2)`
  - expected shape: `(4, 2)`
  - observed names: `Intercept`, `x`
  - expected names: `Intercept`, `x`
  - max absolute difference: `0.0`
- `random_level_plot_codes`: passed
  - observed shape: `(4,)`
  - expected shape: `(4,)`
  - observed levels: `a`, `b`
  - expected levels: `a`, `b`

## Interpretation

The Python-native compiler matches base R for the selected formula expansion,
trait-design expansion, phylogenetic covariance ordering, and iid random-level
factor encoding cases. This supports the Python replacement of the R-side model
construction boundary for the covered model families.

This check does not validate MCMC numerical behavior; that is covered by the
separate simulation, real-data, posterior predictive, and LUMI validation runs.
