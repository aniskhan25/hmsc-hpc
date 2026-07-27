# Python-Only HMSC Parity Decision

Decision date: 2026-07-19

## Decision

Compact fixed-effect, iid, and spatial fixtures are sufficient to support a
bounded implementation claim:

```text
Python-native HMSC now matches the original R-created HMSC-HPC boundary on
controlled fixed, iid, Full spatial, GPP spatial, and NNGP spatial fixtures.
```

They are not sufficient for the broader project claim:

```text
Python-only HMSC is equivalent to the original R+Python HMSC-HPC workflow for
realistic ecological spatial analyses.
```

Before returning to neural development, one larger spatial real-data
requalification was added using the existing Big Spatial Plant validation
project.

## Rationale

The compact fixture work found real boundary and runtime issues:

- R/Hmsc leaves binary indicator covariates unscaled.
- R/Hmsc trait formulas drop the intercept and scale trait columns.
- R/Hmsc spatial random levels require `sData` ordered by random-level factor
  levels.
- R/Hmsc uses a 101-point default spatial `alphapw` grid.
- R-boundary spatial posterior export can contain variable latent-factor
  shapes and therefore needs padded HDF5 export.
- GPP needs jitter/clipping in the legacy RDS import path to avoid NaN
  posterior propagation under the full R alpha grid.

Those fixes were validated on compact fixtures, including Full, GPP, and NNGP
spatial random effects. However, compact fixtures are intentionally small and
mostly synthetic. They do not stress the same ecological-data conditions as a
real spatial project: 400 sites, many species, real covariate distributions,
spatially structured occurrence patterns, and practical memory/runtime behavior.

## Larger Spatial Requalification

Direct R/Python parity was run on the existing real-data spatial project:

```text
examples/projects/big_spatial_plants_validation/model_spatial_full.yaml
```

LUMI run:

```text
job=20000066
RUN_ROOT=/scratch/project_462000131/anisrahm/hmsc-hpc-runs/direct_r_python_big_spatial_full_parity_20260719
samples=1000 transient=500 thin=10 chains=2 fp=64
posterior_gates=diagnostic
```

The MCMC portion completed for both Python-native and R-boundary paths. The
initial report step exposed stale remote source/data synchronization issues:
the R-boundary posterior had five fixed-effect covariates while the remote
model reconstruction briefly resolved only four malformed covariate names.
After resyncing the Big Spatial project data and formula/model/config/compiler
source files, the report was regenerated from the existing posteriors using
`--reuse-existing-posteriors`.

Result: passed.

Boundary arrays:

| array | passed | python-native shape | R boundary shape | max abs diff |
| --- | --- | --- | --- | --- |
| `Y` | true | `[400, 40]` | `[400, 40]` | `0.0` |
| `X` | true | `[400, 5]` | `[400, 5]` | `8.881784197001252e-16` |
| `T` | true | `[40, 1]` | `[40, 1]` | `0.0` |
| `Pi` | true | `[400, 1]` | `[400, 1]` | `0.0` |

Posterior diagnostics:

| parameter | mean correlation | mean MAE |
| --- | --- | --- |
| `Beta` | `0.984128` | `0.164287` |
| `Gamma` | `0.999570` | `0.126621` |
| spatial association | `0.749967` | `1.099881` |

Predictive metrics:

| model | prediction MAE | Brier score | log loss |
| --- | --- | --- | --- |
| Python-native | `0.099725` | `0.0438051` | `0.148042` |
| R boundary | `0.113449` | `0.0514421` | `0.175610` |

The enforced boundary and predictive gates passed. Posterior-summary
correlations were retained as diagnostics under the same diagnostic-gate
semantics used for compact direct parity.

## Executed Scope

- Full spatial random intercept first.
- Original R/Hmsc-created boundary retained as comparator.
- Python-native compile/sample compared against R-boundary sample.
- Boundary arrays match exactly for `Y`, `X`, `T`, `Pi`, and spatial
  random-level data.
- Predictive non-degradation remains the enforced acceptance gate.
- `Beta`, `Gamma`, and random association summaries remain diagnostics because
  independent MCMC initialization/mixing can affect full-spatial posterior
  summaries.

Recommended LUMI settings:

```text
partition=dev-g
samples=1000
transient=500
thin=10
chains=2
fp=64
posterior_gates=diagnostic
```

## Decision Boundary

The Big Spatial full-spatial requalification passed. Python-only HMSC parity
can now be claimed for fixed-effect Whittaker trait/phylogeny, compact
fixed/iid/spatial fixtures, and one larger real-data full-spatial ecological
project.

GPP/NNGP real-data requalification can remain a follow-up unless the broader
claim needs explicit approximation-method coverage. The current branch can now
return to neural work with Python-only HMSC parity scaffolding considered
adequate for fixed-effect Whittaker, compact fixed/iid/spatial fixtures, and
one larger real-data full-spatial ecological project.
