# Direct R/Python HMSC Fixture Parity

LUMI job `19984923` ran the compact direct R/Python HMSC-HPC fixture parity
workflow on 2026-07-18:

```text
RUN_ROOT=/scratch/project_462000131/anisrahm/hmsc-hpc-runs/direct_r_python_parity_19984923
samples=1000 transient=500 thin=10 chains=2 fp=64
```

The workflow compares Python-native HMSC compile/sample against an R-created
`Hmsc` object imported through the original RDS Hmsc-HPC compatibility boundary.
It does not involve neural calibration.

## Fixes Found During Requalification

The first fixture attempts were useful failure probes:

- `19983642`: tiny smoke fixtures were too small for posterior gates and the
  predictive delta gate used absolute error deltas.
- `19983758`: the larger no-trait Poisson fixture exposed that R/Hmsc leaves
  binary 0/1 indicator columns unscaled.
- `19984154`/`19984279`: the iid fixture exposed R/Hmsc requirements that
  `studyDesign` contain factor columns only.

The Python-native/R bridge path now:

- leaves intercept, constant, and 0/1 indicator columns unscaled;
- stores indicator scale parameters as mean `0`, sd `1`;
- applies the same fallback scaling rule during prediction;
- writes iid random-level `studyDesign` columns as factors;
- narrows R `studyDesign` to the random-level columns used by `ranLevels`.

## Fixed-Effect No-Trait Fixture

Config:

```text
examples/projects/simulated_poisson_recovery/model.yaml
```

Result: passed boundary and predictive non-degradation gates.

Boundary arrays:

| array | passed | max abs diff |
| --- | --- | --- |
| `Y` | true | `0.0` |
| `X` | true | `6.661338147750939e-16` |
| `T` | true | `0.0` |

Diagnostics:

| parameter | mean correlation | mean MAE |
| --- | --- | --- |
| `Beta` | `0.685593` | `0.307356` |
| `Gamma` | `0.463099` | `0.193696` |

Predictive metrics:

| model | prediction MAE | prediction RMSE | Poisson deviance |
| --- | --- | --- | --- |
| Python-native | `0.787438` | `1.063690` | `1.092040` |
| R boundary | `1.143570` | `1.436910` | `1.670570` |

The posterior correlations are diagnostic-only for this compact fixture; they
are not used as acceptance gates.

## IID Random-Intercept Fixture

Config:

```text
examples/projects/simulated_spatial_validation/model_iid.yaml
```

Result: passed boundary, predictive non-degradation, and random-association
diagnostic gates.

Boundary arrays:

| array | passed | max abs diff |
| --- | --- | --- |
| `Y` | true | `0.0` |
| `X` | true | `1.1102230246251565e-16` |
| `T` | true | `0.0` |
| `Pi` | true | `0.0` |

Diagnostics:

| parameter | mean correlation | mean MAE |
| --- | --- | --- |
| `Beta` | `0.998911` | `0.119891` |
| `Gamma` | `1.000000` | `0.075238` |
| random association | `0.951692` | `0.822850` |

Predictive metrics:

| model | prediction MAE | Brier score | log loss |
| --- | --- | --- | --- |
| Python-native | `0.267434` | `0.0928862` | `0.333748` |
| R boundary | `0.376020` | `0.1781320` | `0.528461` |

## Spatial Random-Effect Fixtures

Spatial boundary semantics were inspected first in
`docs/spatial_r_python_hmsc_boundary_inspection_2026-07-18.md`; direct parity
was then run for compact Full, GPP, and NNGP fixtures.

Two implementation issues were found and fixed during the spatial parity run:

- random-level HDF5 export now pads variable latent-factor shapes across saved
  samples/chains;
- the legacy RDS GPP import path now uses the same jitter/clipping
  stabilization as the Python-native GPP loader.

Results:

| fixture | LUMI job | parity | boundary | prediction MAE delta | association corr |
| --- | --- | --- | --- | --- | --- |
| `model_spatial_full.yaml` | `19995352` | passed | exact | `-0.077115` | `0.949683` |
| `model_spatial_gpp.yaml` | `19999784` | passed | exact | `0.002227` | `0.999998` |
| `model_spatial_nngp.yaml` | `19999784` | passed | exact | `0.001771` | `0.999999` |

## Interpretation

This extends Python-only HMSC boundary parity beyond the Whittaker
trait/phylogeny model to:

- fixed-effect no-trait Poisson models with continuous and binary indicators;
- iid random-intercept models with study-design factor coding and `Pi` parity.
- compact spatial Full, GPP, and NNGP random-effect fixtures with exact
  boundary arrays and predictive non-degradation.

It does not claim full posterior equivalence for compact fixed-effect fixtures:
posterior mean correlations are reported as diagnostics because independent
MCMC initialization/mixing can dominate small fixture comparisons. The enforced
claims are boundary equality and predictive non-degradation for these fixtures,
with random-association diagnostics enforced for the iid case.

## Larger Spatial Real-Data Requalification

After compact spatial fixtures passed, the same direct R/Python parity protocol
was extended to the real-data Big Spatial Plant full-spatial configuration:

```text
examples/projects/big_spatial_plants_validation/model_spatial_full.yaml
```

LUMI job `20000066` completed both Python-native and R-boundary MCMC runs with
`samples=1000`, `transient=500`, `thin=10`, `chains=2`, and `fp=64`. The final
report was regenerated from the completed posteriors using
`--reuse-existing-posteriors` after correcting stale remote source/data sync.

Result: passed.

| gate/metric | result |
| --- | --- |
| boundary arrays | `Y`, `X`, `T`, and `Pi` passed; max `X` difference `8.881784197001252e-16` |
| `Beta` diagnostic | mean correlation `0.984128`, mean MAE `0.164287` |
| `Gamma` diagnostic | mean correlation `0.999570`, mean MAE `0.126621` |
| spatial association diagnostic | correlation `0.749967`, MAE `1.099881` |
| predictive MAE delta | `-0.013724`, Python-native better than R boundary |
| Python-native predictive metrics | MAE `0.099725`, Brier `0.0438051`, log loss `0.148042` |
| R-boundary predictive metrics | MAE `0.113449`, Brier `0.0514421`, log loss `0.175610` |

This supports the broader Python-only HMSC parity claim for the currently
covered scope: fixed-effect Whittaker trait/phylogeny, compact fixed/iid/spatial
fixtures, and one larger real-data full-spatial ecological project.
