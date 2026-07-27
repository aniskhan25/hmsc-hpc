# Whittaker R/Python HMSC Boundary Parity

LUMI job `19967782` ran the direct Whittaker parity workflow on 2026-07-17:

```text
RUN_NAME=whittaker_r_python_parity_staged3_20260717_123506
RUN_ROOT=/scratch/project_462000131/anisrahm/hmsc-hpc-runs/whittaker_r_python_parity_staged3_20260717_123506
samples=1000 transient=500 thin=10 chains=2 fp=64
```

The workflow compares Python-native HMSC compile/sample against an R-created
`Hmsc` object imported through the original RDS Hmsc-HPC compatibility boundary.
It does not involve neural calibration.

## Initial Result

Parity did not pass in job `19967782`.

Boundary arrays:

| array | passed | python-native shape | R boundary shape | max abs diff |
| --- | --- | --- | --- | --- |
| `Y` | true | `[40, 75]` | `[40, 75]` | `0.0` |
| `X` | false | `[40, 2]` | `[40, 2]` | `0.05213465840545273` |
| `T` | false | `[75, 2]` | `[75, 1]` | n/a |
| `C` | true | `[75, 75]` | `[75, 75]` | `0.0` |

Posterior summaries:

| parameter | shape agreement | mean correlation | mean MAE |
| --- | --- | --- | --- |
| `Beta` | true (`[2,75]` vs `[2,75]`) | `0.988715` | `0.096009` |
| `Gamma` | false (`[2,2]` vs `[2,1]`) | n/a | n/a |

Held-out predictive metrics were close:

| model | Brier | log loss | macro AUC | prevalence MAE | richness MAE |
| --- | --- | --- | --- | --- | --- |
| Python-native | `0.074204` | `0.264660` | `0.548967` | `0.070449` | `3.357272` |
| R boundary | `0.075277` | `0.264236` | `0.551808` | `0.072541` | `3.740925` |

## Initial Interpretation

The Python-only sampler path is predictively close to the R-boundary sampler on
this Whittaker split, and the core response and phylogeny arrays match exactly.
However, Python-native HMSC is not yet equivalent to the original R+Python
HMSC-HPC boundary for this trait/phylogeny model.

The immediate mismatch is preprocessing:

- R/Hmsc `XScaled` differs from Python-native `X` despite matching shape.
- R/Hmsc `TrScaled` has one trait column, while Python-native `T` includes
  intercept plus `CN`.
- Consequently, `Gamma` posterior shapes differ and trait-effect parity cannot
  be claimed.

## Fix

The Python-native compiler now matches the R/Hmsc boundary semantics for this
fixed-effect trait/phylogeny model:

- non-intercept fixed-effect covariates are centered and scaled by sample
  standard deviation, matching R `scale()`;
- trait design matrices drop the intercept before scaling;
- trait values are centered and scaled by sample standard deviation;
- scale parameters are stored in native metadata and applied during prediction;
- model-backed RDS/HDF5 prediction falls back to training-data scaling when
  native metadata is absent.

## Requalification

LUMI job `19983202` reran the same Whittaker parity workflow on 2026-07-18:

```text
RUN_NAME=whittaker_r_python_parity_scaled_20260718_082539
RUN_ROOT=/scratch/project_462000131/anisrahm/hmsc-hpc-runs/whittaker_r_python_parity_scaled_20260718_082539
samples=1000 transient=500 thin=10 chains=2 fp=64
```

Parity passed.

Boundary arrays:

| array | passed | python-native shape | R boundary shape | max abs diff |
| --- | --- | --- | --- | --- |
| `Y` | true | `[40, 75]` | `[40, 75]` | `0.0` |
| `X` | true | `[40, 2]` | `[40, 2]` | `2.7755575615628914e-17` |
| `T` | true | `[75, 1]` | `[75, 1]` | `2.6492141813605485e-11` |
| `C` | true | `[75, 75]` | `[75, 75]` | `0.0` |

Posterior summaries:

| parameter | shape agreement | mean correlation | mean MAE |
| --- | --- | --- | --- |
| `Beta` | true (`[2,75]` vs `[2,75]`) | `0.999832` | `0.012697` |
| `Gamma` | true (`[2,1]` vs `[2,1]`) | `1.000000` | `0.015976` |

Held-out predictive metrics were effectively equivalent:

| model | Brier | log loss | macro AUC | prevalence MAE | richness MAE |
| --- | --- | --- | --- | --- | --- |
| Python-native | `0.074833` | `0.262531` | `0.551808` | `0.071772` | `3.611590` |
| R boundary | `0.074833` | `0.262716` | `0.554649` | `0.071622` | `3.604650` |

## Current Interpretation

The direct Whittaker parity gate now supports Python-only HMSC equivalence to
the original R+Python HMSC-HPC boundary for this fixed-effect trait/phylogeny
model. This is not a claim about every HMSC model family; random effects,
spatial terms, and additional trait/formula cases still need direct boundary
parity checks.

## Next Step

The implementation path for the next direct R/Python parity surface has been
added:

- `examples/run_direct_r_python_parity.py` runs the reusable direct parity
  workflow for arbitrary YAML configs;
- `docs/lumi_direct_r_python_parity_sbatch.sh` stages the R init and then runs
  `examples/projects/simulated_poisson_recovery/model.yaml` followed by
  `examples/projects/simulated_spatial_validation/model_iid.yaml`;
- `pyhmsc.r_bridge` now emits iid `studyDesign` and `ranLevels` into the
  R-created `Hmsc` boundary object.

LUMI job `19984923` completed this direct fixture parity workflow successfully;
the result is recorded in
`docs/direct_r_python_hmsc_fixture_parity_2026-07-18.md`.

The next direct parity surface is spatial/random-slope fixtures. Those should
wait until their R/Hmsc preprocessing and random-level boundary semantics are
inspected explicitly.
