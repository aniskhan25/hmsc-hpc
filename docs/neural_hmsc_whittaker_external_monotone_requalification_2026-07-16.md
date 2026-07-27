# Whittaker External-Monotone Real-Data Requalification

Date: 2026-07-16

LUMI job: `19948534`

Remote run root:
`/scratch/project_462000131/anisrahm/hmsc-hpc-runs/neural_whittaker_external_monotone_requalification_retry_20260716`

Downloaded local summary:
`/private/tmp/neural_whittaker_extmono_19948534`

## Scope

This run evaluates the promoted `external_monotone` coefficient calibration on
the real Whittaker plant held-out-site split. The model is the fixed-effect
probit `presence ~ TMG` runner used by the previous Whittaker neural
requalification. Traits, phylogeny, and latent site effects remain excluded.

The run compares neural artifacts against the Python-native MCMC reference
written by `HmscModel.sample(...)`. It does not by itself prove equivalence to
an R-built HMSC object exported into the original R+Python HMSC-HPC workflow.
That remains a separate parity step.

## Configuration

- coefficient calibration: `external_monotone`
- training datasets: `512`
- calibration datasets: `128`
- SBC datasets: `128`
- SBC draws: `512`
- external monotone datasets per regime: `4`
- external monotone max multiplier: `2.0`
- neural epochs: `120`
- neural chains/draws: `4 / 1000`
- MCMC chains/samples/transient/thin: `2 / 1000 / 500 / 5`
- elapsed job time: `00:10:09`
- workflow wall time: `576` seconds
- exit code: `0:0`

The first submission, job `19947994`, failed before report generation because
real Whittaker coefficient calibration was applied with unbatched `X`/`Y`
arrays. The runner was fixed to pass batched real-data arrays, and the retry
completed successfully.

## Acceptance

| gate | result |
| --- | --- |
| coefficient SBC acceptance | pass |
| held-out predictive acceptance | pass |
| combined qualification | pass |

Key acceptance metrics:

- coefficient SBC coverage: `0.9579`
- coefficient SBC rank mean: `0.4950`
- coefficient SBC rank variance: `0.0697`
- predictive Brier ratio vs uncalibrated: `1.0035`
- predictive log-loss ratio vs uncalibrated: `0.9981`
- predictive Brier ratio vs MCMC: `1.0425`
- predictive log-loss ratio vs MCMC: `1.0448`

## Held-Out Metrics

| model | Brier | log loss | macro AUC | prevalence MAE | richness MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| neural uncalibrated | 0.0769 | 0.2745 | 0.5386 | 0.0740 | 3.4263 |
| neural coefficient-calibrated | 0.0811 | 0.2915 | 0.5253 | 0.0947 | 4.4928 |
| neural predictive-only | 0.0772 | 0.2740 | 0.5493 | 0.0777 | 3.7169 |
| Python-native MCMC fixed | 0.0740 | 0.2622 | 0.5490 | 0.0695 | 3.3754 |

The predictive-only artifact passed the real held-out gate and slightly
improved log loss and macro AUC relative to the uncalibrated neural artifact,
but Brier score, prevalence MAE, and richness MAE were slightly worse than the
uncalibrated neural artifact. MCMC remains better on Brier, log loss,
prevalence MAE, and richness MAE.

## Coefficient SBC

| variant | coverage | rank mean | rank variance | beta RMSE |
| --- | ---: | ---: | ---: | ---: |
| uncalibrated | 0.5814 | 0.4978 | 0.1559 | 0.4599 |
| external_monotone | 0.9579 | 0.4950 | 0.0697 | 0.4604 |

The promoted coefficient calibration fixes the severe undercoverage on the
shape-matched Whittaker simulations while preserving posterior-mean RMSE. Rank
variance remains below the uniform expectation, so the result should be read as
passing the current coverage/non-degradation gate, not as proof of exact HMSC
posterior equivalence.

## Decision

`external_monotone` passes Whittaker real-data requalification under the
current split coefficient/predictive semantics. This supports real-data
transfer utility for the promoted neural calibration.

It does not close the Python-only HMSC parity question. The next parity-focused
step should compare the Python-native Whittaker fixed model against an
R-created HMSC model exported through the original R+Python HMSC-HPC boundary,
using identical data, formula, split, MCMC settings, posterior summaries,
held-out predictions, and qualitative book checks.

## Qualified Comparator Rerun

Date: 2026-07-19

LUMI job: `20000918`

Remote run root:
`/scratch/project_462000131/anisrahm/hmsc-hpc-runs/neural_whittaker_extmono_qualified_20260719`

This rerun used the promoted `external_monotone` calibration and attached the
passed direct Whittaker R/Python parity metrics:

```text
/scratch/project_462000131/anisrahm/hmsc-hpc-runs/whittaker_r_python_parity_scaled_20260718_082539/whittaker_r_python_parity_metrics.json
```

The MCMC comparator is therefore reported as
`qualified_python_mcmc_fixed`: a Python-native MCMC reference qualified against
the original R+Python HMSC-HPC boundary for the fixed-effect Whittaker
trait/phylogeny scope.

Acceptance:

| gate | result |
| --- | --- |
| coefficient SBC acceptance | pass |
| held-out predictive acceptance | pass |
| combined qualification | pass |
| reference parity qualification | pass |

Held-out metrics:

| model | Brier | log loss | macro AUC | prevalence MAE | richness MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| neural uncalibrated | `0.076894` | `0.274523` | `0.538601` | `0.073989` | `3.426325` |
| neural coefficient-calibrated | `0.081051` | `0.291507` | `0.525263` | `0.094690` | `4.492758` |
| neural predictive-only | `0.077161` | `0.273998` | `0.549285` | `0.077720` | `3.716901` |
| qualified Python MCMC fixed | `0.073933` | `0.261750` | `0.548967` | `0.069339` | `3.369126` |

Coefficient SBC remained calibrated under the shape-matched simulation gate:
overall 95% coverage was `0.957865`, rank mean was `0.494969`, rank variance
was `0.069729`, and Beta mean RMSE was `0.460380`. The reference parity metrics
reported `Beta` correlation `0.999832` and `Gamma` correlation `1.000000`.

Runtime:

| phase | seconds |
| --- | ---: |
| neural training | `447.505` |
| neural real-data inference | `0.157` |
| qualified Python MCMC sampling | `34.109` |

This rerun preserves the earlier interpretation: `external_monotone` qualifies
for Whittaker real-data predictive transfer under split coefficient/predictive
semantics, while the direct parity attachment qualifies the comparator path. It
does not make neural predictive transfer an exact HMSC posterior-equivalence
claim.

## Next Decision

The bounded three-seed real-data sensitivity check completed as LUMI job
`20001710`; see
`docs/neural_hmsc_realdata_sensitivity_2026-07-19.md`. Whittaker passed in all
three seeds with mean coefficient SBC coverage `0.9550`, rank mean `0.4943`,
and rank variance `0.0702`. The promoted neural path remained stable under the
Whittaker qualification gates, but the qualified Python MCMC comparator retained
the proper-score advantage in all three Whittaker seeds: mean Brier ratio
versus MCMC was `1.0386` and mean log-loss ratio was `1.0576`.

The next roadmap step is to resume simulated neural competitor development,
using this real-data result as a frozen qualification constraint rather than
tuning Whittaker-specific gates.
