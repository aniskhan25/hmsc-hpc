# Response-Scale Predictive Mean Production Evaluation

Date: 2026-07-19

LUMI job: `20005059`

Run root:
`/scratch/project_462000131/anisrahm/hmsc-hpc-runs/neural_hmsc_response_mean_production_eval_20005059`

Local aggregate:
`/private/tmp/neural_response_mean_production_eval_20005059`

Purpose: evaluate `probit_response_affine` on the previously qualified
production-shape `external_monotone` baseline from job `19942240`, so the
response-mean decision is not blocked by the underqualified compact local
checkpoint used in earlier experiments.

## Evaluation Shape

- Seeds: `20260716`, `20260717`, `20260718`, `20260719`, `20260720`
- Shape: `40 x 75`
- SBC datasets: `32`
- SBC draws: `256`
- External monotone calibration datasets: `4`
- OOD regimes: `covariate_shift`, `effect_size_shift`, `combined_shift`
- Baseline: qualified `external_monotone`
- Candidate: qualified `external_monotone + probit_response_affine`

The candidate reused each seed's scalar checkpoint and compared against the
frozen production baseline on identical fixed SBC/OOD rows. Wall time was
`205` seconds on `dev-g`.

## Fixed SBC/OOD Summary

| run | seeds | in-domain 95% | rare 95% | mean OOD 95% | worst OOD 95% | effect-size 95% | combined 95% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `external_monotone` | 5 | `0.9442` | `0.9520` | `0.9203` | `0.8214` | `0.8214` | `0.9684` |
| `external_monotone_response` | 5 | `0.9442` | `0.9520` | `0.9203` | `0.8214` | `0.8214` | `0.9684` |

The fixed SBC/OOD metrics are identical, as intended. `probit_response_affine`
only changes `neural_predictive_distribution.h5`; it does not change
coefficient posterior samples, coefficient-posterior calibration, SBC rank
diagnostics, OOD coefficient gates, or rare-validation semantics.

The fixed-evaluation comparison marked the response candidate as accepted in
all five seeds under the zero-delta production reuse comparison. Baseline rows
are not themselves candidates in that comparison, so their acceptance flag is
not used as a baseline qualification signal.

## Predictive Proper Scores

| run | Brier | Brier ratio | log-loss | log-loss ratio | predictive RMSE | RMSE ratio | prevalence MAE | richness MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `external_monotone` | `0.147620` | `1.0000` | `0.446577` | `1.0000` | `0.384179` | `1.0000` | `0.043553` | `3.023169` |
| `external_monotone_response` | `0.147153` | `0.9968` | `0.443869` | `0.9938` | `0.383563` | `0.9984` | `0.042621` | `2.982169` |

Per-seed predictive ratios for `external_monotone_response`:

| seed | Brier ratio | log-loss ratio | RMSE ratio |
| ---: | ---: | ---: | ---: |
| `20260716` | `0.9937` | `0.9906` | `0.9968` |
| `20260717` | `0.9966` | `0.9909` | `0.9983` |
| `20260718` | `1.0028` | `1.0022` | `1.0014` |
| `20260719` | `0.9978` | `0.9960` | `0.9989` |
| `20260720` | `0.9929` | `0.9894` | `0.9964` |

## Decision

`probit_response_affine` qualifies as a viable predictive-only competitor on
the production-shape simulated evaluation. It gives small mean improvements in
Brier, log-loss, predictive RMSE, prevalence MAE, and richness MAE while
preserving the frozen coefficient/SBC/OOD/rare gates.

Do not promote it as the default yet. The gains are modest and one seed
slightly worsened proper scores, so the next decision point should be real-data
transfer: run Whittaker and Big Spatial with `external_monotone_response`
against the promoted `external_monotone` path, keeping Python-only HMSC parity
metrics attached and keeping the response layer explicitly labelled
predictive-only.
