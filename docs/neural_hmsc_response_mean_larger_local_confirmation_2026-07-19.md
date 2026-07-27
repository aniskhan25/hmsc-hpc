# Response-Scale Predictive Mean Larger Local Confirmation

Date: 2026-07-19

Run root:
`/private/tmp/neural_response_mean_larger_eval_20260719`

Baseline:
`/private/tmp/neural_response_mean_larger_eval_20260719/external_monotone`

Candidate:
`/private/tmp/neural_response_mean_larger_eval_20260719/external_monotone_response`

Comparison:
`/private/tmp/neural_response_mean_larger_eval_20260719/comparison`

Purpose: larger local fixed-evaluation confirmation for
`probit_response_affine` before any five-seed LUMI run. This repeated the
compact comparison with the same frozen checkpoint and seed schedule, but
increased SBC/OOD evaluation from `8 x 64` to `24 x 128`.

## Candidate Selection

The response-scale selector again accepted the same non-identity
predictive-only mean correction.

| field | value |
| --- | ---: |
| selected | `true` |
| slope | `1.2500` |
| intercept | `0.0250` |
| validation Brier ratio | `0.9911` |
| validation log-loss ratio | `0.9877` |

The metadata remained predictive-only under `predictive_mean_calibration`; the
coefficient posterior and SBC/OOD semantics were unchanged.

## Predictive Rows

| run | variant | Beta RMSE truth | predictive RMSE | MCMC predictive RMSE |
| --- | --- | ---: | ---: | ---: |
| `external_monotone` | uncalibrated | `0.1560` | `0.3493` | `0.3508` |
| `external_monotone` | calibrated | `0.1536` | `0.3529` | `0.3508` |
| `external_monotone_response` | uncalibrated | `0.1560` | `0.3493` | `0.3495` |
| `external_monotone_response` | calibrated | `0.1536` | `0.3505` | `0.3495` |

The calibrated predictive RMSE again improved from `0.3529` to `0.3505`, a
ratio of `0.9934`.

## Fixed SBC/OOD Rows

| run | in-domain 95% | rare 95% | mean OOD 95% | worst OOD 95% | effect-size 95% | combined 95% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `external_monotone` | `0.8519` | `1.0000` | `0.7701` | `0.6898` | `0.6898` | `0.7130` |
| `external_monotone_response` | `0.8519` | `1.0000` | `0.7701` | `0.6898` | `0.6898` | `0.7130` |

The fixed SBC/OOD rows were identical between baseline and candidate. This is
expected because `probit_response_affine` only changes
`neural_predictive_distribution.h5`; it does not change the coefficient
posterior used by SBC.

## Decision

Do not submit `probit_response_affine` to five-seed LUMI yet.

The larger local run confirms that `probit_response_affine` is a promising
predictive-only mean competitor: it selected non-identity movement and
improved predictive RMSE without touching coefficient calibration. However, the
fixed coefficient-calibration gate did not pass for either baseline or
candidate under this frozen compact checkpoint/evaluation setup. The in-domain
coverage issue is therefore not caused by the response-mean layer, but the
candidate cannot be promoted while the frozen coefficient baseline fails the
gate.

The next step should be to evaluate `probit_response_affine` on a previously
qualified external-monotone fixed-evaluation/prod-shape baseline, or rerun the
larger local confirmation with the promoted production-shape settings that
already qualified `external_monotone`, so the response-mean decision is not
blocked by an underqualified compact checkpoint.
