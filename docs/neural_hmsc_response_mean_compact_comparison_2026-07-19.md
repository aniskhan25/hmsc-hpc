# Response-Scale Predictive Mean Compact Comparison

Date: 2026-07-19

Baseline run root:
`/private/tmp/neural_mean_fixed_eval_20260719/external_monotone`

Candidate run root:
`/private/tmp/neural_response_mean_fixed_eval_20260719/external_monotone_response`

Comparison root:
`/private/tmp/neural_response_mean_fixed_eval_20260719/comparison`

Candidates:

- `external_monotone`
- `external_monotone_response`: `external_monotone` plus
  `--predictive-mean-calibration probit_response_affine`

Distribution: `probit`

Purpose: compact local fixed-evaluation comparison before any five-seed LUMI
run. The candidate reused the baseline neural checkpoint, seed schedule,
calibration settings, rare settings, and fixed SBC/OOD settings.

## Candidate Command

```text
python3 examples/run_neural_hmsc_benchmark.py --output /private/tmp/neural_response_mean_fixed_eval_20260719/external_monotone_response --suite probit --n-sites 24 --n-species 3 --train-datasets 12 --calibration-datasets 6 --rare-calibration-datasets 3 --rare-validation-datasets 3 --epochs 8 --batch-size 4 --seed 20260724 --model-seed 20260724 --checkpoint /private/tmp/neural_mean_fixed_eval_20260719/external_monotone/probit/neural_checkpoint --coefficient-calibration external_monotone --external-monotone-datasets 2 --external-monotone-min-ood-gain 0.0 --external-monotone-min-combined-gain 0.0 --predictive-mean-calibration probit_response_affine --predictive-mean-calibration-validation-datasets 4 --predictive-mean-calibration-min-improvement 0.0001 --sbc-datasets 8 --sbc-draws 64 --ood-regimes covariate_shift effect_size_shift combined_shift --run-mcmc-reference --mcmc-samples 40 --mcmc-transient 20 --mcmc-chains 1 --neural-chains 1 --neural-draws 64
```

Fixed-evaluation comparison:

```text
python3 examples/compare_neural_hmsc_fixed_evaluation.py --run external_monotone=/private/tmp/neural_mean_fixed_eval_20260719/external_monotone --run external_monotone_response=/private/tmp/neural_response_mean_fixed_eval_20260719/external_monotone_response --baseline external_monotone --output /private/tmp/neural_response_mean_fixed_eval_20260719/comparison --min-mean-ood-delta 0.0 --min-combined-delta 0.0
```

## Response-Scale Selection

The response-scale mean selector accepted a non-identity predictive-only
correction on the independent simulated response-validation pool.

| field | value |
| --- | ---: |
| selected | `true` |
| slope | `1.2500` |
| intercept | `0.0250` |
| calibration Brier ratio | `0.9872` |
| calibration log-loss ratio | `0.9767` |
| validation Brier ratio | `0.9911` |
| validation log-loss ratio | `0.9877` |

The metadata was written under `predictive_mean_calibration` with
`artifact_role = predictive_only_mean`; coefficient-posterior calibration and
SBC semantics were not changed.

## Predictive Rows

| run | variant | Beta RMSE truth | predictive RMSE | MCMC predictive RMSE |
| --- | --- | ---: | ---: | ---: |
| `external_monotone` | uncalibrated | `0.1560` | `0.3493` | `0.3558` |
| `external_monotone` | calibrated | `0.1536` | `0.3529` | `0.3558` |
| `external_monotone_response` | uncalibrated | `0.1560` | `0.3493` | `0.3527` |
| `external_monotone_response` | calibrated | `0.1536` | `0.3505` | `0.3527` |

The calibrated predictive RMSE improved from `0.3529` to `0.3505`, a ratio of
`0.9934`. The MCMC columns differ slightly because these compact local runs use
tiny stochastic MCMC references; they should not be interpreted as a neural
candidate effect.

## Fixed SBC/OOD Rows

| run | in-domain 95% | rare 95% | mean OOD 95% | worst OOD 95% | effect-size 95% | combined 95% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `external_monotone` | `0.8750` | `1.0000` | `0.7731` | `0.7222` | `0.7222` | `0.7361` |
| `external_monotone_response` | `0.8750` | `1.0000` | `0.7731` | `0.7222` | `0.7222` | `0.7361` |

The fixed SBC/OOD rows were unchanged, as expected, because the response-scale
mean correction is only applied to `neural_predictive_distribution.h5`.

The fixed-evaluation acceptance flag did not pass because the compact
in-domain coverage was `0.8750`, below the frozen `0.9000` gate. This is the
same failure as the baseline and is likely sensitive to the very small
`sbc_datasets = 8` compact check, but it still means this result is not enough
to submit a five-seed LUMI comparison.

## Decision

Do not submit the response-scale mean competitor to five-seed LUMI yet.

The candidate is more promising than the previous coefficient-RMSE affine
selector because it selected non-identity movement and improved compact
predictive RMSE without changing SBC/OOD rows. The next step should be a
larger local fixed-evaluation confirmation, using the same shared checkpoint
and response-scale selector but enough SBC/OOD evaluation rows to determine
whether the in-domain gate failure was compact-sample noise.
