# Predictive Mean Calibration Compact Comparison

Date: 2026-07-19

Local run root:
`/private/tmp/neural_mean_fixed_eval_20260719`

Candidates:

- `external_monotone`
- `external_monotone_affine`: `external_monotone` plus
  `--predictive-mean-calibration affine_shrinkage`

Distribution: `probit`

Purpose: compact local fixed-evaluation comparison before any five-seed LUMI
run. This check used a shared seed schedule and reused the baseline neural
checkpoint for the affine run so the only intended candidate difference was
the predictive-only mean calibrator.

## Commands

Baseline:

```text
python3 examples/run_neural_hmsc_benchmark.py --output /private/tmp/neural_mean_fixed_eval_20260719/external_monotone --suite probit --n-sites 24 --n-species 3 --train-datasets 12 --calibration-datasets 6 --rare-calibration-datasets 3 --rare-validation-datasets 3 --epochs 8 --batch-size 4 --seed 20260724 --model-seed 20260724 --coefficient-calibration external_monotone --external-monotone-datasets 2 --external-monotone-min-ood-gain 0.0 --external-monotone-min-combined-gain 0.0 --sbc-datasets 8 --sbc-draws 64 --ood-regimes covariate_shift effect_size_shift combined_shift --run-mcmc-reference --mcmc-samples 40 --mcmc-transient 20 --mcmc-chains 1 --neural-chains 1 --neural-draws 64
```

Affine candidate:

```text
python3 examples/run_neural_hmsc_benchmark.py --output /private/tmp/neural_mean_fixed_eval_20260719/external_monotone_affine --suite probit --n-sites 24 --n-species 3 --train-datasets 12 --calibration-datasets 6 --rare-calibration-datasets 3 --rare-validation-datasets 3 --epochs 8 --batch-size 4 --seed 20260724 --model-seed 20260724 --checkpoint /private/tmp/neural_mean_fixed_eval_20260719/external_monotone/probit/neural_checkpoint --coefficient-calibration external_monotone --external-monotone-datasets 2 --external-monotone-min-ood-gain 0.0 --external-monotone-min-combined-gain 0.0 --predictive-mean-calibration affine_shrinkage --predictive-mean-calibration-validation-datasets 4 --predictive-mean-calibration-min-improvement 0.0001 --sbc-datasets 8 --sbc-draws 64 --ood-regimes covariate_shift effect_size_shift combined_shift --run-mcmc-reference --mcmc-samples 40 --mcmc-transient 20 --mcmc-chains 1 --neural-chains 1 --neural-draws 64
```

Fixed-evaluation comparison:

```text
python3 examples/compare_neural_hmsc_fixed_evaluation.py --run external_monotone=/private/tmp/neural_mean_fixed_eval_20260719/external_monotone --run external_monotone_affine=/private/tmp/neural_mean_fixed_eval_20260719/external_monotone_affine --baseline external_monotone --output /private/tmp/neural_mean_fixed_eval_20260719/comparison --min-mean-ood-delta 0.0 --min-combined-delta 0.0
```

## Mean-Calibrator Selection

The affine calibrator did not select a non-identity correction on the
independent validation pool.

| field | value |
| --- | ---: |
| selected | `false` |
| slope | `1.0000` |
| intercept | `0.0000` |
| calibration RMSE, uncalibrated | `0.3063` |
| calibration RMSE, fitted affine | `0.2943` |
| validation RMSE, uncalibrated | `0.3276` |
| validation RMSE, selected | `0.3276` |
| validation RMSE ratio | `1.0000` |

The training/calibration pool supported an affine correction, but the
independent validation pool did not. The fallback-to-identity behavior worked as
intended.

## Predictive Rows

| run | variant | Beta RMSE truth | predictive RMSE | MCMC predictive RMSE |
| --- | --- | ---: | ---: | ---: |
| `external_monotone` | uncalibrated | `0.1560` | `0.3493` | `0.3558` |
| `external_monotone` | calibrated | `0.1536` | `0.3529` | `0.3558` |
| `external_monotone_affine` | uncalibrated | `0.1560` | `0.3493` | `0.3542` |
| `external_monotone_affine` | calibrated | `0.1536` | `0.3529` | `0.3542` |

Because the affine calibrator selected identity, the neural predictive RMSE was
unchanged. The MCMC columns differ slightly between the two local runs because
each compact run produced its own tiny stochastic MCMC reference; they should
not be interpreted as a neural candidate effect.

## Fixed SBC/OOD Rows

| run | in-domain 95% | rare 95% | mean OOD 95% | worst OOD 95% | effect-size 95% | combined 95% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `external_monotone` | `0.8750` | `1.0000` | `0.7731` | `0.7222` | `0.7222` | `0.7361` |
| `external_monotone_affine` | `0.8750` | `1.0000` | `0.7731` | `0.7222` | `0.7222` | `0.7361` |

The coefficient-posterior SBC/OOD rows were unchanged, as expected, because
the predictive mean calibrator is only applied to
`neural_predictive_distribution.h5`.

The compact fixed-evaluation gate did not pass because in-domain coverage was
`0.8750`, below the frozen `0.9000` gate. This is a compact local run with only
eight SBC datasets, but it still means the candidate is not eligible for a
five-seed LUMI comparison.

## Decision

Do not submit the affine predictive-mean competitor to LUMI.

The failure mode is useful: a global coefficient-RMSE affine correction is too
weak and too indirect for the real-data proper-score gap. The next candidate
should fit and gate predictive mean movement on response-scale proper scores
directly, such as probit Brier/log-loss on independent simulated holdouts, while
leaving coefficient posterior calibration and SBC semantics frozen.
