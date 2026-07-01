# Neural HMSC Stratified SBC Scalar Baseline

Date: 2026-07-01

## Scope

This run establishes the multi-seed global coefficient-scale baseline for
Milestone 12. It uses commit `e7fec41` and keeps coefficient-posterior
calibration separate from predictive-only calibration.

Each seed trained a probit neural posterior independently and fitted one
positive scalar multiplier on its own independent calibration simulations.
SBC was then evaluated on independent in-domain and OOD simulations.

Configuration:

- seeds: `20260626`, `20260627`, `20260628`, `20260629`, `20260630`
- shape: 40 sites, 75 species, 3 fixed-effect coefficients
- 512 training and 128 calibration datasets per seed
- 128 SBC datasets, 512 posterior draws, and 10 rank bins per domain
- 120 epochs, batch size 16, 4 neural chains, and 500 inference draws
- OOD regimes: covariate shift, effect-size shift, and combined shift
- MCMC disabled because this run compares conditional calibration against the
  frozen neural scalar baseline

LUMI jobs `19654959`, `19654960`, `19655041`, `19655042`, and `19655127` all
completed with exit code 0. Artifacts are under
`/scratch/project_462000131/anisrahm/hmsc-hpc-runs/neural_hmsc_stratified_sbc_e7fec41_seed_<seed>/benchmark`.

## Scalar Fit

| Seed | Coefficient scale multiplier |
|---:|---:|
| 20260626 | 2.4605 |
| 20260627 | 2.3435 |
| 20260628 | 2.5246 |
| 20260629 | 2.3133 |
| 20260630 | 2.3292 |

The mean multiplier was `2.3942 +/- 0.0932`, with range `2.3133-2.5246`.

## Overall Results

Values are five-seed mean `+/-` sample standard deviation. Rank errors are
absolute deviations from the uniform-rank expectations.

| Domain | Variant | Coverage 95 | Rank mean error | Rank variance error | Beta mean RMSE |
|---|---|---:|---:|---:|---:|
| In-domain | uncalibrated | 0.7408 +/- 0.0056 | 0.0064 +/- 0.0017 | 0.0402 +/- 0.0013 | 0.5886 +/- 0.0032 |
| In-domain | scalar | 0.9420 +/- 0.0034 | 0.0039 +/- 0.0015 | 0.0243 +/- 0.0015 | 0.5903 +/- 0.0033 |
| OOD covariate | uncalibrated | 0.3942 +/- 0.0140 | 0.0258 +/- 0.0074 | 0.1006 +/- 0.0025 | 0.9587 +/- 0.0195 |
| OOD covariate | scalar | 0.5987 +/- 0.0161 | 0.0219 +/- 0.0072 | 0.0562 +/- 0.0031 | 0.9600 +/- 0.0195 |
| OOD effect-size | uncalibrated | 0.4664 +/- 0.0079 | 0.0029 +/- 0.0013 | 0.0935 +/- 0.0011 | 1.2518 +/- 0.0044 |
| OOD effect-size | scalar | 0.7597 +/- 0.0069 | 0.0021 +/- 0.0008 | 0.0330 +/- 0.0018 | 1.2526 +/- 0.0045 |
| OOD combined | uncalibrated | 0.2966 +/- 0.0121 | 0.0209 +/- 0.0057 | 0.1188 +/- 0.0019 | 1.4951 +/- 0.0144 |
| OOD combined | scalar | 0.4927 +/- 0.0170 | 0.0187 +/- 0.0056 | 0.0802 +/- 0.0034 | 1.4961 +/- 0.0145 |

Scalar calibration materially improves coverage and rank variance without a
material posterior-mean RMSE change. It does not satisfy the in-domain rank
variance gate and remains badly underdispersed under every OOD regime.

## In-Domain Strata

The final column is the number of seeds passing the coverage (`>= 0.90`), rank
mean (`<= 0.025`), and rank variance (`<= 0.015`) thresholds.

| Stratum | Coverage 95 | Rank mean error | Rank variance error | Seed passes: coverage / mean / variance |
|---|---:|---:|---:|---:|
| Overall | 0.9420 +/- 0.0034 | 0.0039 +/- 0.0015 | 0.0243 +/- 0.0015 | 5/5 / 5/5 / 0/5 |
| Prevalence: rare | 0.8970 +/- 0.0066 | 0.1070 +/- 0.0036 | 0.0149 +/- 0.0020 | 1/5 / 0/5 / 2/5 |
| Prevalence: intermediate | 0.9352 +/- 0.0049 | 0.0680 +/- 0.0021 | 0.0244 +/- 0.0022 | 5/5 / 0/5 / 0/5 |
| Prevalence: common | 0.9450 +/- 0.0035 | 0.0134 +/- 0.0020 | 0.0261 +/- 0.0014 | 5/5 / 5/5 / 0/5 |
| Coefficient: Intercept | 0.9419 +/- 0.0027 | 0.0144 +/- 0.0042 | 0.0232 +/- 0.0010 | 5/5 / 5/5 / 0/5 |
| Coefficient: x1 | 0.9426 +/- 0.0036 | 0.0019 +/- 0.0009 | 0.0248 +/- 0.0017 | 5/5 / 5/5 / 0/5 |
| Coefficient: x2 | 0.9416 +/- 0.0045 | 0.0042 +/- 0.0020 | 0.0250 +/- 0.0018 | 5/5 / 5/5 / 0/5 |
| Design information: low | 0.9459 +/- 0.0020 | 0.0026 +/- 0.0021 | 0.0258 +/- 0.0017 | 5/5 / 5/5 / 0/5 |
| Design information: intermediate | 0.9435 +/- 0.0051 | 0.0049 +/- 0.0012 | 0.0248 +/- 0.0018 | 5/5 / 5/5 / 0/5 |
| Design information: high | 0.9367 +/- 0.0051 | 0.0049 +/- 0.0030 | 0.0223 +/- 0.0016 | 5/5 / 5/5 / 0/5 |

The scalar baseline therefore fails Milestone 12. Aggregate coverage hides two
different defects: strong conditional rank bias for rare and intermediate
species, and nonuniform rank variance across every coefficient and information
stratum.

## Conditional Calibrator Decision

The first conditional implementation will use a structured additive residual
scale head:

```text
log(scale_ij) = log(global_scale)
              + f_prevalence(logit(prevalence_j))
              + coefficient_effect[k]
              + f_information(log(expected_information_ij))
              + f_raw_scale(log(raw_posterior_sd_ij))
              + prevalence_by_coefficient[k]
```

The output remains positive through exponentiation and is regularized toward
the scalar baseline. It changes posterior scale only, leaves posterior means
unchanged, and applies coefficient-wise scaling as `D Sigma D` for
full-covariance posterior families. Training uses simulated calibration truth;
held-out SBC, not either ecological dataset or MCMC output, selects the model.

Prevalence is the primary feature because its rank-mean failures are large and
repeat across every seed. Raw scale and expected information allow continuous
adaptation within prevalence groups. Coefficient identity and its prevalence
interaction provide limited structure for intercept-specific behavior without
creating a high-capacity calibrator.

The next implementation step is to add the conditional-calibration module and
compare it against these exact five scalar runs. Absolute in-domain gates and
improvement over the scalar under each OOD regime remain mandatory.
