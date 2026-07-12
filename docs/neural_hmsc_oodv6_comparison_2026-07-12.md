# Neural HMSC OOD-aware IRLS v6 paired LUMI comparison

Date: 2026-07-12

Branch: `feature/neural-hmsc`

Candidate commit: `f06e7b9 Add OOD-aware IRLS uncertainty scaling`

## Objective

Evaluate the version 6 conditional coefficient calibrator, which keeps the
IRLS-Laplace probit anchor from version 5 and adds bounded OOD uncertainty
inflation from support excess.

The comparison is paired by seed against frozen scalar, version 4, and version 5
IRLS references.

## Candidate configuration

- Distribution: probit
- Seeds: `20260626`, `20260627`, `20260628`, `20260629`, `20260630`
- Sites/species: `40` / `75`
- Training/calibration/SBC datasets: `512` / `128` / `128`
- SBC draws/bins: `512` / `10`
- OOD regimes: covariate shift, effect-size shift, combined shift
- Neural chains/draws: `4` / `500`
- Coefficient calibration: `conditional`
- Conditional method: `conditional_rank_aware_anchor_scale`
- Conditional semantics version: `6`
- OOD uncertainty strength/max multiplier: `0.75` / `4.0`
- Prevalence weights: rare/intermediate/common `4.0` / `2.0` / `1.0`
- Rank penalty: `0.02`
- Support fallback quantile/strength: `0.99` / `2.0`
- Probit anchor: `irls_laplace`
- IRLS iterations/prior precision/eta clip: `8` / `1.0` / `6.0`

All five candidate checkpoints reported checkpoint version `0.3` and
`probit_anchor=irls_laplace`. Each candidate diagnostics file contained 80 SBC
rows. All calibration records reported `semantics_version=6` with OOD
uncertainty transform `support_excess_exp`.

## LUMI jobs

| Seed | Job | Partition | State | Elapsed |
| --- | ---: | --- | --- | --- |
| 20260626 | 19829618 | standard-g | COMPLETED | 00:11:32 |
| 20260627 | 19829619 | standard-g | COMPLETED | 00:11:32 |
| 20260628 | 19829624 | standard-g | COMPLETED | 00:11:31 |
| 20260629 | 19829625 | standard-g | COMPLETED | 00:11:35 |
| 20260630 | 19829627 | standard-g | COMPLETED | 00:11:34 |

The first two attempts, jobs `19829616` and `19829617`, were cancelled because
they were submitted to maintenance-blocked `dev-g`. All final jobs ran on
`standard-g`.

## Overall SBC comparison

Metrics are five-seed mean +/- standard error. Coverage is coefficient posterior
95% interval coverage, not predictive calibration. Rank-mean and rank-variance
columns are absolute errors from the SBC uniform target.

| Domain | Model | Coverage 95 | Rank mean error | Rank variance error | Beta RMSE | Seed passes: coverage / mean / variance |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| In-domain | scalar | 0.9420 +/- 0.0015 | 0.0039 +/- 0.0007 | 0.0243 +/- 0.0007 | 0.5903 +/- 0.0015 | 5/5 / 5/5 / 0/5 |
| In-domain | version 4 | 0.9439 +/- 0.0007 | 0.0014 +/- 0.0003 | 0.0077 +/- 0.0002 | 0.5889 +/- 0.0014 | 5/5 / 5/5 / 5/5 |
| In-domain | version 5 IRLS | 0.9411 +/- 0.0009 | 0.0033 +/- 0.0009 | 0.0011 +/- 0.0002 | 0.3247 +/- 0.0011 | 5/5 / 5/5 / 5/5 |
| In-domain | version 6 OOD | 0.9416 +/- 0.0008 | 0.0033 +/- 0.0010 | 0.0011 +/- 0.0002 | 0.3247 +/- 0.0011 | 5/5 / 5/5 / 5/5 |
| OOD covariate | scalar | 0.5987 +/- 0.0072 | 0.0219 +/- 0.0032 | 0.0562 +/- 0.0014 | 0.9600 +/- 0.0087 | 0/5 / 3/5 / 0/5 |
| OOD covariate | version 4 | 0.5963 +/- 0.0067 | 0.0229 +/- 0.0031 | 0.0622 +/- 0.0012 | 0.9595 +/- 0.0087 | 0/5 / 3/5 / 0/5 |
| OOD covariate | version 5 IRLS | 0.4649 +/- 0.0499 | 0.0109 +/- 0.0028 | 0.0912 +/- 0.0084 | 0.5186 +/- 0.0153 | 0/5 / 5/5 / 0/5 |
| OOD covariate | version 6 OOD | 0.6384 +/- 0.0483 | 0.0091 +/- 0.0025 | 0.0535 +/- 0.0098 | 0.5206 +/- 0.0152 | 0/5 / 5/5 / 0/5 |
| OOD effect-size | scalar | 0.7597 +/- 0.0031 | 0.0021 +/- 0.0004 | 0.0330 +/- 0.0008 | 1.2526 +/- 0.0020 | 0/5 / 5/5 / 0/5 |
| OOD effect-size | version 4 | 0.6882 +/- 0.0009 | 0.0040 +/- 0.0005 | 0.0570 +/- 0.0001 | 1.2519 +/- 0.0020 | 0/5 / 5/5 / 0/5 |
| OOD effect-size | version 5 IRLS | 0.7280 +/- 0.0026 | 0.0016 +/- 0.0005 | 0.0476 +/- 0.0005 | 0.6783 +/- 0.0005 | 0/5 / 5/5 / 0/5 |
| OOD effect-size | version 6 OOD | 0.7468 +/- 0.0020 | 0.0018 +/- 0.0006 | 0.0434 +/- 0.0004 | 0.6784 +/- 0.0005 | 0/5 / 5/5 / 0/5 |
| OOD combined | scalar | 0.4927 +/- 0.0076 | 0.0187 +/- 0.0025 | 0.0802 +/- 0.0015 | 1.4961 +/- 0.0065 | 0/5 / 5/5 / 0/5 |
| OOD combined | version 4 | 0.4726 +/- 0.0065 | 0.0191 +/- 0.0024 | 0.0869 +/- 0.0012 | 1.4957 +/- 0.0064 | 0/5 / 5/5 / 0/5 |
| OOD combined | version 5 IRLS | 0.3226 +/- 0.0342 | 0.0054 +/- 0.0012 | 0.1157 +/- 0.0056 | 0.9304 +/- 0.0089 | 0/5 / 5/5 / 0/5 |
| OOD combined | version 6 OOD | 0.4854 +/- 0.0393 | 0.0054 +/- 0.0011 | 0.0837 +/- 0.0074 | 0.9315 +/- 0.0088 | 0/5 / 5/5 / 0/5 |

## OOD inflation diagnostics

| Domain | Inflation mean | Inflation max | Inflated fraction | Support trust mean | Fallback fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| In-domain | 1.0170 +/- 0.0014 | 4.0000 +/- 0.0000 | 0.0336 +/- 0.0015 | 0.9889 +/- 0.0008 | 0.0113 +/- 0.0011 |
| OOD covariate | 2.7460 +/- 0.1031 | 4.0000 +/- 0.0000 | 0.7467 +/- 0.0256 | 0.3523 +/- 0.0310 | 0.6575 +/- 0.0304 |
| OOD effect-size | 1.1490 +/- 0.0055 | 4.0000 +/- 0.0000 | 0.1497 +/- 0.0031 | 0.9221 +/- 0.0023 | 0.0804 +/- 0.0026 |
| OOD combined | 2.7942 +/- 0.1098 | 4.0000 +/- 0.0000 | 0.7777 +/- 0.0243 | 0.3296 +/- 0.0321 | 0.6824 +/- 0.0315 |

The support detector is behaving directionally: covariate and combined shifts
trigger high fallback and inflation, while in-domain data mostly retain the
learned conditional head. The max multiplier saturates in every domain,
including a small in-domain tail, so the current cap is active and should be
treated as a tuning parameter rather than a settled default.

## Paired deltas

Positive coverage deltas are better when coverage is below target. Negative
rank-error and RMSE deltas are better.

| Domain | Delta | Coverage 95 | Rank mean error | Rank variance error | Beta RMSE |
| --- | --- | ---: | ---: | ---: | ---: |
| In-domain | v6 - scalar | -0.0004 +/- 0.0021 | -0.0006 +/- 0.0011 | -0.0232 +/- 0.0008 | -0.2656 +/- 0.0021 |
| In-domain | v6 - version 4 | -0.0022 +/- 0.0012 | 0.0019 +/- 0.0011 | -0.0066 +/- 0.0004 | -0.2641 +/- 0.0021 |
| In-domain | v6 - version 5 IRLS | 0.0005 +/- 0.0002 | -0.0000 +/- 0.0000 | 0.0000 +/- 0.0001 | 0.0001 +/- 0.0000 |
| OOD covariate | v6 - scalar | 0.0397 +/- 0.0481 | -0.0128 +/- 0.0033 | -0.0027 +/- 0.0098 | -0.4394 +/- 0.0208 |
| OOD covariate | v6 - version 4 | 0.0421 +/- 0.0490 | -0.0138 +/- 0.0031 | -0.0087 +/- 0.0100 | -0.4389 +/- 0.0208 |
| OOD covariate | v6 - version 5 IRLS | 0.1735 +/- 0.0034 | -0.0017 +/- 0.0007 | -0.0377 +/- 0.0015 | 0.0019 +/- 0.0003 |
| OOD effect-size | v6 - scalar | -0.0128 +/- 0.0039 | -0.0003 +/- 0.0006 | 0.0104 +/- 0.0008 | -0.5741 +/- 0.0024 |
| OOD effect-size | v6 - version 4 | 0.0587 +/- 0.0018 | -0.0022 +/- 0.0009 | -0.0136 +/- 0.0004 | -0.5735 +/- 0.0023 |
| OOD effect-size | v6 - version 5 IRLS | 0.0188 +/- 0.0009 | 0.0002 +/- 0.0002 | -0.0042 +/- 0.0002 | 0.0002 +/- 0.0001 |
| OOD combined | v6 - scalar | -0.0073 +/- 0.0398 | -0.0133 +/- 0.0028 | 0.0035 +/- 0.0074 | -0.5646 +/- 0.0130 |
| OOD combined | v6 - version 4 | 0.0128 +/- 0.0399 | -0.0137 +/- 0.0027 | -0.0032 +/- 0.0075 | -0.5643 +/- 0.0129 |
| OOD combined | v6 - version 5 IRLS | 0.1628 +/- 0.0052 | -0.0001 +/- 0.0004 | -0.0320 +/- 0.0018 | 0.0011 +/- 0.0002 |

## In-domain stratified gates for version 6 OOD

| Stratum | Coverage 95 | Rank mean error | Rank variance error | Seed passes: coverage / mean / variance |
| --- | ---: | ---: | ---: | --- |
| Coefficient: Intercept | 0.9438 +/- 0.0010 | 0.0018 +/- 0.0008 | 0.0011 +/- 0.0002 | 5/5 / 5/5 / 5/5 |
| Coefficient: x1 | 0.9417 +/- 0.0005 | 0.0015 +/- 0.0006 | 0.0010 +/- 0.0002 | 5/5 / 5/5 / 5/5 |
| Coefficient: x2 | 0.9394 +/- 0.0013 | 0.0091 +/- 0.0020 | 0.0014 +/- 0.0002 | 5/5 / 5/5 / 5/5 |
| Design information: high | 0.9217 +/- 0.0007 | 0.0028 +/- 0.0007 | 0.0055 +/- 0.0005 | 5/5 / 5/5 / 5/5 |
| Design information: intermediate | 0.9602 +/- 0.0008 | 0.0042 +/- 0.0013 | 0.0078 +/- 0.0003 | 5/5 / 5/5 / 5/5 |
| Design information: low | 0.9429 +/- 0.0012 | 0.0042 +/- 0.0010 | 0.0010 +/- 0.0003 | 5/5 / 5/5 / 5/5 |
| Prevalence: common | 0.9421 +/- 0.0008 | 0.0046 +/- 0.0008 | 0.0015 +/- 0.0003 | 5/5 / 5/5 / 5/5 |
| Prevalence: intermediate | 0.9442 +/- 0.0011 | 0.0041 +/- 0.0010 | 0.0006 +/- 0.0002 | 5/5 / 5/5 / 5/5 |
| Prevalence: rare | 0.9112 +/- 0.0039 | 0.0062 +/- 0.0007 | 0.0050 +/- 0.0017 | 5/5 / 5/5 / 5/5 |

## Decision

Version 6 OOD is not qualified as the next default calibration path.

It is a clear improvement over version 5 IRLS under OOD SBC. Covariate-shift
coverage rises from `0.4649` to `0.6384`, combined-shift coverage rises from
`0.3226` to `0.4854`, and rank-variance errors drop materially under all OOD
regimes. In-domain behavior is preserved, including the rare-prevalence stratum,
which improves from 4/5 coverage passes in version 5 to 5/5 in version 6.

The blocker remains the absolute OOD coverage gate. Every OOD regime still has
0/5 coverage passes, and effect-size coverage remains slightly below the scalar
baseline despite large RMSE gains. The fixed support-excess multiplier therefore
helps but is not sufficient.

## Recommended next step

Run a small LUMI tuning sweep rather than changing the posterior mean anchor
again. The sweep should vary OOD uncertainty strength and cap, for example:

- strength `1.0`, max multiplier `8.0`
- strength `1.5`, max multiplier `8.0`
- strength `1.5`, max multiplier `12.0`

The target is to improve OOD coverage while preserving the version 6 in-domain
gates and the IRLS posterior-mean RMSE advantage. If the sweep still cannot
approach OOD coverage, the next implementation should fit an explicit
OOD-calibration objective on held-out OOD simulations instead of using a fixed
support-excess rule.
