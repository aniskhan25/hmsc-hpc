# Neural HMSC IRLS v5 paired LUMI comparison

Date: 2026-07-10

Branch: `feature/neural-hmsc`

Candidate commit: `5d1a8b8 Add probit IRLS Laplace anchor`

## Objective

Run a paired five-seed LUMI retraining comparison for the version 5 conditional
calibrator with the IRLS-Laplace probit anchor, against the frozen scalar and
version 4 legacy references.

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
- Prevalence weights: rare/intermediate/common `4.0` / `2.0` / `1.0`
- Rank penalty: `0.02`
- Support fallback quantile/strength: `0.99` / `2.0`
- Probit anchor: `irls_laplace`
- IRLS iterations/prior precision/eta clip: `8` / `1.0` / `6.0`

All five candidate checkpoints reported checkpoint version `0.3` and
`probit_anchor=irls_laplace`. Each candidate diagnostics file contained 80 SBC
rows.

## LUMI jobs

| Seed | Job | Partition | State | Elapsed |
| --- | ---: | --- | --- | --- |
| 20260626 | 19812408 | dev-g | COMPLETED | 00:11:42 |
| 20260627 | 19812409 | dev-g | COMPLETED | 00:11:28 |
| 20260628 | 19814814 | dev-g | COMPLETED | 00:12:34 |
| 20260629 | 19814815 | dev-g | COMPLETED | 00:12:18 |
| 20260630 | 19822344 | standard-g | COMPLETED | 00:11:43 |

The first final-seed submission, job `19822341`, was cancelled because `dev-g`
was unavailable for maintenance. It was resubmitted unchanged on `standard-g` as
job `19822344`.

## Overall SBC comparison

Metrics are five-seed mean +/- standard error. Coverage is the coefficient
posterior 95% interval coverage, not predictive calibration. Rank-mean and
rank-variance columns are absolute errors from the SBC uniform target.

| Domain | Model | Coverage 95 | Rank mean error | Rank variance error | Beta RMSE | Seed passes: coverage / mean / variance |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| In-domain | scalar | 0.9420 +/- 0.0015 | 0.0039 +/- 0.0007 | 0.0243 +/- 0.0007 | 0.5903 +/- 0.0015 | 5/5 / 5/5 / 0/5 |
| In-domain | version 4 | 0.9439 +/- 0.0007 | 0.0014 +/- 0.0003 | 0.0077 +/- 0.0002 | 0.5889 +/- 0.0014 | 5/5 / 5/5 / 5/5 |
| In-domain | version 5 IRLS | 0.9411 +/- 0.0009 | 0.0033 +/- 0.0009 | 0.0011 +/- 0.0002 | 0.3247 +/- 0.0011 | 5/5 / 5/5 / 5/5 |
| OOD covariate | scalar | 0.5987 +/- 0.0072 | 0.0219 +/- 0.0032 | 0.0562 +/- 0.0014 | 0.9600 +/- 0.0087 | 0/5 / 3/5 / 0/5 |
| OOD covariate | version 4 | 0.5963 +/- 0.0067 | 0.0229 +/- 0.0031 | 0.0622 +/- 0.0012 | 0.9595 +/- 0.0087 | 0/5 / 3/5 / 0/5 |
| OOD covariate | version 5 IRLS | 0.4649 +/- 0.0499 | 0.0109 +/- 0.0028 | 0.0912 +/- 0.0084 | 0.5186 +/- 0.0153 | 0/5 / 5/5 / 0/5 |
| OOD effect-size | scalar | 0.7597 +/- 0.0031 | 0.0021 +/- 0.0004 | 0.0330 +/- 0.0008 | 1.2526 +/- 0.0020 | 0/5 / 5/5 / 0/5 |
| OOD effect-size | version 4 | 0.6882 +/- 0.0009 | 0.0040 +/- 0.0005 | 0.0570 +/- 0.0001 | 1.2519 +/- 0.0020 | 0/5 / 5/5 / 0/5 |
| OOD effect-size | version 5 IRLS | 0.7280 +/- 0.0026 | 0.0016 +/- 0.0005 | 0.0476 +/- 0.0005 | 0.6783 +/- 0.0005 | 0/5 / 5/5 / 0/5 |
| OOD combined | scalar | 0.4927 +/- 0.0076 | 0.0187 +/- 0.0025 | 0.0802 +/- 0.0015 | 1.4961 +/- 0.0065 | 0/5 / 5/5 / 0/5 |
| OOD combined | version 4 | 0.4726 +/- 0.0065 | 0.0191 +/- 0.0024 | 0.0869 +/- 0.0012 | 1.4957 +/- 0.0064 | 0/5 / 5/5 / 0/5 |
| OOD combined | version 5 IRLS | 0.3226 +/- 0.0342 | 0.0054 +/- 0.0012 | 0.1157 +/- 0.0056 | 0.9304 +/- 0.0089 | 0/5 / 5/5 / 0/5 |

## Paired deltas

Positive coverage deltas are better when coverage is below target. Negative
rank-error and RMSE deltas are better.

| Domain | Delta | Coverage 95 | Rank mean error | Rank variance error | Beta RMSE |
| --- | --- | ---: | ---: | ---: | ---: |
| In-domain | v5 - scalar | -0.0009 +/- 0.0020 | -0.0006 +/- 0.0011 | -0.0232 +/- 0.0008 | -0.2657 +/- 0.0021 |
| In-domain | v5 - version 4 | -0.0027 +/- 0.0012 | 0.0020 +/- 0.0012 | -0.0066 +/- 0.0004 | -0.2642 +/- 0.0021 |
| OOD covariate | v5 - scalar | -0.1338 +/- 0.0495 | -0.0110 +/- 0.0036 | 0.0350 +/- 0.0084 | -0.4414 +/- 0.0209 |
| OOD covariate | v5 - version 4 | -0.1314 +/- 0.0504 | -0.0120 +/- 0.0034 | 0.0290 +/- 0.0086 | -0.4409 +/- 0.0209 |
| OOD effect-size | v5 - scalar | -0.0317 +/- 0.0046 | -0.0004 +/- 0.0006 | 0.0146 +/- 0.0009 | -0.5743 +/- 0.0024 |
| OOD effect-size | v5 - version 4 | 0.0398 +/- 0.0025 | -0.0023 +/- 0.0008 | -0.0095 +/- 0.0004 | -0.5737 +/- 0.0023 |
| OOD combined | v5 - scalar | -0.1701 +/- 0.0346 | -0.0133 +/- 0.0029 | 0.0355 +/- 0.0056 | -0.5657 +/- 0.0130 |
| OOD combined | v5 - version 4 | -0.1500 +/- 0.0348 | -0.0137 +/- 0.0028 | 0.0288 +/- 0.0056 | -0.5654 +/- 0.0129 |

## In-domain stratified gates for version 5 IRLS

| Stratum | Coverage 95 | Rank mean error | Rank variance error | Seed passes: coverage / mean / variance |
| --- | ---: | ---: | ---: | --- |
| Coefficient: Intercept | 0.9438 +/- 0.0011 | 0.0018 +/- 0.0008 | 0.0012 +/- 0.0002 | 5/5 / 5/5 / 5/5 |
| Coefficient: x1 | 0.9411 +/- 0.0007 | 0.0015 +/- 0.0006 | 0.0009 +/- 0.0002 | 5/5 / 5/5 / 5/5 |
| Coefficient: x2 | 0.9385 +/- 0.0013 | 0.0090 +/- 0.0020 | 0.0012 +/- 0.0002 | 5/5 / 5/5 / 5/5 |
| Design information: high | 0.9225 +/- 0.0008 | 0.0028 +/- 0.0007 | 0.0051 +/- 0.0005 | 5/5 / 5/5 / 5/5 |
| Design information: intermediate | 0.9608 +/- 0.0008 | 0.0042 +/- 0.0013 | 0.0082 +/- 0.0002 | 5/5 / 5/5 / 5/5 |
| Design information: low | 0.9401 +/- 0.0014 | 0.0043 +/- 0.0010 | 0.0004 +/- 0.0001 | 5/5 / 5/5 / 5/5 |
| Prevalence: common | 0.9417 +/- 0.0008 | 0.0046 +/- 0.0008 | 0.0015 +/- 0.0003 | 5/5 / 5/5 / 5/5 |
| Prevalence: intermediate | 0.9441 +/- 0.0015 | 0.0041 +/- 0.0010 | 0.0008 +/- 0.0003 | 5/5 / 5/5 / 5/5 |
| Prevalence: rare | 0.9067 +/- 0.0046 | 0.0064 +/- 0.0006 | 0.0087 +/- 0.0017 | 4/5 / 5/5 / 5/5 |

## Decision

Version 5 IRLS is not qualified as the next default calibration path.

It is a strong in-domain improvement over scalar calibration and version 4:
overall rank-variance error drops to `0.0011`, all overall gates pass in all
five seeds, and Beta RMSE improves by roughly `0.26` in-domain. It also fixes
the version 4 rare/intermediate rank-mean pathology under in-domain SBC.

The blocker is OOD robustness. The IRLS anchor improves posterior mean RMSE
substantially, but OOD coefficient coverage regresses versus scalar under
covariate and combined shifts, and rank-variance error worsens under covariate
and combined shifts. The acceptance gate therefore remains failed for every OOD
regime.

## Recommended next step

Keep the IRLS-Laplace anchor as an experimental candidate, but do not promote it.
The next implementation step should decouple the sharper IRLS posterior mean
from the OOD uncertainty scale, most likely by adding an OOD-aware variance
inflation or support-distance term on top of the IRLS anchor rather than
changing the mean anchor again.
