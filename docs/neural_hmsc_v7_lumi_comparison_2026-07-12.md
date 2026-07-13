# Neural HMSC v7 learned OOD five-seed LUMI comparison

Date: 2026-07-12

Branch: `feature/neural-hmsc`

Candidate commit: `cdc9c6a Add learned OOD calibration objective`

## Objective

Evaluate the version 7 learned OOD calibration objective against the frozen
scalar, version 4, version 5 IRLS, version 6 default, and conservative version 6
strength-1.5/cap-8 references.

Version 7 keeps the IRLS/Laplace probit anchor and conditional rank-aware
coefficient calibrator, but replaces the fixed support-excess OOD inflation
rule with a learned bounded softplus curve fit from held-out OOD simulations.

## Configuration

- Distribution: probit
- Seeds: `20260626`, `20260627`, `20260628`, `20260629`, `20260630`
- Sites/species: `40` / `75`
- Training/calibration/SBC datasets: `512` / `128` / `128`
- OOD calibration datasets per regime: `128`
- SBC draws/bins: `512` / `10`
- OOD regimes: covariate shift, effect-size shift, combined shift
- Neural chains/draws: `4` / `500`
- Coefficient calibration: `conditional`
- Conditional method: `conditional_rank_aware_anchor_scale`
- Conditional semantics version: `7`
- OOD objective: `support_excess_rank_coverage`
- OOD uncertainty transform: `support_excess_learned_softplus`
- OOD max multiplier: `8.0`
- Probit anchor: `irls_laplace`
- IRLS iterations/prior precision/eta clip: `8` / `1.0` / `6.0`
- Predictive calibration semantics version: `2`

Each v7 diagnostics file contained 80 SBC rows. All five calibration records
reported `semantics_version=7`, `support_excess_learned_softplus`, predictive
semantics version `2`, and OOD objective domains `covariate_shift`,
`effect_size_shift`, and `combined_shift`.

## LUMI Jobs

| Seed | Job | Partition | State | Elapsed | MaxRSS |
| --- | ---: | --- | --- | --- | ---: |
| 20260626 | 19831708 | standard-g | COMPLETED | 00:12:00 | 4615268K |
| 20260627 | 19831709 | standard-g | COMPLETED | 00:11:56 | 4628172K |
| 20260628 | 19831710 | standard-g | COMPLETED | 00:12:05 | 4604276K |
| 20260629 | 19831711 | standard-g | COMPLETED | 00:11:59 | 4615176K |
| 20260630 | 19831712 | standard-g | COMPLETED | 00:12:06 | 4603720K |

Downloaded artifacts are staged locally under:

```text
/private/tmp/neural_hmsc_v7_ood_objective_cdc9c6a/
```

## Overall SBC Comparison

Metrics are five-seed mean +/- standard error. Coverage is coefficient posterior
95% interval coverage, not predictive calibration. Rank-mean and rank-variance
columns are absolute errors from the SBC uniform target.

| Domain | Model | Coverage 95 | Rank mean error | Rank variance error | Beta RMSE | Seed passes: coverage / mean / variance |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| In-domain | scalar | 0.9420 +/- 0.0015 | 0.0039 +/- 0.0007 | 0.0243 +/- 0.0007 | 0.5903 +/- 0.0015 | 5/5 / 5/5 / 0/5 |
| In-domain | version 4 | 0.9439 +/- 0.0007 | 0.0014 +/- 0.0003 | 0.0077 +/- 0.0002 | 0.5889 +/- 0.0014 | 5/5 / 5/5 / 5/5 |
| In-domain | version 5 IRLS | 0.9411 +/- 0.0009 | 0.0033 +/- 0.0009 | 0.0011 +/- 0.0002 | 0.3247 +/- 0.0011 | 5/5 / 5/5 / 5/5 |
| In-domain | version 6 default | 0.9416 +/- 0.0008 | 0.0033 +/- 0.0010 | 0.0011 +/- 0.0002 | 0.3247 +/- 0.0011 | 5/5 / 5/5 / 5/5 |
| In-domain | version 6 strength 1.5 cap 8 | 0.9418 +/- 0.0007 | 0.0033 +/- 0.0010 | 0.0011 +/- 0.0002 | 0.3250 +/- 0.0011 | 5/5 / 5/5 / 5/5 |
| In-domain | version 7 learned OOD | 0.9417 +/- 0.0007 | 0.0033 +/- 0.0010 | 0.0013 +/- 0.0002 | 0.3253 +/- 0.0012 | 5/5 / 5/5 / 5/5 |
| OOD covariate | scalar | 0.5987 +/- 0.0072 | 0.0219 +/- 0.0032 | 0.0562 +/- 0.0014 | 0.9600 +/- 0.0087 | 0/5 / 3/5 / 0/5 |
| OOD covariate | version 4 | 0.5963 +/- 0.0067 | 0.0229 +/- 0.0031 | 0.0622 +/- 0.0012 | 0.9595 +/- 0.0087 | 0/5 / 3/5 / 0/5 |
| OOD covariate | version 5 IRLS | 0.4649 +/- 0.0499 | 0.0109 +/- 0.0028 | 0.0912 +/- 0.0084 | 0.5186 +/- 0.0153 | 0/5 / 5/5 / 0/5 |
| OOD covariate | version 6 default | 0.6384 +/- 0.0483 | 0.0091 +/- 0.0025 | 0.0535 +/- 0.0098 | 0.5206 +/- 0.0152 | 0/5 / 5/5 / 0/5 |
| OOD covariate | version 6 strength 1.5 cap 8 | 0.7316 +/- 0.0384 | 0.0080 +/- 0.0022 | 0.0309 +/- 0.0087 | 0.5264 +/- 0.0150 | 0/5 / 5/5 / 1/5 |
| OOD covariate | version 7 learned OOD | 0.7623 +/- 0.0382 | 0.0077 +/- 0.0022 | 0.0224 +/- 0.0090 | 0.5273 +/- 0.0149 | 0/5 / 5/5 / 3/5 |
| OOD effect-size | scalar | 0.7597 +/- 0.0031 | 0.0021 +/- 0.0004 | 0.0330 +/- 0.0008 | 1.2526 +/- 0.0020 | 0/5 / 5/5 / 0/5 |
| OOD effect-size | version 4 | 0.6882 +/- 0.0009 | 0.0040 +/- 0.0005 | 0.0570 +/- 0.0001 | 1.2519 +/- 0.0020 | 0/5 / 5/5 / 0/5 |
| OOD effect-size | version 5 IRLS | 0.7280 +/- 0.0026 | 0.0016 +/- 0.0005 | 0.0476 +/- 0.0005 | 0.6783 +/- 0.0005 | 0/5 / 5/5 / 0/5 |
| OOD effect-size | version 6 default | 0.7468 +/- 0.0020 | 0.0018 +/- 0.0006 | 0.0434 +/- 0.0004 | 0.6784 +/- 0.0005 | 0/5 / 5/5 / 0/5 |
| OOD effect-size | version 6 strength 1.5 cap 8 | 0.7581 +/- 0.0018 | 0.0019 +/- 0.0006 | 0.0404 +/- 0.0004 | 0.6792 +/- 0.0005 | 0/5 / 5/5 / 0/5 |
| OOD effect-size | version 7 learned OOD | 0.7715 +/- 0.0015 | 0.0020 +/- 0.0006 | 0.0366 +/- 0.0003 | 0.6796 +/- 0.0005 | 0/5 / 5/5 / 0/5 |
| OOD combined | scalar | 0.4927 +/- 0.0076 | 0.0187 +/- 0.0025 | 0.0802 +/- 0.0015 | 1.4961 +/- 0.0065 | 0/5 / 5/5 / 0/5 |
| OOD combined | version 4 | 0.4726 +/- 0.0065 | 0.0191 +/- 0.0024 | 0.0869 +/- 0.0012 | 1.4957 +/- 0.0064 | 0/5 / 5/5 / 0/5 |
| OOD combined | version 5 IRLS | 0.3226 +/- 0.0342 | 0.0054 +/- 0.0012 | 0.1157 +/- 0.0056 | 0.9304 +/- 0.0089 | 0/5 / 5/5 / 0/5 |
| OOD combined | version 6 default | 0.4854 +/- 0.0393 | 0.0054 +/- 0.0011 | 0.0837 +/- 0.0074 | 0.9315 +/- 0.0088 | 0/5 / 5/5 / 0/5 |
| OOD combined | version 6 strength 1.5 cap 8 | 0.5887 +/- 0.0345 | 0.0049 +/- 0.0010 | 0.0611 +/- 0.0073 | 0.9349 +/- 0.0088 | 0/5 / 5/5 / 0/5 |
| OOD combined | version 7 learned OOD | 0.6278 +/- 0.0362 | 0.0044 +/- 0.0009 | 0.0518 +/- 0.0079 | 0.9355 +/- 0.0088 | 0/5 / 5/5 / 0/5 |

## Inflation Diagnostics

| Domain | Model | Inflation mean | Inflation max | Inflated fraction | Support trust mean | Fallback fraction |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| In-domain | version 6 default | 1.0170 +/- 0.0014 | 4.0000 +/- 0.0000 | 0.0336 +/- 0.0015 | 0.9889 +/- 0.0008 | 0.0113 +/- 0.0011 |
| In-domain | version 6 strength 1.5 cap 8 | 1.0436 +/- 0.0030 | 8.0000 +/- 0.0000 | 0.0337 +/- 0.0015 | 0.9889 +/- 0.0008 | 0.0113 +/- 0.0011 |
| In-domain | version 7 learned OOD | 1.0947 +/- 0.0075 | 8.0000 +/- 0.0000 | 0.0340 +/- 0.0015 | 0.9889 +/- 0.0008 | 0.0113 +/- 0.0011 |
| OOD covariate | version 6 default | 2.7460 +/- 0.1031 | 4.0000 +/- 0.0000 | 0.7467 +/- 0.0256 | 0.3523 +/- 0.0310 | 0.6575 +/- 0.0304 |
| OOD covariate | version 6 strength 1.5 cap 8 | 5.1787 +/- 0.2361 | 8.0000 +/- 0.0000 | 0.7467 +/- 0.0256 | 0.3523 +/- 0.0310 | 0.6575 +/- 0.0304 |
| OOD covariate | version 7 learned OOD | 5.7352 +/- 0.2065 | 8.0000 +/- 0.0000 | 0.7469 +/- 0.0255 | 0.3523 +/- 0.0310 | 0.6575 +/- 0.0304 |
| OOD effect-size | version 6 default | 1.1490 +/- 0.0055 | 4.0000 +/- 0.0000 | 0.1497 +/- 0.0031 | 0.9221 +/- 0.0023 | 0.0804 +/- 0.0026 |
| OOD effect-size | version 6 strength 1.5 cap 8 | 1.3777 +/- 0.0120 | 8.0000 +/- 0.0000 | 0.1498 +/- 0.0031 | 0.9221 +/- 0.0023 | 0.0804 +/- 0.0026 |
| OOD effect-size | version 7 learned OOD | 1.6326 +/- 0.0191 | 8.0000 +/- 0.0000 | 0.1500 +/- 0.0031 | 0.9221 +/- 0.0023 | 0.0804 +/- 0.0026 |
| OOD combined | version 6 default | 2.7942 +/- 0.1098 | 4.0000 +/- 0.0000 | 0.7777 +/- 0.0243 | 0.3296 +/- 0.0321 | 0.6824 +/- 0.0315 |
| OOD combined | version 6 strength 1.5 cap 8 | 5.3032 +/- 0.2498 | 8.0000 +/- 0.0000 | 0.7778 +/- 0.0242 | 0.3296 +/- 0.0321 | 0.6824 +/- 0.0315 |
| OOD combined | version 7 learned OOD | 5.9152 +/- 0.2110 | 8.0000 +/- 0.0000 | 0.7779 +/- 0.0242 | 0.3296 +/- 0.0321 | 0.6824 +/- 0.0315 |

## Paired Deltas

Positive coverage deltas are better when coverage is below target. Negative
rank-error and RMSE deltas are better.

| Domain | Delta | Coverage 95 | Rank mean error | Rank variance error | Beta RMSE |
| --- | --- | ---: | ---: | ---: | ---: |
| In-domain | v7 - version 6 default | 0.0001 +/- 0.0003 | -0.0000 +/- 0.0001 | 0.0002 +/- 0.0001 | 0.0006 +/- 0.0001 |
| In-domain | v7 - version 6 strength 1.5 cap 8 | -0.0001 +/- 0.0002 | -0.0000 +/- 0.0000 | 0.0002 +/- 0.0001 | 0.0004 +/- 0.0001 |
| OOD covariate | v7 - version 6 default | 0.1239 +/- 0.0104 | -0.0014 +/- 0.0004 | -0.0311 +/- 0.0009 | 0.0067 +/- 0.0013 |
| OOD covariate | v7 - version 6 strength 1.5 cap 8 | 0.0307 +/- 0.0008 | -0.0003 +/- 0.0001 | -0.0084 +/- 0.0003 | 0.0008 +/- 0.0001 |
| OOD effect-size | v7 - version 6 default | 0.0247 +/- 0.0014 | 0.0002 +/- 0.0001 | -0.0068 +/- 0.0003 | 0.0012 +/- 0.0001 |
| OOD effect-size | v7 - version 6 strength 1.5 cap 8 | 0.0134 +/- 0.0008 | 0.0001 +/- 0.0001 | -0.0038 +/- 0.0002 | 0.0005 +/- 0.0001 |
| OOD combined | v7 - version 6 default | 0.1424 +/- 0.0047 | -0.0009 +/- 0.0003 | -0.0318 +/- 0.0008 | 0.0040 +/- 0.0006 |
| OOD combined | v7 - version 6 strength 1.5 cap 8 | 0.0390 +/- 0.0020 | -0.0005 +/- 0.0001 | -0.0092 +/- 0.0006 | 0.0005 +/- 0.0001 |

## In-domain Stratified Gates

| Stratum | Coverage 95 | Rank mean error | Rank variance error | Seed passes: coverage / mean / variance |
| --- | ---: | ---: | ---: | --- |
| Coefficient: Intercept | 0.9433 +/- 0.0010 | 0.0019 +/- 0.0008 | 0.0011 +/- 0.0001 | 5/5 / 5/5 / 5/5 |
| Coefficient: x1 | 0.9421 +/- 0.0005 | 0.0016 +/- 0.0007 | 0.0012 +/- 0.0002 | 5/5 / 5/5 / 5/5 |
| Coefficient: x2 | 0.9397 +/- 0.0013 | 0.0092 +/- 0.0020 | 0.0017 +/- 0.0002 | 5/5 / 5/5 / 5/5 |
| Design information: high | 0.9210 +/- 0.0009 | 0.0029 +/- 0.0008 | 0.0057 +/- 0.0005 | 5/5 / 5/5 / 5/5 |
| Design information: intermediate | 0.9595 +/- 0.0006 | 0.0041 +/- 0.0013 | 0.0073 +/- 0.0002 | 5/5 / 5/5 / 5/5 |
| Design information: low | 0.9446 +/- 0.0010 | 0.0042 +/- 0.0010 | 0.0022 +/- 0.0003 | 5/5 / 5/5 / 5/5 |
| Prevalence: common | 0.9423 +/- 0.0008 | 0.0046 +/- 0.0009 | 0.0016 +/- 0.0003 | 5/5 / 5/5 / 5/5 |
| Prevalence: intermediate | 0.9432 +/- 0.0011 | 0.0043 +/- 0.0010 | 0.0007 +/- 0.0002 | 5/5 / 5/5 / 5/5 |
| Prevalence: rare | 0.9162 +/- 0.0037 | 0.0061 +/- 0.0010 | 0.0025 +/- 0.0013 | 5/5 / 5/5 / 5/5 |

## Decision

Version 7 learned OOD is not qualified as the default calibration path because
every OOD regime still fails the absolute 95% coverage gate.

It is nevertheless the strongest OOD candidate so far. It preserves all overall
and stratified in-domain gates, including rare prevalence, and improves OOD
coverage and rank-variance error over both version 6 default and the
strength-1.5/cap-8 tuning candidate. The largest gains are under covariate and
combined shifts, where v7 increases coverage by `0.0307` and `0.0390`
respectively over the conservative v6 sweep candidate.

The remaining failure is structural: the support-excess curve improves
uncertainty where support shift is detected, but it still cannot deliver nominal
coverage under effect-size and combined shifts. More inflation alone may keep
improving coverage, but the cap is already saturated in all domains including a
small in-domain tail.

## Recommended Next Step

Do not promote v7 as the default. Keep it as the best current OOD candidate.

The next implementation should add regime-specific OOD calibration features or
an effect-size-shift detector. The support detector correctly identifies
covariate and combined shifts, but effect-size shift has high support trust
(`0.9221`) and therefore remains under-inflated relative to the coverage target.
