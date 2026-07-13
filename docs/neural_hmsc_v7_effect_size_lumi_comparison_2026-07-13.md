# Neural HMSC v7 effect-size-aware OOD five-seed LUMI comparison

Date: 2026-07-13

Branch: `feature/neural-hmsc`

Candidate commit: `2563e4c Add effect-size-aware v7 OOD uncertainty`

## Objective

Evaluate the effect-size-aware revision of the version 7 learned OOD
calibration objective against the frozen scalar, version 4, version 5 IRLS,
version 6 default, conservative version 6 strength-1.5/cap-8, and support-only
version 7 references.

The effect-size-aware version keeps the IRLS/Laplace probit anchor and learned
OOD objective, but changes the learned OOD inflation curve from support-only to
support plus positive standardized posterior-mean magnitude:
`support_effect_learned_softplus`.

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
- OOD uncertainty transform: `support_effect_learned_softplus`
- OOD max multiplier: `8.0`
- Probit anchor: `irls_laplace`
- IRLS iterations/prior precision/eta clip: `8` / `1.0` / `6.0`
- Predictive calibration semantics version: `2`

Each diagnostics file contained 80 SBC rows. All five calibration records
reported `semantics_version=7`, `support_effect_learned_softplus`, predictive
semantics version `2`, and OOD objective domains `covariate_shift`,
`effect_size_shift`, and `combined_shift`.

Downloaded summary artifacts are staged locally under:

```text
/private/tmp/neural_hmsc_v7_effect_ood_2563e4c_summary/
```

## LUMI Jobs

The initial `standard-g` jobs were canceled while pending and replaced with
`dev-g` jobs because the workflow fits the development queue and prior runs
completed in about 12 minutes.

| Seed | Job | Partition | State | Elapsed | MaxRSS |
| --- | ---: | --- | --- | --- | ---: |
| 20260626 | 19835554 | dev-g | COMPLETED | 00:13:12 | 2217032K |
| 20260627 | 19835555 | dev-g | COMPLETED | 00:13:17 | 2228976K |
| 20260628 | 19835716 | dev-g | COMPLETED | 00:11:59 | 2241892K |
| 20260629 | 19835717 | dev-g | COMPLETED | 00:12:15 | 2231004K |
| 20260630 | 19835779 | dev-g | COMPLETED | 00:12:12 | 2220808K |

## Overall SBC Comparison

Metrics are five-seed mean +/- standard error. Coverage is coefficient
posterior 95% interval coverage, not predictive calibration.

| Domain | Model | Coverage 95 | Rank mean error | Rank variance error | Beta RMSE | Seed passes: coverage / mean / variance |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| In-domain | scalar | 0.9420 +/- 0.0015 | 0.0039 +/- 0.0007 | 0.0243 +/- 0.0007 | 0.5903 +/- 0.0015 | 5/5 / 5/5 / 0/5 |
| In-domain | version 4 | 0.9439 +/- 0.0007 | 0.0014 +/- 0.0003 | 0.0077 +/- 0.0002 | 0.5889 +/- 0.0014 | 5/5 / 5/5 / 5/5 |
| In-domain | version 5 IRLS | 0.9411 +/- 0.0009 | 0.0033 +/- 0.0009 | 0.0011 +/- 0.0002 | 0.3247 +/- 0.0011 | 5/5 / 5/5 / 5/5 |
| In-domain | version 6 default | 0.9416 +/- 0.0008 | 0.0033 +/- 0.0010 | 0.0011 +/- 0.0002 | 0.3247 +/- 0.0011 | 5/5 / 5/5 / 5/5 |
| In-domain | version 6 strength 1.5 cap 8 | 0.9418 +/- 0.0007 | 0.0033 +/- 0.0010 | 0.0011 +/- 0.0002 | 0.3250 +/- 0.0011 | 5/5 / 5/5 / 5/5 |
| In-domain | version 7 support-only | 0.9417 +/- 0.0007 | 0.0033 +/- 0.0010 | 0.0013 +/- 0.0002 | 0.3253 +/- 0.0012 | 5/5 / 5/5 / 5/5 |
| In-domain | version 7 effect-aware | 0.9427 +/- 0.0008 | 0.0031 +/- 0.0009 | 0.0062 +/- 0.0003 | 0.3259 +/- 0.0012 | 5/5 / 5/5 / 5/5 |
| OOD covariate | scalar | 0.5987 +/- 0.0072 | 0.0219 +/- 0.0032 | 0.0562 +/- 0.0014 | 0.9600 +/- 0.0087 | 0/5 / 3/5 / 0/5 |
| OOD covariate | version 4 | 0.5963 +/- 0.0067 | 0.0229 +/- 0.0031 | 0.0622 +/- 0.0012 | 0.9595 +/- 0.0087 | 0/5 / 3/5 / 0/5 |
| OOD covariate | version 5 IRLS | 0.4649 +/- 0.0499 | 0.0109 +/- 0.0028 | 0.0912 +/- 0.0084 | 0.5186 +/- 0.0153 | 0/5 / 5/5 / 0/5 |
| OOD covariate | version 6 default | 0.6384 +/- 0.0483 | 0.0091 +/- 0.0025 | 0.0535 +/- 0.0098 | 0.5206 +/- 0.0152 | 0/5 / 5/5 / 0/5 |
| OOD covariate | version 6 strength 1.5 cap 8 | 0.7316 +/- 0.0384 | 0.0080 +/- 0.0022 | 0.0309 +/- 0.0087 | 0.5264 +/- 0.0150 | 0/5 / 5/5 / 1/5 |
| OOD covariate | version 7 support-only | 0.7623 +/- 0.0382 | 0.0077 +/- 0.0022 | 0.0224 +/- 0.0090 | 0.5273 +/- 0.0149 | 0/5 / 5/5 / 3/5 |
| OOD covariate | version 7 effect-aware | 0.7675 +/- 0.0375 | 0.0075 +/- 0.0022 | 0.0211 +/- 0.0090 | 0.5272 +/- 0.0149 | 0/5 / 5/5 / 5/5 |
| OOD effect-size | scalar | 0.7597 +/- 0.0031 | 0.0021 +/- 0.0004 | 0.0330 +/- 0.0008 | 1.2526 +/- 0.0020 | 0/5 / 5/5 / 0/5 |
| OOD effect-size | version 4 | 0.6882 +/- 0.0009 | 0.0040 +/- 0.0005 | 0.0570 +/- 0.0001 | 1.2519 +/- 0.0020 | 0/5 / 5/5 / 0/5 |
| OOD effect-size | version 5 IRLS | 0.7280 +/- 0.0026 | 0.0016 +/- 0.0005 | 0.0476 +/- 0.0005 | 0.6783 +/- 0.0005 | 0/5 / 5/5 / 0/5 |
| OOD effect-size | version 6 default | 0.7468 +/- 0.0020 | 0.0018 +/- 0.0006 | 0.0434 +/- 0.0004 | 0.6784 +/- 0.0005 | 0/5 / 5/5 / 0/5 |
| OOD effect-size | version 6 strength 1.5 cap 8 | 0.7581 +/- 0.0018 | 0.0019 +/- 0.0006 | 0.0404 +/- 0.0004 | 0.6792 +/- 0.0005 | 0/5 / 5/5 / 0/5 |
| OOD effect-size | version 7 support-only | 0.7715 +/- 0.0015 | 0.0020 +/- 0.0006 | 0.0366 +/- 0.0003 | 0.6796 +/- 0.0005 | 0/5 / 5/5 / 0/5 |
| OOD effect-size | version 7 effect-aware | 0.8481 +/- 0.0019 | 0.0015 +/- 0.0005 | 0.0146 +/- 0.0006 | 0.6804 +/- 0.0005 | 0/5 / 5/5 / 5/5 |
| OOD combined | scalar | 0.4927 +/- 0.0076 | 0.0187 +/- 0.0025 | 0.0802 +/- 0.0015 | 1.4961 +/- 0.0065 | 0/5 / 5/5 / 0/5 |
| OOD combined | version 4 | 0.4726 +/- 0.0065 | 0.0191 +/- 0.0024 | 0.0869 +/- 0.0012 | 1.4957 +/- 0.0064 | 0/5 / 5/5 / 0/5 |
| OOD combined | version 5 IRLS | 0.3226 +/- 0.0342 | 0.0054 +/- 0.0012 | 0.1157 +/- 0.0056 | 0.9304 +/- 0.0089 | 0/5 / 5/5 / 0/5 |
| OOD combined | version 6 default | 0.4854 +/- 0.0393 | 0.0054 +/- 0.0011 | 0.0837 +/- 0.0074 | 0.9315 +/- 0.0088 | 0/5 / 5/5 / 0/5 |
| OOD combined | version 6 strength 1.5 cap 8 | 0.5887 +/- 0.0345 | 0.0049 +/- 0.0010 | 0.0611 +/- 0.0073 | 0.9349 +/- 0.0088 | 0/5 / 5/5 / 0/5 |
| OOD combined | version 7 support-only | 0.6278 +/- 0.0362 | 0.0044 +/- 0.0009 | 0.0518 +/- 0.0079 | 0.9355 +/- 0.0088 | 0/5 / 5/5 / 0/5 |
| OOD combined | version 7 effect-aware | 0.6439 +/- 0.0373 | 0.0046 +/- 0.0009 | 0.0480 +/- 0.0081 | 0.9355 +/- 0.0088 | 0/5 / 5/5 / 5/5 |

## Inflation Diagnostics

| Domain | Inflation mean | Inflation max | Inflated fraction | Support trust mean | Fallback fraction | Effect signal mean | Effect positive fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| In-domain | 1.3972 +/- 0.0147 | 8.0000 +/- 0.0000 | 0.4933 +/- 0.0009 | 0.9889 +/- 0.0008 | 0.0113 +/- 0.0011 | 0.4205 +/- 0.0005 | 0.4745 +/- 0.0003 |
| OOD covariate | 5.7616 +/- 0.2158 | 8.0000 +/- 0.0000 | 0.8872 +/- 0.0117 | 0.3523 +/- 0.0310 | 0.6575 +/- 0.0304 | 0.4389 +/- 0.0125 | 0.4922 +/- 0.0041 |
| OOD effect-size | 2.5326 +/- 0.0477 | 8.0000 +/- 0.0000 | 0.6990 +/- 0.0020 | 0.9221 +/- 0.0023 | 0.0804 +/- 0.0026 | 0.8187 +/- 0.0011 | 0.6443 +/- 0.0009 |
| OOD combined | 6.0762 +/- 0.2076 | 8.0000 +/- 0.0000 | 0.9151 +/- 0.0097 | 0.3296 +/- 0.0321 | 0.6824 +/- 0.0315 | 0.6448 +/- 0.0098 | 0.5989 +/- 0.0020 |

## Paired Deltas

Positive coverage deltas are better when coverage is below target. Negative
rank-error and RMSE deltas are better.

| Domain | Delta | Coverage 95 | Rank mean error | Rank variance error | Beta RMSE |
| --- | --- | ---: | ---: | ---: | ---: |
| In-domain | effect-aware v7 - support-only v7 | +0.0010 | -0.0002 | +0.0049 | +0.0006 |
| OOD covariate | effect-aware v7 - support-only v7 | +0.0052 | -0.0002 | -0.0013 | -0.0001 |
| OOD effect-size | effect-aware v7 - support-only v7 | +0.0766 | -0.0005 | -0.0220 | +0.0008 |
| OOD combined | effect-aware v7 - support-only v7 | +0.0161 | +0.0002 | -0.0038 | -0.0000 |
| OOD covariate | effect-aware v7 - version 6 strength 1.5 cap 8 | +0.0359 | -0.0005 | -0.0098 | +0.0008 |
| OOD effect-size | effect-aware v7 - version 6 strength 1.5 cap 8 | +0.0900 | -0.0004 | -0.0258 | +0.0012 |
| OOD combined | effect-aware v7 - version 6 strength 1.5 cap 8 | +0.0552 | -0.0003 | -0.0131 | +0.0006 |

## In-domain Stratified Gates

| Stratum | Coverage 95 | Rank mean error | Rank variance error | Seed passes: coverage / mean / variance |
| --- | ---: | ---: | ---: | --- |
| Coefficient: Intercept | 0.9467 +/- 0.0009 | 0.0019 +/- 0.0008 | 0.0065 +/- 0.0003 | 5/5 / 5/5 / 5/5 |
| Coefficient: x1 | 0.9417 +/- 0.0009 | 0.0016 +/- 0.0006 | 0.0057 +/- 0.0004 | 5/5 / 5/5 / 5/5 |
| Coefficient: x2 | 0.9398 +/- 0.0012 | 0.0085 +/- 0.0019 | 0.0064 +/- 0.0003 | 5/5 / 5/5 / 5/5 |
| Design information: high | 0.9021 +/- 0.0013 | 0.0029 +/- 0.0007 | 0.0116 +/- 0.0004 | 3/5 / 5/5 / 5/5 |
| Design information: intermediate | 0.9550 +/- 0.0010 | 0.0041 +/- 0.0014 | 0.0054 +/- 0.0003 | 5/5 / 5/5 / 5/5 |
| Design information: low | 0.9711 +/- 0.0005 | 0.0034 +/- 0.0008 | 0.0248 +/- 0.0008 | 5/5 / 5/5 / 5/5 |
| Prevalence: common | 0.9418 +/- 0.0009 | 0.0028 +/- 0.0008 | 0.0058 +/- 0.0004 | 5/5 / 5/5 / 5/5 |
| Prevalence: intermediate | 0.9493 +/- 0.0010 | 0.0033 +/- 0.0015 | 0.0069 +/- 0.0004 | 5/5 / 5/5 / 5/5 |
| Prevalence: rare | 0.9278 +/- 0.0027 | 0.0143 +/- 0.0017 | 0.0147 +/- 0.0012 | 5/5 / 5/5 / 5/5 |

## Decision

Effect-aware version 7 is not qualified as the default calibration path.

It successfully addresses the intended structural gap in the support-only OOD
objective: effect-size-shift coverage improves from `0.7715` to `0.8481`, and
rank-variance error improves from `0.0366` to `0.0146`. Combined-shift coverage
also improves from `0.6278` to `0.6439`.

The result still fails the OOD coverage requirement. No OOD regime reaches
nominal 95% coefficient coverage, and effect-size shift remains below the
minimum 0.90 stress gate. The effect-aware curve also inflates in-domain
uncertainty much more often than support-only v7, with an in-domain inflated
fraction of `0.4933`; overall in-domain gates pass, but the high-design
information stratum only passes coverage in `3/5` seeds.

## Recommended Next Step

Do not promote effect-aware v7 as default. Keep it as evidence that
posterior-mean magnitude is a useful OOD signal, but the next implementation
should make the effect-size signal conditional on OOD context. A plausible next
step is to gate effect-size inflation by support trust, prevalence/design
strata, or an explicit in-domain penalty on the effect-size branch so it can
retain the OOD effect-size gain without weakening high-information in-domain
coverage.
