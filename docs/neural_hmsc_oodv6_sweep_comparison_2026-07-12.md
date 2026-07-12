# Neural HMSC OOD v6 tuning sweep comparison

Date: 2026-07-12

Branch: `feature/neural-hmsc`

Base candidate commit: `f06e7b9 Add OOD-aware IRLS uncertainty scaling`

## Objective

Evaluate a small tuning sweep over version 6 OOD uncertainty strength and
maximum multiplier. The sweep was requested after the default version 6 setting
preserved in-domain gates but still failed the absolute OOD coefficient
coverage gate.

The three tested settings were:

| Setting | OOD strength | Max multiplier |
| --- | ---: | ---: |
| `s10_m8` | 1.0 | 8.0 |
| `s15_m8` | 1.5 | 8.0 |
| `s15_m12` | 1.5 | 12.0 |

## Validation Caveat

The intended run was a frozen-checkpoint tuning sweep. Because the Slurm
environment did not propagate `RUN_NAME` and `NEURAL_CHECKPOINT`, the jobs wrote
under Slurm-default run names and retrained the neural checkpoint before
calibration. The OOD tuning settings did propagate correctly and are recorded in
each `benchmark_record.json`.

This report therefore treats the output as a retrained five-seed tuning sweep,
not as a pure frozen-checkpoint comparison. The scalar, version 4, version 5,
and version 6 default references remain frozen local references from the prior
reports.

Validation checks:

- all 15 sweep diagnostics files were downloaded and contained 80 SBC rows
- all 15 calibration records reported `semantics_version=6`
- all 15 calibration records reported
  `method=conditional_rank_aware_anchor_scale`
- all 15 calibration records recorded the requested OOD strength/cap pair
- no sweep record reported `reused_frozen_checkpoint=true`

## LUMI Jobs

| Setting | Seed | Job |
| --- | ---: | ---: |
| `s10_m8` | 20260626 | 19829666 |
| `s10_m8` | 20260627 | 19830568 |
| `s10_m8` | 20260628 | 19830602 |
| `s10_m8` | 20260629 | 19830621 |
| `s10_m8` | 20260630 | 19830638 |
| `s15_m8` | 20260626 | 19829667 |
| `s15_m8` | 20260627 | 19830598 |
| `s15_m8` | 20260628 | 19830603 |
| `s15_m8` | 20260629 | 19830630 |
| `s15_m8` | 20260630 | 19830639 |
| `s15_m12` | 20260626 | 19830567 |
| `s15_m12` | 20260627 | 19830599 |
| `s15_m12` | 20260628 | 19830620 |
| `s15_m12` | 20260629 | 19830631 |
| `s15_m12` | 20260630 | 19830643 |

Downloaded artifacts are staged locally under:

```text
/private/tmp/neural_hmsc_oodv6_sweep_f06e7b9/
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
| In-domain | strength 1.0 cap 8 | 0.9416 +/- 0.0007 | 0.0033 +/- 0.0010 | 0.0011 +/- 0.0002 | 0.3249 +/- 0.0011 | 5/5 / 5/5 / 5/5 |
| In-domain | strength 1.5 cap 8 | 0.9418 +/- 0.0007 | 0.0033 +/- 0.0010 | 0.0011 +/- 0.0002 | 0.3250 +/- 0.0011 | 5/5 / 5/5 / 5/5 |
| In-domain | strength 1.5 cap 12 | 0.9418 +/- 0.0007 | 0.0033 +/- 0.0010 | 0.0011 +/- 0.0002 | 0.3250 +/- 0.0011 | 5/5 / 5/5 / 5/5 |
| OOD covariate | scalar | 0.5987 +/- 0.0072 | 0.0219 +/- 0.0032 | 0.0562 +/- 0.0014 | 0.9600 +/- 0.0087 | 0/5 / 3/5 / 0/5 |
| OOD covariate | version 4 | 0.5963 +/- 0.0067 | 0.0229 +/- 0.0031 | 0.0622 +/- 0.0012 | 0.9595 +/- 0.0087 | 0/5 / 3/5 / 0/5 |
| OOD covariate | version 5 IRLS | 0.4649 +/- 0.0499 | 0.0109 +/- 0.0028 | 0.0912 +/- 0.0084 | 0.5186 +/- 0.0153 | 0/5 / 5/5 / 0/5 |
| OOD covariate | version 6 default | 0.6384 +/- 0.0483 | 0.0091 +/- 0.0025 | 0.0535 +/- 0.0098 | 0.5206 +/- 0.0152 | 0/5 / 5/5 / 0/5 |
| OOD covariate | strength 1.0 cap 8 | 0.7156 +/- 0.0385 | 0.0082 +/- 0.0023 | 0.0346 +/- 0.0086 | 0.5259 +/- 0.0150 | 0/5 / 5/5 / 0/5 |
| OOD covariate | strength 1.5 cap 8 | 0.7316 +/- 0.0384 | 0.0080 +/- 0.0022 | 0.0309 +/- 0.0087 | 0.5264 +/- 0.0150 | 0/5 / 5/5 / 1/5 |
| OOD covariate | strength 1.5 cap 12 | 0.7321 +/- 0.0383 | 0.0080 +/- 0.0022 | 0.0307 +/- 0.0087 | 0.5266 +/- 0.0151 | 0/5 / 5/5 / 1/5 |
| OOD effect-size | scalar | 0.7597 +/- 0.0031 | 0.0021 +/- 0.0004 | 0.0330 +/- 0.0008 | 1.2526 +/- 0.0020 | 0/5 / 5/5 / 0/5 |
| OOD effect-size | version 4 | 0.6882 +/- 0.0009 | 0.0040 +/- 0.0005 | 0.0570 +/- 0.0001 | 1.2519 +/- 0.0020 | 0/5 / 5/5 / 0/5 |
| OOD effect-size | version 5 IRLS | 0.7280 +/- 0.0026 | 0.0016 +/- 0.0005 | 0.0476 +/- 0.0005 | 0.6783 +/- 0.0005 | 0/5 / 5/5 / 0/5 |
| OOD effect-size | version 6 default | 0.7468 +/- 0.0020 | 0.0018 +/- 0.0006 | 0.0434 +/- 0.0004 | 0.6784 +/- 0.0005 | 0/5 / 5/5 / 0/5 |
| OOD effect-size | strength 1.0 cap 8 | 0.7532 +/- 0.0019 | 0.0019 +/- 0.0006 | 0.0417 +/- 0.0004 | 0.6789 +/- 0.0006 | 0/5 / 5/5 / 0/5 |
| OOD effect-size | strength 1.5 cap 8 | 0.7581 +/- 0.0018 | 0.0019 +/- 0.0006 | 0.0404 +/- 0.0004 | 0.6792 +/- 0.0005 | 0/5 / 5/5 / 0/5 |
| OOD effect-size | strength 1.5 cap 12 | 0.7581 +/- 0.0018 | 0.0019 +/- 0.0006 | 0.0404 +/- 0.0003 | 0.6792 +/- 0.0005 | 0/5 / 5/5 / 0/5 |
| OOD combined | scalar | 0.4927 +/- 0.0076 | 0.0187 +/- 0.0025 | 0.0802 +/- 0.0015 | 1.4961 +/- 0.0065 | 0/5 / 5/5 / 0/5 |
| OOD combined | version 4 | 0.4726 +/- 0.0065 | 0.0191 +/- 0.0024 | 0.0869 +/- 0.0012 | 1.4957 +/- 0.0064 | 0/5 / 5/5 / 0/5 |
| OOD combined | version 5 IRLS | 0.3226 +/- 0.0342 | 0.0054 +/- 0.0012 | 0.1157 +/- 0.0056 | 0.9304 +/- 0.0089 | 0/5 / 5/5 / 0/5 |
| OOD combined | version 6 default | 0.4854 +/- 0.0393 | 0.0054 +/- 0.0011 | 0.0837 +/- 0.0074 | 0.9315 +/- 0.0088 | 0/5 / 5/5 / 0/5 |
| OOD combined | strength 1.0 cap 8 | 0.5704 +/- 0.0339 | 0.0053 +/- 0.0010 | 0.0651 +/- 0.0071 | 0.9346 +/- 0.0089 | 0/5 / 5/5 / 0/5 |
| OOD combined | strength 1.5 cap 8 | 0.5887 +/- 0.0345 | 0.0049 +/- 0.0010 | 0.0611 +/- 0.0073 | 0.9349 +/- 0.0088 | 0/5 / 5/5 / 0/5 |
| OOD combined | strength 1.5 cap 12 | 0.5894 +/- 0.0344 | 0.0049 +/- 0.0010 | 0.0609 +/- 0.0073 | 0.9350 +/- 0.0089 | 0/5 / 5/5 / 0/5 |

## OOD Inflation Diagnostics

| Domain | Model | Inflation mean | Inflation max | Inflated fraction | Support trust mean | Fallback fraction |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| In-domain | version 6 default | 1.0170 +/- 0.0014 | 4.0000 +/- 0.0000 | 0.0336 +/- 0.0015 | 0.9889 +/- 0.0008 | 0.0113 +/- 0.0011 |
| In-domain | strength 1.0 cap 8 | 1.0330 +/- 0.0030 | 8.0000 +/- 0.0000 | 0.0337 +/- 0.0015 | 0.9889 +/- 0.0008 | 0.0113 +/- 0.0011 |
| In-domain | strength 1.5 cap 8 | 1.0436 +/- 0.0030 | 8.0000 +/- 0.0000 | 0.0337 +/- 0.0015 | 0.9889 +/- 0.0008 | 0.0113 +/- 0.0011 |
| In-domain | strength 1.5 cap 12 | 1.0577 +/- 0.0045 | 12.0000 +/- 0.0000 | 0.0337 +/- 0.0015 | 0.9889 +/- 0.0008 | 0.0113 +/- 0.0011 |
| OOD covariate | version 6 default | 2.7460 +/- 0.1031 | 4.0000 +/- 0.0000 | 0.7467 +/- 0.0256 | 0.3523 +/- 0.0310 | 0.6575 +/- 0.0304 |
| OOD covariate | strength 1.0 cap 8 | 4.9426 +/- 0.2464 | 8.0000 +/- 0.0000 | 0.7467 +/- 0.0256 | 0.3523 +/- 0.0310 | 0.6575 +/- 0.0304 |
| OOD covariate | strength 1.5 cap 8 | 5.1787 +/- 0.2361 | 8.0000 +/- 0.0000 | 0.7467 +/- 0.0256 | 0.3523 +/- 0.0310 | 0.6575 +/- 0.0304 |
| OOD covariate | strength 1.5 cap 12 | 7.3672 +/- 0.3800 | 12.0000 +/- 0.0000 | 0.7467 +/- 0.0256 | 0.3523 +/- 0.0310 | 0.6575 +/- 0.0304 |
| OOD effect-size | version 6 default | 1.1490 +/- 0.0055 | 4.0000 +/- 0.0000 | 0.1497 +/- 0.0031 | 0.9221 +/- 0.0023 | 0.0804 +/- 0.0026 |
| OOD effect-size | strength 1.0 cap 8 | 1.3044 +/- 0.0128 | 8.0000 +/- 0.0000 | 0.1497 +/- 0.0031 | 0.9221 +/- 0.0023 | 0.0804 +/- 0.0026 |
| OOD effect-size | strength 1.5 cap 8 | 1.3777 +/- 0.0120 | 8.0000 +/- 0.0000 | 0.1498 +/- 0.0031 | 0.9221 +/- 0.0023 | 0.0804 +/- 0.0026 |
| OOD effect-size | strength 1.5 cap 12 | 1.5247 +/- 0.0194 | 12.0000 +/- 0.0000 | 0.1498 +/- 0.0031 | 0.9221 +/- 0.0023 | 0.0804 +/- 0.0026 |
| OOD combined | version 6 default | 2.7942 +/- 0.1098 | 4.0000 +/- 0.0000 | 0.7777 +/- 0.0243 | 0.3296 +/- 0.0321 | 0.6824 +/- 0.0315 |
| OOD combined | strength 1.0 cap 8 | 5.0394 +/- 0.2652 | 8.0000 +/- 0.0000 | 0.7777 +/- 0.0243 | 0.3296 +/- 0.0321 | 0.6824 +/- 0.0315 |
| OOD combined | strength 1.5 cap 8 | 5.3032 +/- 0.2498 | 8.0000 +/- 0.0000 | 0.7778 +/- 0.0242 | 0.3296 +/- 0.0321 | 0.6824 +/- 0.0315 |
| OOD combined | strength 1.5 cap 12 | 7.5414 +/- 0.4052 | 12.0000 +/- 0.0000 | 0.7778 +/- 0.0242 | 0.3296 +/- 0.0321 | 0.6824 +/- 0.0315 |

## Paired Deltas Versus Version 6 Default

Positive coverage deltas are better when coverage is below target. Negative
rank-error and RMSE deltas are better.

| Domain | Setting | Coverage 95 | Rank mean error | Rank variance error | Beta RMSE |
| --- | --- | ---: | ---: | ---: | ---: |
| In-domain | strength 1.0 cap 8 | -0.0000 +/- 0.0001 | -0.0000 +/- 0.0000 | -0.0000 +/- 0.0000 | 0.0002 +/- 0.0000 |
| In-domain | strength 1.5 cap 8 | 0.0002 +/- 0.0001 | -0.0000 +/- 0.0000 | 0.0000 +/- 0.0001 | 0.0002 +/- 0.0000 |
| In-domain | strength 1.5 cap 12 | 0.0002 +/- 0.0001 | -0.0000 +/- 0.0000 | 0.0000 +/- 0.0001 | 0.0003 +/- 0.0000 |
| OOD covariate | strength 1.0 cap 8 | 0.0772 +/- 0.0100 | -0.0009 +/- 0.0004 | -0.0189 +/- 0.0012 | 0.0053 +/- 0.0010 |
| OOD covariate | strength 1.5 cap 8 | 0.0932 +/- 0.0101 | -0.0011 +/- 0.0004 | -0.0227 +/- 0.0011 | 0.0059 +/- 0.0011 |
| OOD covariate | strength 1.5 cap 12 | 0.0937 +/- 0.0103 | -0.0011 +/- 0.0004 | -0.0228 +/- 0.0012 | 0.0060 +/- 0.0012 |
| OOD effect-size | strength 1.0 cap 8 | 0.0064 +/- 0.0004 | 0.0001 +/- 0.0001 | -0.0017 +/- 0.0001 | 0.0005 +/- 0.0001 |
| OOD effect-size | strength 1.5 cap 8 | 0.0113 +/- 0.0006 | 0.0001 +/- 0.0001 | -0.0030 +/- 0.0001 | 0.0007 +/- 0.0001 |
| OOD effect-size | strength 1.5 cap 12 | 0.0113 +/- 0.0006 | 0.0001 +/- 0.0001 | -0.0030 +/- 0.0001 | 0.0008 +/- 0.0001 |
| OOD combined | strength 1.0 cap 8 | 0.0850 +/- 0.0057 | -0.0001 +/- 0.0002 | -0.0185 +/- 0.0005 | 0.0031 +/- 0.0005 |
| OOD combined | strength 1.5 cap 8 | 0.1033 +/- 0.0054 | -0.0004 +/- 0.0002 | -0.0226 +/- 0.0005 | 0.0034 +/- 0.0005 |
| OOD combined | strength 1.5 cap 12 | 0.1040 +/- 0.0057 | -0.0005 +/- 0.0002 | -0.0227 +/- 0.0006 | 0.0035 +/- 0.0006 |

## In-domain Stratified Gates

All three sweep settings preserved the in-domain stratified acceptance gates
for coefficient identity, design-information stratum, and prevalence stratum.
The rare-prevalence rows are the most sensitive:

| Setting | Rare coverage 95 | Rare rank mean error | Rare rank variance error | Seed passes: coverage / mean / variance |
| --- | ---: | ---: | ---: | --- |
| strength 1.0 cap 8 | 0.9137 +/- 0.0037 | 0.0061 +/- 0.0008 | 0.0044 +/- 0.0017 | 5/5 / 5/5 / 5/5 |
| strength 1.5 cap 8 | 0.9145 +/- 0.0037 | 0.0059 +/- 0.0008 | 0.0040 +/- 0.0016 | 5/5 / 5/5 / 5/5 |
| strength 1.5 cap 12 | 0.9145 +/- 0.0037 | 0.0059 +/- 0.0008 | 0.0040 +/- 0.0016 | 5/5 / 5/5 / 5/5 |

## Decision

The stronger OOD multiplier settings improve OOD coefficient coverage and
rank-variance error while preserving in-domain gates, but none qualifies as the
default calibration path.

The best OOD coverage in this sweep is still far below the acceptance gate:

- covariate shift: `0.7321 +/- 0.0383` at strength 1.5, cap 12
- effect-size shift: `0.7581 +/- 0.0018` at strength 1.5, cap 8 or 12
- combined shift: `0.5894 +/- 0.0344` at strength 1.5, cap 12

The cap-12 setting provides only negligible coverage gains over cap 8 while
allowing larger in-domain tail inflation. If a tuned v6 candidate is needed for
the next experiment, use strength 1.5 and cap 8 as the more conservative
candidate.

## Recommended Next Step

Stop increasing the fixed support-excess multiplier. The sweep shows that
stronger bounded inflation helps directionally but cannot solve OOD coverage
with the current rule.

The next implementation should add an explicit OOD calibration objective that
learns regime-aware uncertainty inflation from held-out OOD simulations while
keeping the in-domain coefficient gates as hard acceptance constraints. The
first comparison should run the conservative strength-1.5/cap-8 candidate
against the learned OOD-calibrated variant and the frozen scalar/v4/v5/v6
references.
