# Neural HMSC Rank-Aware Version 4 Comparison

Date: 2026-07-09

## Scope

This run evaluates conditional calibration version 4 against the frozen
five-seed scalar baseline and the failed version 3 conditional calibrator.
Version 4 adds prevalence-weighted analytic rank penalties and support-aware
log-scale fallback to the scalar multiplier.

Each run loaded the exact scalar checkpoint. Uncalibrated diagnostic rows
matched the scalar runs exactly, so all differences are attributable to
calibration.

Configuration:

- implementation commit: `5b33f28`
- seeds: `20260626`, `20260627`, `20260628`, `20260629`, `20260630`
- shape: 40 sites, 75 species, 3 fixed-effect coefficients
- 128 calibration datasets per seed
- 128 SBC datasets and 512 posterior draws per domain
- OOD regimes: covariate shift, effect-size shift, and combined shift
- 400 calibration epochs, learning rate `0.03`, regularization `0.001`
- rank penalty weight `0.02`
- rare/intermediate/common weights `4.0/2.0/1.0`
- support quantile `0.99`, fallback strength `2.0`

LUMI jobs `19810746`, `19810747`, `19810816`, `19810817`, and `19810885`
all completed with exit code 0. Artifacts are under
`/scratch/project_462000131/anisrahm/hmsc-hpc-runs/neural_hmsc_rankaware_v4_5b33f28_frozen_seed_<seed>/benchmark`.

## Overall Comparison

Values are five-seed mean `+/-` sample standard deviation.

| Domain | Method | Coverage 95 | Rank mean error | Rank variance error |
|---|---|---:|---:|---:|
| In-domain | scalar | 0.9420 +/- 0.0034 | 0.0039 +/- 0.0015 | 0.0243 +/- 0.0015 |
| In-domain | version 3 | 0.9440 +/- 0.0006 | 0.0043 +/- 0.0012 | 0.0055 +/- 0.0005 |
| In-domain | version 4 | 0.9439 +/- 0.0015 | 0.0014 +/- 0.0007 | 0.0077 +/- 0.0004 |
| OOD covariate | scalar | 0.5987 +/- 0.0161 | 0.0219 +/- 0.0072 | 0.0562 +/- 0.0031 |
| OOD covariate | version 3 | 0.4628 +/- 0.0138 | 0.0266 +/- 0.0068 | 0.0950 +/- 0.0023 |
| OOD covariate | version 4 | 0.5963 +/- 0.0150 | 0.0229 +/- 0.0069 | 0.0622 +/- 0.0027 |
| OOD effect-size | scalar | 0.7597 +/- 0.0069 | 0.0021 +/- 0.0008 | 0.0330 +/- 0.0018 |
| OOD effect-size | version 3 | 0.6651 +/- 0.0024 | 0.0021 +/- 0.0010 | 0.0622 +/- 0.0004 |
| OOD effect-size | version 4 | 0.6882 +/- 0.0021 | 0.0040 +/- 0.0011 | 0.0570 +/- 0.0003 |
| OOD combined | scalar | 0.4927 +/- 0.0170 | 0.0187 +/- 0.0056 | 0.0802 +/- 0.0034 |
| OOD combined | version 3 | 0.3106 +/- 0.0096 | 0.0214 +/- 0.0056 | 0.1192 +/- 0.0013 |
| OOD combined | version 4 | 0.4726 +/- 0.0146 | 0.0191 +/- 0.0055 | 0.0869 +/- 0.0027 |

Version 4 recovers most of the version 3 OOD degradation but does not improve
over the scalar baseline. Effect-size shift remains particularly weak because
the current calibration features assign it high in-domain support.

## In-Domain Strata

The pass counts apply coverage `>= 0.90`, rank-mean error `<= 0.025`, and
rank-variance error `<= 0.015`.

| Stratum | Coverage 95 | Rank mean error | Rank variance error | Seed passes: coverage / mean / variance |
|---|---:|---:|---:|---:|
| Overall | 0.9439 +/- 0.0015 | 0.0014 +/- 0.0007 | 0.0077 +/- 0.0004 | 5/5 / 5/5 / 5/5 |
| Prevalence: rare | 0.9372 +/- 0.0065 | 0.0917 +/- 0.0027 | 0.0020 +/- 0.0014 | 5/5 / 0/5 / 5/5 |
| Prevalence: intermediate | 0.9428 +/- 0.0031 | 0.0643 +/- 0.0023 | 0.0086 +/- 0.0011 | 5/5 / 0/5 / 5/5 |
| Prevalence: common | 0.9443 +/- 0.0014 | 0.0184 +/- 0.0017 | 0.0092 +/- 0.0004 | 5/5 / 5/5 / 5/5 |
| Coefficient: Intercept | 0.9663 +/- 0.0008 | 0.0018 +/- 0.0017 | 0.0189 +/- 0.0009 | 5/5 / 5/5 / 0/5 |
| Coefficient: x1 | 0.9324 +/- 0.0028 | 0.0016 +/- 0.0007 | 0.0019 +/- 0.0011 | 5/5 / 5/5 / 5/5 |
| Coefficient: x2 | 0.9329 +/- 0.0028 | 0.0050 +/- 0.0018 | 0.0023 +/- 0.0010 | 5/5 / 5/5 / 5/5 |
| Design information: low | 0.9418 +/- 0.0016 | 0.0028 +/- 0.0020 | 0.0060 +/- 0.0011 | 5/5 / 5/5 / 5/5 |
| Design information: intermediate | 0.9605 +/- 0.0015 | 0.0015 +/- 0.0010 | 0.0147 +/- 0.0008 | 5/5 / 5/5 / 3/5 |
| Design information: high | 0.9293 +/- 0.0045 | 0.0019 +/- 0.0010 | 0.0024 +/- 0.0014 | 5/5 / 5/5 / 5/5 |

Version 4 fixes the version 3 rare-coverage failure (`0.8718` to `0.9372`) and
reduces rare/intermediate rank-mean errors (`0.1291/0.0823` to
`0.0917/0.0643`). Both rank-mean strata still fail in every seed. Intercept
rank variance also fails in every seed.

## Support Diagnostics

| Domain | Mean trust | Fraction below 0.5 |
|---|---:|---:|
| In-domain | 0.9912 +/- 0.0010 | 0.0098 +/- 0.0015 |
| OOD covariate | 0.3236 +/- 0.0107 | 0.6839 +/- 0.0119 |
| OOD effect-size | 0.9544 +/- 0.0043 | 0.0532 +/- 0.0048 |
| OOD combined | 0.2915 +/- 0.0115 | 0.7181 +/- 0.0121 |

The support gate correctly detects covariate and combined shifts and recovers
the scalar behavior for most coefficients. It does not detect effect-size shift
because prevalence, design information, and raw posterior scale remain mostly
inside calibration support.

## Decision

Version 4 is not qualified. Two iterations of coefficient-scale correction
have failed the prevalence rank-mean gates. Further scale inflation would trade
that directional rank bias for overcoverage and collapsed rank variance.

The next Milestone 12 step is the reserved probit-aware IRLS/Laplace encoder
anchor. It must improve posterior means during amortized inference rather than
post-hoc calibration, preserve the frozen predictive path, and add posterior
mean magnitude to support diagnostics for effect-size shift. The same frozen
five-seed comparison remains the acceptance test.

Implementation status: the anchor and version 5 support diagnostics are
implemented in checkpoint version `0.3`. Validation must retrain anchored
candidates on the same seeded corpora; frozen legacy checkpoints remain the
comparison reference.
