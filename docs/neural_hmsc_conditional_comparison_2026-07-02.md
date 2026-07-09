# Neural HMSC Conditional Calibration Comparison

Date: 2026-07-02

## Scope

This run compares the first Milestone 12 conditional coefficient calibrator
against the frozen five-seed scalar baseline. Each conditional run loaded the
exact scalar run checkpoint. Calibration data, SBC data, posterior draws, and
random seeds were unchanged, and all uncalibrated diagnostic rows reproduced
the scalar runs exactly.

Configuration:

- implementation commit: `647aa5a`
- seeds: `20260626`, `20260627`, `20260628`, `20260629`, `20260630`
- shape: 40 sites, 75 species, 3 fixed-effect coefficients
- 128 calibration datasets per seed
- 128 SBC datasets and 512 posterior draws per domain
- OOD regimes: covariate shift, effect-size shift, and combined shift
- conditional optimizer: 400 epochs, learning rate `0.03`, regularization
  `0.001`
- no MCMC reference and no neural checkpoint retraining

LUMI jobs `19661640`, `19661641`, `19661719`, `19661720`, and `19661805` all
completed with exit code 0. Artifacts are under
`/scratch/project_462000131/anisrahm/hmsc-hpc-runs/neural_hmsc_conditional_647aa5a_frozen_seed_<seed>/benchmark`.

## Overall Comparison

Values are five-seed mean `+/-` sample standard deviation.

| Domain | Method | Coverage 95 | Rank mean error | Rank variance error | Beta mean RMSE |
|---|---|---:|---:|---:|---:|
| In-domain | scalar | 0.9420 +/- 0.0034 | 0.0039 +/- 0.0015 | 0.0243 +/- 0.0015 | 0.5903 +/- 0.0033 |
| In-domain | conditional | 0.9440 +/- 0.0006 | 0.0043 +/- 0.0012 | 0.0055 +/- 0.0005 | 0.5888 +/- 0.0033 |
| OOD covariate | scalar | 0.5987 +/- 0.0161 | 0.0219 +/- 0.0072 | 0.0562 +/- 0.0031 | 0.9600 +/- 0.0195 |
| OOD covariate | conditional | 0.4628 +/- 0.0138 | 0.0266 +/- 0.0068 | 0.0950 +/- 0.0023 | 0.9585 +/- 0.0194 |
| OOD effect-size | scalar | 0.7597 +/- 0.0069 | 0.0021 +/- 0.0008 | 0.0330 +/- 0.0018 | 1.2526 +/- 0.0045 |
| OOD effect-size | conditional | 0.6651 +/- 0.0024 | 0.0021 +/- 0.0010 | 0.0622 +/- 0.0004 | 1.2518 +/- 0.0044 |
| OOD combined | scalar | 0.4927 +/- 0.0170 | 0.0187 +/- 0.0056 | 0.0802 +/- 0.0034 | 1.4961 +/- 0.0145 |
| OOD combined | conditional | 0.3106 +/- 0.0096 | 0.0214 +/- 0.0056 | 0.1192 +/- 0.0013 | 1.4949 +/- 0.0144 |

The conditional model fixes overall in-domain rank variance without materially
changing posterior-mean RMSE. It materially worsens coverage and rank variance
under every OOD regime and therefore fails the OOD improvement gate.

## In-Domain Strata

The pass counts apply the frozen coverage (`>= 0.90`), rank-mean (`<= 0.025`),
and rank-variance (`<= 0.015`) thresholds.

| Stratum | Coverage 95 | Rank mean error | Rank variance error | Seed passes: coverage / mean / variance |
|---|---:|---:|---:|---:|
| Overall | 0.9440 +/- 0.0006 | 0.0043 +/- 0.0012 | 0.0055 +/- 0.0005 | 5/5 / 5/5 / 5/5 |
| Prevalence: rare | 0.8718 +/- 0.0069 | 0.1291 +/- 0.0032 | 0.0075 +/- 0.0020 | 0/5 / 0/5 / 5/5 |
| Prevalence: intermediate | 0.9344 +/- 0.0039 | 0.0823 +/- 0.0019 | 0.0055 +/- 0.0010 | 5/5 / 0/5 / 5/5 |
| Prevalence: common | 0.9486 +/- 0.0009 | 0.0167 +/- 0.0018 | 0.0082 +/- 0.0003 | 5/5 / 5/5 / 5/5 |
| Coefficient: Intercept | 0.9451 +/- 0.0013 | 0.0165 +/- 0.0035 | 0.0049 +/- 0.0008 | 5/5 / 5/5 / 5/5 |
| Coefficient: x1 | 0.9434 +/- 0.0025 | 0.0016 +/- 0.0007 | 0.0059 +/- 0.0008 | 5/5 / 5/5 / 5/5 |
| Coefficient: x2 | 0.9437 +/- 0.0008 | 0.0048 +/- 0.0019 | 0.0060 +/- 0.0006 | 5/5 / 5/5 / 5/5 |
| Design information: low | 0.9425 +/- 0.0037 | 0.0029 +/- 0.0026 | 0.0036 +/- 0.0014 | 5/5 / 5/5 / 5/5 |
| Design information: intermediate | 0.9603 +/- 0.0020 | 0.0053 +/- 0.0012 | 0.0124 +/- 0.0010 | 5/5 / 5/5 / 5/5 |
| Design information: high | 0.9294 +/- 0.0031 | 0.0051 +/- 0.0024 | 0.0009 +/- 0.0009 | 5/5 / 5/5 / 5/5 |

Rare rank-mean error increased from `0.1070` under scalar calibration to
`0.1291`; intermediate error increased from `0.0680` to `0.0823`. The
Gaussian log-score objective corrects conditional dispersion but does not
correct these directional rank failures.

## Decision

The first conditional architecture is not qualified. The next revision will:

- add differentiable stratum-level rank-mean and rank-variance penalties to
  the simulated calibration objective
- explicitly upweight rare and intermediate prevalence strata
- store calibration-feature support and shrink conditional adjustments back to
  the scalar multiplier outside that support
- retain the scalar method as the OOD fallback instead of extrapolating the
  learned raw-scale and information effects
- rerun the same frozen five-seed comparison before either real-data workflow

The acceptance gates and scalar baseline remain unchanged.

Implementation status: these corrections are implemented in conditional
metadata version 4 and must now be evaluated with the same frozen checkpoints.
