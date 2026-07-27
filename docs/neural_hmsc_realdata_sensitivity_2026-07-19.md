# Neural HMSC Real-Data Sensitivity

Date: 2026-07-19

Job: `20001710`

Run root:
`/scratch/project_462000131/anisrahm/hmsc-hpc-runs/neural_realdata_sensitivity_20260719`

Local aggregate copy:
`/private/tmp/neural_realdata_sensitivity_20001710`

Elapsed wall time: `1962` seconds

## Purpose

This bounded sensitivity check tested whether the promoted
`external_monotone` real-data result was stable across seeds before returning
to simulated neural competitor development. It deliberately did not tune
thresholds or add a new calibration objective.

Each seed ran:

1. Whittaker promoted-default requalification with Whittaker direct R/Python
   parity metrics attached.
2. Big Spatial frozen-transfer validation using the matching Whittaker
   artifact and Big Spatial direct R/Python parity metrics attached.

Seeds: `20260721`, `20260722`, `20260723`

## Aggregate Decision

Decision: `stable_return_to_competitor_development`

All six dataset-seed rows completed and passed their acceptance gates. The
paired pass count was `3/3`. The qualified Python MCMC comparator retained the
proper-score advantage in all three paired seeds, so the neural result should
be interpreted as stable predictive transfer with qualified comparator
provenance, not as Python-only HMSC posterior equivalence or proper-score
superiority over MCMC.

## Whittaker Summary

| metric | value |
| --- | ---: |
| acceptance pass count | `3/3` |
| predictive vs uncalibrated Brier ratio | `1.0046` |
| predictive vs qualified MCMC Brier ratio | `1.0386` |
| predictive vs uncalibrated log-loss ratio | `1.0007` |
| predictive vs qualified MCMC log-loss ratio | `1.0576` |
| predictive minus qualified MCMC macro AUC | `0.0019` |
| predictive vs qualified MCMC prevalence MAE ratio | `1.1092` |
| predictive vs qualified MCMC richness MAE ratio | `1.1133` |
| coefficient SBC coverage | `0.9550` |
| coefficient SBC rank mean | `0.4943` |
| coefficient SBC rank variance | `0.0702` |
| MCMC Brier advantage count | `3/3` |
| MCMC log-loss advantage count | `3/3` |

## Big Spatial Summary

| metric | value |
| --- | ---: |
| acceptance pass count | `3/3` |
| predictive vs uncalibrated Brier ratio | `1.0257` |
| predictive vs qualified MCMC Brier ratio | `1.0945` |
| predictive vs uncalibrated log-loss ratio | `1.0312` |
| predictive vs qualified MCMC log-loss ratio | `1.0866` |
| predictive minus qualified MCMC macro AUC | `-0.0167` |
| predictive vs qualified MCMC prevalence MAE ratio | `1.4469` |
| predictive vs qualified MCMC richness MAE ratio | `1.4147` |
| MCMC Brier advantage count | `3/3` |
| MCMC log-loss advantage count | `3/3` |

## Seed-Level Results

| seed | dataset | passed | Brier vs uncal | Brier vs MCMC | log loss vs uncal | log loss vs MCMC | AUC delta vs MCMC |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `20260721` | Whittaker | yes | `1.0083` | `1.0622` | `1.0048` | `1.0826` | `0.0056` |
| `20260721` | Big Spatial | yes | `1.0266` | `1.0817` | `1.0317` | `1.0740` | `-0.0104` |
| `20260722` | Whittaker | yes | `1.0019` | `1.0225` | `0.9983` | `1.0383` | `0.0031` |
| `20260722` | Big Spatial | yes | `1.0253` | `1.1107` | `1.0314` | `1.0978` | `-0.0234` |
| `20260723` | Whittaker | yes | `1.0037` | `1.0311` | `0.9989` | `1.0518` | `-0.0028` |
| `20260723` | Big Spatial | yes | `1.0251` | `1.0910` | `1.0306` | `1.0881` | `-0.0162` |

## Interpretation

The promoted `external_monotone` path is stable enough for the current
real-data qualification scope: it passes Whittaker and Big Spatial transfer
gates across three seeds with qualified Python-only/R-boundary comparator
metadata attached.

The main remaining gap is not gate instability. It is proper-score performance:
qualified Python MCMC remains stronger on Brier and log loss for both real
datasets across all seeds. The next roadmap step should therefore return to
simulated neural competitor development with the specific aim of improving
real-data proper scores without weakening the already qualified calibration and
transfer gates.
