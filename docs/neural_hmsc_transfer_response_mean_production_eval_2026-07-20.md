# Transfer-Aware Predictive Mean Production Evaluation

Date: 2026-07-20

LUMI job: `20023454`

Run root:
`/scratch/project_462000131/anisrahm/hmsc-hpc-runs/neural_hmsc_transfer_response_mean_production_eval_20260720`

Local result copy:
`/private/tmp/neural_hmsc_transfer_response_mean_production_eval_20260720`

Purpose: evaluate `probit_transfer_response_affine` against the frozen,
previously qualified production-shape `external_monotone` baseline. This run
replaces the underqualified compact checkpoint as the decision basis for the
transfer-aware predictive-only mean competitor.

## Evaluation Shape

- Seeds: `20260716`, `20260717`, `20260718`, `20260719`, `20260720`
- Shape: `40 x 75`
- SBC datasets/draws: `32 x 256`
- External monotone calibration datasets: `4`
- Predictive mean validation datasets: `8`
- OOD regimes: `covariate_shift`, `effect_size_shift`, `combined_shift`
- Baseline: frozen qualified `external_monotone`
- Candidate: `external_monotone + probit_transfer_response_affine`
- Partition: `dev-g`
- Wall time: `222` seconds reported by the harness; Slurm elapsed time `00:05:03`

Each candidate arm reused the corresponding frozen scalar checkpoint and the
same seed, external-monotone settings, and fixed SBC/OOD rows as the baseline.

## Frozen Coefficient Gates

| run | accepted seeds | in-domain 95% | rare 95% | mean OOD 95% | worst OOD 95% | effect-size 95% | combined 95% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `external_monotone` | baseline | `0.9442` | `0.9520` | `0.9203` | `0.8214` | `0.8214` | `0.9684` |
| `external_monotone_response` | `5/5` | `0.9442` | `0.9520` | `0.9203` | `0.8214` | `0.8214` | `0.9684` |

The coefficient posterior, coefficient calibration, SBC ranks, OOD coverage,
and rare-validation metrics are identical between arms. The transfer-aware
affine layer remains explicitly predictive-only.

## Predictive Proper Scores

| run | Brier | Brier ratio | log-loss | log-loss ratio | RMSE | RMSE ratio | prevalence MAE | richness MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `external_monotone` | `0.147620` | `1.0000` | `0.446577` | `1.0000` | `0.384179` | `1.0000` | `0.043553` | `3.023169` |
| `external_monotone_response` | `0.147001` | `0.9957` | `0.443388` | `0.9928` | `0.383366` | `0.9979` | `0.042373` | `2.988936` |

Per-seed candidate ratios versus scale-only:

| seed | Brier | log-loss | RMSE | prevalence MAE | richness MAE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `20260716` | `0.9934` | `0.9907` | `0.9967` | `0.9572` | `0.9323` |
| `20260717` | `0.9954` | `0.9907` | `0.9977` | `1.0150` | `0.9836` |
| `20260718` | `1.0005` | `0.9988` | `1.0002` | `1.0121` | `1.0310` |
| `20260719` | `0.9964` | `0.9939` | `0.9982` | `0.9564` | `1.0003` |
| `20260720` | `0.9931` | `0.9899` | `0.9966` | `0.9234` | `0.9936` |

Brier improved in four of five seeds, log-loss improved in all five, and RMSE
improved in four of five. Seed `20260718` had small Brier/RMSE degradation and
the largest richness degradation, but the mean remained better than scale-only
for every reported predictive metric.

Compared with the earlier source-only `probit_response_affine` production run,
the transfer-aware candidate has better mean Brier, log-loss, RMSE, and
prevalence MAE. Its richness MAE is slightly worse (`2.9889` versus `2.9822`),
while remaining better than scale-only (`3.0232`).

## Calibration Metadata

The transfer-aware candidate was selected in all five seeds. Every selected
fit used slope `1.15`; intercept was `-0.025` for the first three seeds and
`0.0` for the last two. Independent source-validation ratios ranged from
`0.9907` to `0.9942` for Brier and `0.9852` to `0.9892` for log-loss.
Transfer-validation ratios ranged from `0.9598` to `0.9655` for Brier and
`0.9428` to `0.9484` for log-loss, using `24,000` observations per seed and
retaining labels for all three OOD regimes.

## Decision

`probit_transfer_response_affine` qualifies as a simulated production-shape
predictive-only competitor. It preserves all frozen coefficient gates and
delivers a small but stable average proper-score improvement. It is not yet a
default deployment policy because real ecological transfer remains untested
and one simulated seed has small predictive degradation.

The next gate is paired Whittaker and Big Spatial real-data validation against
promoted scale-only `external_monotone`, with Python-only HMSC parity metrics
attached and a frozen cross-dataset no-degradation decision rule.
