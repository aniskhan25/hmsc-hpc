# Transfer-Aware Predictive Mean Real-Data Pair

Date: 2026-07-20

LUMI job: `20026496`

Run root:
`/scratch/project_462000131/anisrahm/hmsc-hpc-runs/neural_hmsc_transfer_response_realdata_pair_20260720`

Local decision artifacts:
`/private/tmp/neural_hmsc_transfer_response_realdata_pair_20260720`

Purpose: evaluate the simulated-production-qualified
`probit_transfer_response_affine` predictive-only layer on Whittaker and the
dependent Big Spatial transfer dataset, with qualified Python-only HMSC parity
metrics attached and a frozen cross-dataset no-degradation gate.

## Configuration

- Seed: `20260721`
- Predictive mean policy: `apply_selected`
- Whittaker train/calibration/SBC datasets: `512/128/128`
- Whittaker SBC draws: `512`
- Predictive mean source-validation datasets: `128`
- Transfer-validation regimes: `covariate_shift`, `effect_size_shift`,
  `combined_shift`
- Transfer-validation observations: `384,000`
- Coefficient calibration: promoted `external_monotone`
- Whittaker parity reference: passed direct R/Python Whittaker parity
- Big Spatial parity reference: passed direct R/Python spatial parity
- Simulated gate: five-seed production result from job `20023454`
- Slurm elapsed time: `00:11:31` on `dev-g`

The affine candidate was selected without using either real held-out response.
It used slope `1.025` and intercept `0.025`. Source-validation Brier/log-loss
ratios were `0.9997/0.9992`; combined transfer-validation ratios were
`0.9984/0.9980`. All three transfer-regime labels were retained in metadata.

## Cross-Dataset Gate

| dataset | Brier ratio | log-loss ratio | RMSE ratio | richness MAE ratio | passed |
| --- | ---: | ---: | ---: | ---: | --- |
| Whittaker | `1.0049` | `1.0015` | `1.0024` | `0.9915` | no |
| Big Spatial | `0.9969` | `0.9948` | `0.9984` | `0.9845` | yes |

The frozen simulated gate passed, with Brier gain `0.000619` and log-loss gain
`0.003189`. The real-data mean log-loss gain was positive (`0.000320`), but the
mean Brier gain was negative (`-0.000112`). The complete cross-dataset gate
therefore failed.

Whittaker scale-only versus transfer-aware candidate:

- Brier: `0.078496` to `0.078879`
- log-loss: `0.283599` to `0.284032`
- predictive RMSE: `0.280171` to `0.280855`
- richness MAE: `4.0801` to `4.0453`

Big Spatial scale-only versus transfer-aware candidate:

- Brier: `0.051323` to `0.051162`
- log-loss: `0.205552` to `0.204480`
- predictive RMSE: `0.226545` to `0.226191`
- richness MAE: `4.8460` to `4.7709`

Both dataset-level neural qualification workflows passed, and Whittaker
coefficient SBC coverage was `0.9559`. The rejection is specifically a
predictive-mean cross-dataset failure, not a coefficient calibration or parity
failure. The qualified Python MCMC comparator remained stronger on Brier and
log-loss for both datasets.

## Decision

Do not promote the globally applied `probit_transfer_response_affine` and do
not run a three-seed sensitivity confirmation. It improves the intended Big
Spatial transfer domain but degrades source-like Whittaker despite passing
independent simulated source and transfer selection gates.

The next implementation should split predictive-mean calibration into
independently selected source and transfer branches. The source branch must use
only source-shaped simulated validation and should remain identity unless it
clears a material validation margin. The transfer branch should use the three
OOD regimes and may be applied only to an explicitly transfer-labelled
context. Real Whittaker and Big Spatial heldouts remain final evaluation data,
not selector training data.
