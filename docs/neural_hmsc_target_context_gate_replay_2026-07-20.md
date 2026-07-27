# Target-Context Simulation Gate Frozen Replay

Date: 2026-07-20

LUMI job: `20031969`

Frozen source run:
`/scratch/project_462000131/anisrahm/hmsc-hpc-runs/neural_hmsc_source_transfer_realdata_sensitivity_20260720`

Replay root:
`/scratch/project_462000131/anisrahm/hmsc-hpc-runs/neural_hmsc_target_context_gate_replay_20260720`

Local decision artifacts:
`/private/tmp/neural_hmsc_target_context_gate_replay_20260720`

## Implementation

The target-context gate keeps the generic OOD-selected transfer branch frozen
and evaluates it on two additional independent synthetic pools. Both pools use:

- deterministic midpoint quantiles of all Big Spatial `TMG` covariates;
- the checkpoint's fixed `40`-site design and `75`-species community size;
- target-support, effect-size, and combined-shift regimes;
- the predeclared rare-species prevalence prior, without target `Y`;
- separate calibration and validation seeds;
- `32` simulated datasets per pool.

The transfer branch is applied only when its original generic OOD gate and both
target-shaped pools pass Brier, log-loss, and combined-score thresholds.
Big Spatial held-out `Y` is not loaded until after the selector decision. The
candidate remains predictive-only; neural weights and coefficient/predictive
calibration parameters are unchanged.

## Frozen Replay

| seed | target gate | slope | intercept | target calibration Brier/log-loss | target validation Brier/log-loss | real Brier/log-loss | real RMSE/richness | outcome |
| ---: | --- | ---: | ---: | --- | --- | --- | --- | --- |
| `20260721` | pass | `1.050` | `0.050` | `0.9990 / 0.9987` | `0.9978 / 0.9974` | `0.9954 / 0.9911` | `0.9977 / 0.9706` | improvement |
| `20260722` | pass | `1.050` | `0.050` | `0.9986 / 0.9979` | `0.9979 / 0.9974` | `0.9954 / 0.9917` | `0.9977 / 0.9718` | improvement |
| `20260723` | pass | `1.025` | `0.050` | `0.9981 / 0.9983` | `0.9978 / 0.9979` | `1.0048 / 1.0042` | `1.0024 / 1.0147` | degradation |

The job completed in `51` seconds (`25` seconds measured inside the replay).
All three target-context gates passed, but only two of three frozen decisions
passed real-data no degradation. The replay decision was
`target_context_gate_failed_no_degradation`.

## Finding

Covariate-shape conditioning does not identify the harmful checkpoint. The
failed seed improves consistently under the generic OOD simulator and under
both target-shaped simulation pools, yet moves all four real Big Spatial scores
in the wrong direction. This is a simulator-to-ecology misspecification: the
synthetic response mechanism lacks information needed to predict the sign of
the real transfer correction. More simulation rows or a stricter scalar margin
would not address that mismatch.

Do not promote the dual-gate selector and do not tune this simulation-gate
family further.

## Next Step

Implement a probability-level deep-ensemble competitor over the three frozen
checkpoints. Compare scale-only ensemble probabilities against affine-branch
ensemble probabilities on Whittaker and Big Spatial, then repeat the comparison
for all three leave-one-seed-out ensembles. Selection must be predeclared and
must not use target outcomes. Advance only if the full ensemble and every
leave-one-out ensemble preserve no degradation, with a genuine Big Spatial
proper-score gain for the full ensemble.
