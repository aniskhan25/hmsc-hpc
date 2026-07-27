# Independent Source/Transfer Branch Real-Data Sensitivity

Date: 2026-07-20

LUMI job: `20029856`

Run root:
`/scratch/project_462000131/anisrahm/hmsc-hpc-runs/neural_hmsc_source_transfer_realdata_sensitivity_20260720`

Local decision artifacts:
`/private/tmp/neural_hmsc_source_transfer_realdata_sensitivity_20260720`

Purpose: apply the frozen bounded three-seed promotion gate to
`probit_source_transfer_response_affine` after its passing one-seed replay.

## Frozen Protocol

- Seeds: `20260721`, `20260722`, `20260723`
- Source branch minimum independent-simulation improvement: `0.0005`
- Transfer branch minimum independent-simulation improvement: `0.0001`
- Cross-dataset maximum Brier, log-loss, RMSE, and richness-MAE ratios: `1.0`
- Promotion rule: all three cross-dataset gates pass and at least two seeds
  produce genuine Big Spatial improvement
- Whittaker and Big Spatial R/Python parity references remained fixed
- The qualified five-seed simulated production result remained fixed
- Real held-out responses were used only by the final evaluation gate
- Slurm elapsed time: `00:31:51` on `dev-g`

## Branch Stability

The source branch was rejected on independent source-shaped simulations for all
three seeds. Whittaker therefore used identity and exactly reproduced the
scale-only predictive metrics in every run.

The transfer branch was selected on independent OOD simulations for all three
seeds and was applied only to Big Spatial.

| seed | source action | transfer slope | transfer intercept | simulated Brier ratio | simulated log-loss ratio |
| ---: | --- | ---: | ---: | ---: | ---: |
| `20260721` | identity | `1.050` | `0.050` | `0.9985` | `0.9979` |
| `20260722` | identity | `1.050` | `0.050` | `0.9985` | `0.9980` |
| `20260723` | identity | `1.025` | `0.050` | `0.9965` | `0.9967` |

## Real-Data Gates

Whittaker passed no degradation in all three seeds with Brier, log-loss, RMSE,
and richness-MAE ratios exactly `1.0`.

| seed | Big Spatial Brier ratio | log-loss ratio | RMSE ratio | richness-MAE ratio | cross-dataset gate |
| ---: | ---: | ---: | ---: | ---: | --- |
| `20260721` | `0.9954` | `0.9911` | `0.9977` | `0.9706` | pass |
| `20260722` | `0.9954` | `0.9917` | `0.9977` | `0.9718` | pass |
| `20260723` | `1.0048` | `1.0042` | `1.0024` | `1.0147` | fail |

All six dataset workflows completed and their dataset-level acceptance checks
passed. Whittaker coefficient SBC also remained stable: 95% coverage ranged
from `0.9529` to `0.9562`, rank mean from `0.4920` to `0.4982`, and rank
variance from `0.0690` to `0.0712`.

The qualified Python MCMC reference retained lower Brier and log loss in both
datasets for all three seeds.

## Decision

Do not promote `probit_source_transfer_response_affine`.

The run met the genuine-improvement count (`2/3`) but failed the required
cross-dataset stability condition (`2/3`). The aggregate decision was
`inspect_seed_level_no_degradation`.

This is a simulation-to-target alignment failure, not evidence that the
independent validation sample was too small. Seed `20260723` had the strongest
simulated transfer-validation gains but degraded every guarded Big Spatial
metric. Raising the existing scalar improvement margin would retain that seed
and can reject useful candidates without making the gate more target-relevant.

## Next Step

Implement a target-context-conditioned simulation gate for the transfer branch.
Construct independent synthetic calibration and validation responses on the
unlabeled Big Spatial covariate, design, community-size, and prevalence context;
require a candidate to pass both the existing generic OOD gate and this
target-shaped gate before applying it. The target held-out responses must remain
unavailable to fitting and selection. Replay the selector against the three
frozen checkpoints before any new training run or promotion attempt.
