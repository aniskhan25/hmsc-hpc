# Independent Source/Transfer Branch Real-Data Pair

Date: 2026-07-20

LUMI job: `20029081`

Run root:
`/scratch/project_462000131/anisrahm/hmsc-hpc-runs/neural_hmsc_source_transfer_realdata_pair_20260720`

Local decision artifacts:
`/private/tmp/neural_hmsc_source_transfer_realdata_pair_20260720`

Purpose: replay the failed global-affine real-data pair with a new
`probit_source_transfer_response_affine` predictive-only artifact containing
independently fitted and selected source and transfer branches.

## Frozen Design

- Seed: `20260721`, matching the global-affine run from job `20026496`
- Source branch fit: shape-matched calibration simulations
- Source branch selection: independent shape-matched validation simulations
- Source minimum Brier+log-loss improvement: `0.0005`
- Transfer branch fit: balanced covariate/effect-size/combined-shift simulations
- Transfer branch selection: a separate seed window with the same three regimes
- Transfer minimum Brier+log-loss improvement: `0.0001`
- Transfer fit observations: `384,000`
- Transfer validation observations: `384,000`
- Context mapping: `whittaker -> source_branch`,
  `big_spatial_transfer -> transfer_branch`
- Real held-out responses used only by the final cross-dataset gate
- Qualified Whittaker and Big Spatial R/Python parity metrics attached
- Five-seed simulated production gate from job `20023454` attached
- Slurm elapsed time: `00:12:30` on `dev-g`

## Branch Decisions

The source branch did not clear its independent `0.0005` improvement margin and
was frozen to identity (`slope=1.0`, `intercept=0.0`). Whittaker therefore
reproduced scale-only metrics exactly.

The transfer branch cleared its separate OOD validation gate and selected
`slope=1.05`, `intercept=0.05`. Its independent transfer-validation ratios were
`0.9985` Brier and `0.9979` log-loss, with all three OOD regime labels retained.
Big Spatial loaded this branch from the frozen Whittaker artifact without
updating weights or calibration parameters.

## Cross-Dataset Gate

| dataset | Brier ratio | log-loss ratio | RMSE ratio | richness MAE ratio | passed |
| --- | ---: | ---: | ---: | ---: | --- |
| Whittaker | `1.0000` | `1.0000` | `1.0000` | `1.0000` | yes |
| Big Spatial | `0.9954` | `0.9911` | `0.9977` | `0.9706` | yes |

Big Spatial scale-only versus transfer branch:

- Brier: `0.051323` to `0.051087`
- log-loss: `0.205552` to `0.203721`
- predictive RMSE: `0.226545` to `0.226024`
- richness MAE: `4.8460` to `4.7036`

The real-data mean gains were `0.000118` Brier and `0.000915` log-loss. The
five-seed simulated gate also passed. The complete frozen cross-dataset gate
passed with no failure reasons.

Both dataset-level neural qualification workflows passed, and coefficient/SBC
metrics remained unchanged. The qualified Python MCMC comparator remained
stronger on core proper scores in both datasets.

## Decision

The independent source/transfer branch design passes the one-seed production
replay and may advance to the bounded three-seed real-data sensitivity gate.
It is not yet promotable: branch selection itself must be stable across seeds,
Whittaker must remain nondegrading in every seed, and at least two seeds must
show genuine Big Spatial transfer improvement under the existing aggregate
decision rule.
