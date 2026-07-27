# Neural-HMSC Milestone 54 Qualification Preregistration

Date: 2026-07-22
Protocol: `neural_hmsc_variable_design_m54_v1_1`

Revision note: the initial `v1` draft assigned production blocks 61M-69M. A
unit test generated the candidate 61M training simulations in memory while
checking factorial balance, before any model training or metric inspection.
No calibration or evaluation block was touched. To preserve the generation
barrier without ambiguity, all 61M-69M blocks are retired and must never be
used. Revision `v1_1` moves every role to untouched 101M-109M blocks and tests
the pure schedule without constructing production simulations.

## Scope

Qualify one target-agnostic variable-design fixed-effect probit checkpoint over
12-128 sites, 2-100 species, and 2-8 numerical design columns with one leading
intercept. This milestone does not add traits, random effects, spatial effects,
other likelihoods, target-outcome routing, or dataset-specific calibration.

`neural_hmsc_v0_1` and `neural_hmsc_variable_probit_v1` remain immutable. The
candidate is experimental until every candidate and sensitivity gate passes.

## Frozen Roles And Seeds

Each corpus contains 243 communities. Seed blocks are contiguous and mutually
disjoint.

| Role | Training start | Calibration start | Reserved evaluation start | Model seed |
| --- | ---: | ---: | ---: | ---: |
| candidate | 101000001 | 102000001 | 103000001 | 101900001 |
| sensitivity_a | 104000001 | 105000001 | 106000001 | 104900001 |
| sensitivity_b | 107000001 | 108000001 | 109000001 | 107900001 |

Sensitivity outcomes may confirm or reject the predeclared candidate but may
not select, tune, replace, or recalibrate it. Every role uses 40 epochs, batch
size 9, learning rate `0.001`, MSE weight `0.25`, and 256 SBC draws. The
coefficient calibration is one 95% finite-sample split-conformal scalar fitted
only on that role's calibration block.

Production train/calibration generation requires a role-specific confirmation
token. Reserved evaluation requires a second role-specific token and a valid
frozen train/calibration manifest. Evaluation data cannot be constructed before
that validation succeeds.

## Simulation Design

Every production corpus is the complete factorial:

- sites: 12, 40, 128;
- species: 2, 20, 100;
- covariates: 2, 5, 8;
- target design condition: 2, 10, 50.

Each of the 81 base cells occurs three times. Within every base cell, rare,
balanced, and common intercept regimes each occur once; weak, moderate, and
strong coefficient scales each occur once. Design matrices are generated with
a centered orthogonal basis and fixed singular-value schedules, not selected
after simulation. Actual condition numbers and all stratum labels are recorded.

## Frozen Gates

Every role must pass all gates:

- checkpoint mean/scale roundtrip maximum delta at most `1e-6`;
- exact factorial cell count of three and complete marginal balance;
- overall 95% Beta coverage in `[0.925, 0.975]`;
- absolute normalized rank-mean error from `0.5` at most `0.025`;
- absolute normalized rank-variance error from `1/12` at most `0.025`;
- neural Beta RMSE no greater than `1.05` times IRLS/Laplace-anchor RMSE;
- neural Brier and log-loss no greater than `1.02` times anchor scores;
- held-out neural/MCMC Brier and log-loss ratios no greater than `1.10` on six
  preselected evaluation contexts;
- every reported covariate-count, coefficient-role, site, species, and
  design-condition stratum has coverage in `[0.90, 0.99]`, rank-mean error at
  most `0.05`, and rank-variance error at most `0.04`;
- no target ecological outcome is used for training, calibration, selection,
  routing, or threshold changes;
- both immutable baseline content hashes remain exact.

All three roles must pass before Whittaker or Big Spatial evaluation. Real-data
gates remain those in the post-M53A scope decision: each proper-score ratio is
at most `1.10` versus qualified MCMC and degrades by no more than `0.02` versus
the applicable frozen neural baseline.

## Disposable Smoke

The disposable smoke uses only blocks `91000001`, `92000001`, and `93000001`,
27 communities per phase, one epoch, batch size 9, and 32 SBC draws. Its
orthogonal schedule balances every marginal level but is not the production
factorial. It runs no MCMC and cannot qualify, select, or modify a candidate.

Smoke acceptance checks deterministic generation, seed disjointness, marginal
balance, finite training history, split-conformal packaging, checkpoint hash
roundtrip, finite evaluation summaries, gate-report construction, and
`production_seed_opened=false`. Statistical production gates are reported for
diagnostics but are not smoke acceptance criteria.

## Outcomes

Three simulated passes allow frozen Whittaker/Big Spatial evaluation. Passing
those real-data gates freezes a separate
`neural_hmsc_variable_design_probit_v2` baseline. A failed production role is
handled by the existing one-redesign/one-fresh-evaluation stop rule. It cannot
start a post-hoc calibration search or modify either existing baseline.

## Disposable Smoke Result

The final `v1_1` smoke used only the 91M-93M disposable blocks and returned
`smoke_passed`. All eight plumbing checks passed, both immutable baseline hashes
validated, and `production_seed_opened=false`. The split-conformal multiplier
was `0.911985404081571` over 5,490 calibration coefficients. Checkpoint hashes:

- weights: `f8b6b81a7bca29e60337e4903c235ff84340f1c81d28e195e15ae834a7a222ba`;
- calibration: `7b35310860e7329c4a7b6b0f022c7f30e384c53e05053534428fe4637c5061ab`;
- manifest: `5072ed61574d65e91fb3b2fceabce3c652edd0c3ed206f486655e5fea0cb5894`;
- smoke report: `4f4e6ef4cff276039a1ddd6f80af09066fd918cfbe407bbac63dc5d8de99e775`.

Aggregate smoke coverage was `0.952459`, rank mean `0.499613`, and rank
variance `0.077149`. All aggregate statistical diagnostics passed; one compact
species-count stratum rank diagnostic failed. Stratum gates are deliberately
nonpromotional in the 27-community smoke and remain unchanged for production.
Two seeded disposable reruns produced identical weight, calibration, training,
and evaluation results.

## Candidate Train/Calibration Freeze

LUMI `dev-g` job `20129822` completed candidate `train-calibrate` in `00:04:07`
using exact confirmation `GENERATE_M54_CANDIDATE_TRAIN_CALIBRATION`. It opened
only the preregistered 101000001-101000243 training block and
102000001-102000243 calibration block. The fitted split-conformal
coefficient-posterior multiplier is `0.9808582145420995` over 49,410
calibration coefficients. Both immutable baseline hashes remained exact.

The wrapper ran the unchanged harness `validate_freeze()` after fitting. Every
protocol, role, disjointness, baseline, and checkpoint-hash check passed. The
downloaded bundle reproduced the hashes locally and passed an independent
checkpoint load. Frozen hashes are:

- freeze: `021488d1868b773232112bfa9199aad74602e26ef119bcd7a7f38bb2ea90728e`;
- checkpoint manifest: `d735ce55c95bddb9df56992d1b6be7f3d8f4f95ae602397e524485997b017df4`;
- weights: `d6f5923873e77c63e51f8b17a65a91e667b97d02fb4681457d7c08d57fbab52b`;
- calibration: `21041be868e38f4d1209f56a8e42336c6b61d54fd2243874f76d5cd1d82da88f`;
- post-freeze validation:
  `1e208014d960ec888584b8f90692c39c320cb6a70df9c414c3be01ae956bef7f`.

The durable run root is
`/scratch/project_462000131/anisrahm/hmsc-hpc-runs/neural_hmsc_m54_candidate_train_calibration_20129822`.
Its status is `frozen_before_reserved_evaluation`, and
`reserved_evaluation_opened=false`.

## Candidate Reserved-Evaluation Result

LUMI `dev-g` job `20134138` opened the one-shot 103000001-103000243 block using
exact confirmation `OPEN_M54_CANDIDATE_EVALUATION`. It completed in `00:03:27`
against the unchanged candidate freeze. The report SHA-256 is
`a0c2ef66365a9a8875ae123941fe8f08f5bcb2c84ababa5581844374a5d2bdbd`.

The frozen decision is `variable_design_role_failed`. All calibration, rank,
stratum, checkpoint, balance, baseline, coefficient-RMSE, and MCMC-reference
gates passed. The candidate failed response-scale no-degradation versus the
IRLS/Laplace anchor: Brier ratio `1.047000` and log-loss ratio `1.036589`
exceeded their `1.02` limits. The candidate is not qualified. Sensitivity
blocks 104M-109M remain unopened and cannot rescue or replace it. Detailed
results and the bounded redesign decision are in
`docs/neural_hmsc_m54_candidate_negative_result_2026-07-22.md`.
