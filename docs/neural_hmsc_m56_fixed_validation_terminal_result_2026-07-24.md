# Neural-HMSC Milestone 56 Fixed Validation Terminal Result

Date: 2026-07-24

Protocol: `neural_hmsc_fixed_probit_covariance_m56_v1`

LUMI job: `20192218`

Decision: `m56_terminal_failure_reserved_evaluation_sealed`

## Execution

The one-shot production train-validation was authorized with
`GENERATE_M56_CORRELATION_TRAIN_VALIDATION`. LUMI `dev-g` job `20192218`
completed successfully in `00:02:39` with exit code `0:0` and peak batch-step
RSS `1851428K`.

The run opened only:

- correlation training seeds `211000001` through `211000324`;
- fixed validation seeds `212000001` through `212000324`;
- model seed `211900001`.

The downloaded run bundle contains no `213M`, `214M`, or `215M` seed or
evaluation-role record. Reserved evaluation remains sealed.

## Independent Validation

The complete run was downloaded to:

`/private/tmp/neural_hmsc_m56_train_validation_20192218`

The independent validator:

- revalidated the frozen preregistration and artifact/seed-audit hashes;
- revalidated the exact v0.1 release, checkpoint, weights, and calibration
  binding;
- loaded the overlay against the immutable local v0.1 release;
- verified exact 211M and 212M seed ranges and the 100-epoch training record;
- recomputed every gate from `validation_metrics`;
- reproduced all 123 stored gate decisions and all 47 failures;
- verified `freeze.json`, post-validation, overlay-manifest, and overlay-weight
  hashes;
- ran a disposable 292M inference check and verified finite positive-definite
  covariance.

All 20 independent operational/provenance checks passed.

Artifact hashes:

- `freeze.json`:
  `c4fcb04cf1ebd7123be12144803de319ce1ff16a31e4fc5a1fb3e224f361a526`;
- `postfreeze_validation.json`:
  `9927084c1549a14e13836911d1bf1595f137bf678b182c6943ae00636586ce25`;
- overlay manifest:
  `24f7eafa4a886afab94711bab77c56e76aef726fc93c0911c372b639bfa0121d`;
- overlay weights:
  `66033d4f84cd443abf94053923e929180c0307fb08ac2a1bb9eaa75fe32ccde5`;
- independent validation report:
  `e91de4f72b4a672fe6f61223e13f916086d756bd23e7282b6727545add5d1149`.

## Fixed Validation Result

Operational invariants passed:

- maximum mean delta versus calibrated v0.1: `0.0`;
- maximum marginal-scale delta versus calibrated v0.1: `0.0`;
- minimum covariance eigenvalue: `0.000430281`;
- maximum absolute correlation: `0.832500`;
- candidate/raw-Laplace joint-NLL ratio: `0.496555`;
- heldout Brier ratio versus diagonal v0.1: `1.002273`;
- heldout log-loss ratio versus diagonal v0.1: `1.006187`;
- mean absolute Fisher-z movement: `0.510867`.

The fixed statistical gates failed materially:

| Metric | Observed | Frozen gate |
| --- | ---: | ---: |
| Marginal 95% coverage | `0.826955` | `[0.925, 0.975]` |
| Joint ellipse coverage | `0.735021` | `[0.925, 0.975]` |
| Candidate/diagonal joint NLL | `1.169488` | `<= 0.99` |
| Radial-rank mean | `0.600648` | error from `0.5 <= 0.025` |
| Radial-rank variance | `0.121397` | error from `1/12 <= 0.025` |

There were 47 failed gates:

- 5 aggregate gates;
- 11 stratum marginal-coverage gates;
- 12 stratum joint-coverage gates;
- 9 stratum joint-NLL gates;
- 8 stratum radial-rank-mean gates;
- 1 stratum marginal-rank-mean gate;
- 1 stratum marginal-rank-variance gate.

The strongest failures occurred for strong effects, common prevalence,
non-centered predictor locations, and predictor scale `0.5`. For example,
common-prevalence marginal coverage was `0.738`, joint coverage was `0.583`,
and candidate/diagonal joint NLL was `1.247`.

## Interpretation

The learned correlation head substantially improved joint NLL relative to the
raw Laplace correlation, but it did not beat the diagonal v0.1 comparator and
could not repair marginal undercoverage because the protocol required v0.1
means and marginal scales to remain unchanged. The failed marginal gate is
therefore a prerequisite failure for this overlay-only family, not a reason to
retune correlation.

Under the frozen stop rule:

- do not open `213M` through `215M`;
- do not run MCMC subsets or real-data replay;
- do not retune the head, bounds, loss, simulator, marginal scale, or gates
  within Milestone 56;
- retain `neural_hmsc_v0_1` as the qualified neural endpoint and qualified
  Python MCMC as the statistical reference.

Any future covariance attempt requires a new capability decision, a
substantially different joint-posterior representation that can model
marginals and covariance together, and a fresh preregistration with unused
seeds.
