# Neural-HMSC Milestone 54 Candidate Result

Date: 2026-07-22
Protocol: `neural_hmsc_variable_design_m54_v1_1`
Candidate freeze job: `20129822`
Candidate evaluation job: `20134138`

## Decision

The preregistered variable-design candidate failed its one-shot 103M
evaluation. It is not qualified and must not be promoted. Sensitivity blocks
104M-109M remain unopened because sensitivity outcomes may confirm or reject a
passing predeclared candidate but may not replace or rescue a failed one.

The evaluated checkpoint was unchanged from the validated candidate freeze:

- freeze SHA-256:
  `021488d1868b773232112bfa9199aad74602e26ef119bcd7a7f38bb2ea90728e`;
- checkpoint manifest:
  `d735ce55c95bddb9df56992d1b6be7f3d8f4f95ae602397e524485997b017df4`;
- weights:
  `d6f5923873e77c63e51f8b17a65a91e667b97d02fb4681457d7c08d57fbab52b`;
- calibration:
  `21041be868e38f4d1209f56a8e42336c6b61d54fd2243874f76d5cd1d82da88f`.

The frozen evaluation report SHA-256 is
`a0c2ef66365a9a8875ae123941fe8f08f5bcb2c84ababa5581844374a5d2bdbd`.
It is stored at
`/scratch/project_462000131/anisrahm/hmsc-hpc-runs/neural_hmsc_m54_candidate_evaluation_20134138`.

## Gate Result

The candidate passed:

- exact factorial balance;
- checkpoint roundtrip;
- immutable baseline hashes;
- aggregate 95% coefficient coverage (`0.950293`);
- normalized rank mean (`0.510994`) and variance (`0.074783`);
- every frozen covariate-count, coefficient-role, site, species, and
  design-condition calibration stratum;
- coefficient posterior-mean RMSE versus the IRLS/Laplace anchor
  (`0.294699` versus `0.379008`, ratio `0.777552`);
- six-context proper scores versus qualified Python MCMC (Brier ratio
  `0.940466`, log-loss ratio `0.949325`).

It failed both aggregate response-scale no-degradation gates versus its own
IRLS/Laplace anchor:

| Metric | Neural | Anchor | Ratio | Gate |
| --- | ---: | ---: | ---: | ---: |
| Brier | 0.138438 | 0.132224 | 1.047000 | <= 1.02 |
| Log loss | 0.432107 | 0.416854 | 1.036589 | <= 1.02 |

## Diagnosis

The degradation is concentrated where the number of sites is small relative
to design width:

| Stratum | Brier ratio | Log-loss ratio |
| --- | ---: | ---: |
| 12 sites | 1.100853 | 1.066665 |
| 40 sites | 1.038796 | 1.034734 |
| 128 sites | 1.012226 | 1.012692 |
| 2 covariates | 1.008023 | 1.006108 |
| 5 covariates | 1.048007 | 1.038864 |
| 8 covariates | 1.092878 | 1.069380 |

The shared bounded residual head therefore learned coefficient-mean movement
that improves Beta RMSE and remains calibrated, but that movement is too broad
for response probabilities in low-support designs. This is not a scale
calibration failure and should not be addressed with another post-hoc scale,
cap, router, or dataset-specific selector.

## Bounded Next Step

Use the single representation-redesign allowance from the active stop rule.
Preregister a fresh variable-design `v2_1` candidate with new untouched
train/calibration/evaluation blocks. The redesign should make posterior-mean
movement an explicit convex anchor-residual mixture with a learned
coefficient-local support gate. Train the gate with an independent
response-scale probit proper-score auxiliary loss in addition to the existing
coefficient-posterior objective. The gate representation must include a smooth
sample-size-to-active-covariate support signal and anchor uncertainty; it must
permit movement in well-supported contexts and converge toward the anchor in
low-support contexts.

All existing aggregate, stratum, MCMC, baseline, and real-data gates remain
unchanged. No threshold may be selected from the opened 103M outcomes. Run a
disposable-seed smoke first, then one fresh preregistered production
train/calibration/evaluation sequence. Failure of that fresh sequence closes
Milestone 54 and leaves `neural_hmsc_variable_probit_v1` as the qualified
variable-shape endpoint.

The redesign is now frozen as protocol
`neural_hmsc_variable_design_m54_v2_1` in
`docs/neural_hmsc_m54_v2_1_redesign_preregistration_2026-07-22.md`. It reserves
111M-115M for the one production sequence and 191M-195M for disposable smoke.
No seed in either range was generated while preregistering the protocol. The
preregistration SHA-256 is
`900af8719fc73947cd7addf3b7dc9fe2f233eadbbd2bf9f37bac1286fc15e54d`.
