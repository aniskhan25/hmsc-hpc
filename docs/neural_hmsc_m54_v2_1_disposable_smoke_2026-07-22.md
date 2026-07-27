# Neural-HMSC Milestone 54 v2.1 Disposable Smoke

Date: 2026-07-22

Status: disposable smoke passed; production seed blocks remain sealed.

## Scope

This smoke exercised the preregistered `neural_hmsc_variable_design_m54_v2_1`
implementation after the frozen three-output gated head, paired predictive
auxiliary objective, checkpoint schema `0.2`, compatibility checks, and sealed
qualification harness were implemented.

Only the disposable seed roles were opened:

- coefficient training: 191M
- predictive auxiliary contexts: 192M
- independent heldout partners: 193M
- coefficient calibration: 194M
- smoke evaluation: 195M
- model initialization: `191900001`

The production 111M-115M roles were not generated or opened. This run is not
promotion evidence and cannot be used to tune the frozen representation,
objective, thresholds, or gates.

## Immutable Inputs

- preregistration SHA-256:
  `900af8719fc73947cd7addf3b7dc9fe2f233eadbbd2bf9f37bac1286fc15e54d`
- fixed-shape v0.1 regression hash:
  `affcfe10d2f9586e97432ae07754ab829e929ffdb62f41fb71779ad5f3ed12c8`
- variable-design v1 regression hash:
  `badaf8b8244cbd850693723147c6094bf196482237c52d2cc279ce42b286d2f9`

## Result

The harness returned `smoke_passed`. All eleven disposable smoke checks passed:

- immutable baseline hashes
- packaged calibration
- checkpoint roundtrip
- finite evaluation metrics
- finite training metrics
- marginal-balance check
- independent paired heldouts
- preregistration hash
- production seeds unopened
- disjoint seed blocks
- bounded support gate

The smoke checkpoint hashes are:

- weights: `2aee31c80c241833d31c33c5a5da8e8eceed9d10c321dddaac267e05e65c32f3`
- calibration: `2c408b103372a2e8099d73dbfd663999aa442633e733cf13975d7a90b9e9ae44`
- manifest: `fb446719c56e424ec19f0a854ddc512bcd43924b248f1bebcb45df50e1a0f552`
- smoke report: `b9d725dbf5122a4df701be3565b56cfbae57bb8e3cec659b0c67db1f745b104a`

The one-epoch disposable evaluation produced coefficient coverage `0.949909`,
rank mean `0.503341`, and rank variance `0.077867`. Predictive Brier score was
`0.135677` versus anchor `0.136159`; predictive log loss was `0.423580` versus
anchor `0.425539`. Every smoke site-count and covariate-count proper-score ratio
was below `1.0`.

The gate remained bounded (`0.431357` to `0.495650`). The production-only
genuine Beta-RMSE gain and ordered support-movement diagnostics did not pass in
the one-epoch smoke. Those checks are deliberately nonpromotional here; their
values do not authorize objective changes or predict the sealed production
decision.

The machine-readable report is outside the repository at
`/private/tmp/neural_hmsc_m54_v2_1_disposable_smoke_20260722/m54_v2_1_smoke.json`.

## Decision And Next Sealed Action

The implementation is operational and the disposable smoke authorizes no
production evaluation by itself. The next step requires an explicit,
separate authorization using
`GENERATE_M54_V2_1_TRAIN_AUX_CALIBRATION`. That action may generate only the
111M coefficient-training, 112M auxiliary-context, 113M independent-heldout,
and 114M calibration blocks and freeze their artifacts. The 115M reserved
evaluation block must remain sealed until the frozen production artifact,
manifest, hashes, and seed roles are independently validated.
