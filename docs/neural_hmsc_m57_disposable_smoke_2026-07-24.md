# Neural-HMSC Milestone 57 Disposable Smoke

Date: 2026-07-24

## Scope

This run exercised only the disposable Milestone 57 roles:

- paired training contexts `391000001` through `391000027`;
- evaluation contexts `392000001` through `392000027`;
- model seed `321900001`;
- one training epoch;
- 32 posterior draws per evaluation context.

No `321M` through `325M` production, validation, reserved-evaluation, MCMC,
or real-data seed was opened. This run is plumbing evidence only and is not
promotion or calibration evidence.

The retained run root is:

`/private/tmp/neural_hmsc_m57_disposable_smoke_20260724_v2`

## Result

All operational smoke checks passed:

- frozen decision, audit, and preregistration hashes matched;
- immutable `neural_hmsc_v0_1`, variable-v1, and failed-M56 bindings matched;
- all 54 paired training realizations shared exact `X` and `Beta` within each
  owning context;
- training loss was finite;
- all posterior parameters and draws were finite;
- minimum covariance eigenvalue was `0.0010956672`;
- maximum absolute correlation was `0.01015946`;
- degrees of freedom remained between `9.80391` and `9.97595`;
- checkpoint roundtrip deltas were exactly zero for location, marginal scale,
  covariance Cholesky, Student-t scale Cholesky, and degrees of freedom;
- HDF5 output had shape `2 x 16 x 2 x 75` and retained Student-t parameter
  datasets and metadata.

The disposable artifact hashes are:

- manifest:
  `e70f983527e03b5f35d3dae9315ba16a13b1bfe3977849cc0d33a730d453ec32`;
- weights:
  `f3302b1584fec28ba1beeef216995f7bb34ee0312058d61d24e8f5d68ede6ef3`.

## Non-Promotional Diagnostics

The one-epoch candidate had marginal 95% coverage `0.79358`, joint 95%
coverage `0.78420`, and marginal 50% coverage `0.43630`. These values are not
qualification results: the smoke uses one of the preregistered 150 epochs,
27 rather than 324 evaluation contexts, and 32 rather than 512 draws. They
show that the completed harness exposes calibration and stratum failures
instead of treating operational success as statistical qualification.

## Decision

The frozen implementation, artifact roundtrip, seed barriers, and disposable
workflow are operational. Production `321M` training and `322M` fixed
validation remain sealed pending the exact confirmation
`GENERATE_M57_STUDENT_T_TRAIN_VALIDATION`.
