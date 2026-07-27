# Neural-HMSC Milestone 56 Disposable Smoke

Date: 2026-07-24

Protocol: `neural_hmsc_fixed_probit_covariance_m56_v1`

Status: disposable implementation smoke passed; no production or reserved
evaluation seed was opened.

## Scope

The smoke exercised the frozen nine-feature correlation head and full
IRLS/Laplace covariance extraction around immutable `neural_hmsc_v0_1` member
`20260721`. It used:

- disposable training seeds `291000001` through `291000027`;
- disposable evaluation seeds `292000001` through `292000027`;
- one training epoch, batch size 9, and model seed `291900001`;
- 32 posterior draws per evaluation community for plumbing diagnostics.

The production ranges `211000001` through `215000324` remained unopened.

## Immutable Inputs

The run revalidated:

- preregistration SHA-256
  `d99b63da87103c3d8891cb2fab5bb7ffad30a188ed7be920950345581f8b2d4b`;
- artifact/seed audit SHA-256
  `5bb9236967afb5a2a1adc166781f4a34359a7469150aa2e19117752dd1fce29c`;
- v0.1 release content SHA-256
  `affcfe10d2f9586e97432ae07754ab829e929ffdb62f41fb71779ad5f3ed12c8`;
- bound checkpoint weights SHA-256
  `bb6e76d3ec9bc5e294ceac3051c3b2d7e5273db5053cfa5ceac676913d6265d9`;
- bound calibration SHA-256
  `595fc0796d36802002cee09b270d53162f1fce100b83aecd32476e0958a0fd94`.

## Result

All operational smoke checks passed:

- finite training;
- exact disposable seed roles;
- exact base artifact hashes;
- zero posterior-mean delta versus calibrated v0.1;
- zero marginal-scale delta versus calibrated v0.1;
- positive-definite covariance, minimum eigenvalue `0.000470678`;
- bounded correlation, maximum absolute value `0.905283`;
- exact overlay save/load roundtrip for mean, scale, and correlation.

The smoke overlay is:

- manifest SHA-256
  `d803093c13ac5f2b3bf6fe026cf471da0f82b2c8e5dd1eb36b15bbf6c54d28cf`;
- weights SHA-256
  `1148a360023277f26b4148a1cff62214365aa8f78db0f922e3e6a24ec0592380`;
- report SHA-256
  `82105823f31671d89122d4c88f55472cd7bb322d9efcc16825b97ae1b50fca96`.

The retained local report is
`/private/tmp/neural_hmsc_m56_disposable_smoke_20260724/smoke_report.json`.

## Statistical Boundary

This is not promotion evidence. With only 27 training communities and one
epoch, candidate/diagonal joint NLL ratio was `1.932640`, candidate/raw-Laplace
ratio was `0.995338`, and mean absolute Fisher-z movement was `0.005374`.
Marginal and joint coverage were also below production gates. These numbers
neither qualify nor reject the preregistered 100-epoch candidate; the smoke was
authorized only to validate implementation plumbing.

The 211M training and 212M fixed validation may run only after the exact
confirmation `GENERATE_M56_CORRELATION_TRAIN_VALIDATION`. The 213M-215M
evaluation remains independently sealed behind
`OPEN_M56_RESERVED_COVARIANCE_EVALUATION`.
