# Neural-HMSC Milestone 57 Fixed-Validation Terminal Result

Date: 2026-07-26

## Decision

Milestone 57 failed its one permitted production fixed validation. The
reserved `323M` through `325M` simulation/MCMC blocks and both real-data
replays remain sealed.

The preregistered stop rule is terminal for this Student-t representation.
There is no permitted scale, tail, objective, threshold, calibration, or
representation retry.

## Production Run

- LUMI job: `20272020`
- partition: `dev-g`
- scheduler state: `COMPLETED`
- elapsed time: `00:06:27`
- exit code: `0:0`
- peak batch-step memory: `1,955,316 KiB`
- run root:
  `/scratch/project_462000131/anisrahm/hmsc-hpc-runs/neural_hmsc_m57_train_validation_20272020`
- downloaded evidence:
  `/private/tmp/neural_hmsc_m57_train_validation_20272020`

The run opened only:

- paired training contexts `321000001` through `321000324`, producing 648
  response realizations;
- fixed-validation contexts `322000001` through `322000324`.

The model used the frozen 150 epochs, batch size of nine owning contexts,
learning rate `0.0005`, model seed `321900001`, and unweighted bivariate
Student-t negative log likelihood. No `323M` through `325M` seed was opened.

## Independent Validation

The downloaded bundle was validated against independently downloaded copies
of the exact immutable v0.1 and variable-v1 registries and the retained failed
Milestone 56 reference.

All independent checks passed:

- frozen protocol hashes and immutable artifact bindings;
- exact training and validation seed roles;
- exact paired-response training contract;
- overlay load and compatibility validation;
- freeze, sidecar, manifest, weight, and internal weight hashes;
- post-freeze validation checks;
- exact recomputation of all 176 gate booleans;
- exact agreement between the recomputed and recorded decision;
- absence of a reserved-evaluation artifact.

Artifact hashes:

- `freeze.json`:
  `c9cba213b67052d08103c853daff37a762f9d0fdeb6c583843df3cbd00fae1c9`
- `postfreeze_validation.json`:
  `b0deb81473c1637bf06506f671263724b1dfeb267af8da9ecf0a0ea6602d372d`
- Student-t overlay manifest:
  `fe6d49886f40c53545b72ae23af978171401fd120f72dd7f487a4088e5359b74`
- Student-t head weights:
  `425a9de7c7485ce65463242cf7eefbd1c1caf44809074256fd527be84db3252d`

## Gate Result

Of 176 frozen gates, 173 passed and three failed:

| Gate | Observed | Required |
| --- | ---: | ---: |
| geometric marginal-width ratio versus v0.1 | `0.771923` | `[0.80, 2.00]` |
| degrees-of-freedom bound fraction | `0.141399` | at most `0.10` |
| strong-effect radial-rank mean | `0.565492` | absolute error from `0.5` at most `0.05` |

The failure is not attributable to an operational or provenance defect. It
shows a statistically material combination of intervals narrower than the
frozen lower bound, excessive tail-parameter saturation, and strong-effect
joint radial-rank bias.

Several aggregate results were strong but cannot override a frozen gate:

- marginal 95% coverage: `0.957737`;
- marginal 50% coverage: `0.479856`;
- joint 95% coverage: `0.958930`;
- candidate/v0.1 location-RMSE ratio: `0.773319`;
- candidate/v0.1 normalized joint-log-score delta: `-0.911607`;
- candidate/v0.1 energy-score ratio: `0.535888`;
- candidate/v0.1 heldout Brier ratio: `0.842421`;
- candidate/v0.1 heldout log-loss ratio: `0.857968`.

These gains do not permit post-hoc widening, degrees-of-freedom clipping,
threshold changes, or a calibration-only retry.

## Consequence

`neural_hmsc_fixed_probit_student_t_v1` is retained only as a frozen negative
research artifact. It is not qualified for reserved MCMC comparison,
real-data replay, public inference, or promotion.

`neural_hmsc_v0_1` remains the qualified fixed-shape neural endpoint.
Qualified Python MCMC remains the statistical reference and the required
fallback for posterior capabilities outside the qualified neural claim.

The next roadmap step is outside neural joint-posterior development: implement
an explicit applicability decision and automatic Python-MCMC fallback at the
public inference boundary, with immutable v0.1 retained for supported neural
prediction and no further Milestone 57 tuning.
